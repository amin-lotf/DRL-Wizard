"""
Lightweight vectorized env wrappers for **Gymnasium** (single-agent),
with a sync (DummyVecEnv) and async (SubprocVecEnv) implementation.

Key points
- Correct Gymnasium API handling: `reset()` -> (obs, info), `step()` -> (obs, reward, terminated, truncated, info)
- `done = terminated or truncated` normalization in workers and dummy path
- Robust stacking: falls back to `dtype=object` if shapes are ragged (e.g., dict or variable-sized obs)
- Safe close: handles pending waits and broken pipes
- Minimal rendering contract: only `mode="rgb_array"` returns frames (no human viewer to avoid gym-classic-control dependency)
- Utility commands: `seed`, `set_attr`, `get_attr`, `call`

Later extensions you can plug in easily:
- shared_obs / available_actions (centralized critic)
- choose-reset hooks for curriculum
- multi-agent shims (PettingZoo wrapper) that adapt to the same pipe protocol

Tested on Gymnasium 0.29+ style API.
"""

import os
import time
import typing as T
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe

import numpy as np
from gymnasium.spaces import Discrete

from drl_wizard.common.utils import squeeze_scalar


# -------------------------------
# Utilities
# -------------------------------

class CloudpickleWrapper:
    """Use cloudpickle to serialize callables/env_fns across processes."""

    def __init__(self, x: T.Callable[[], T.Any]):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import cloudpickle
        self.x = cloudpickle.loads(ob)


def _safe_stack(seq: T.Sequence[T.Any]) -> np.ndarray:
    """Stack sequence into a np.array; if shapes don't match, return object array."""
    try:
        return np.stack(seq)
    except Exception:
        return np.array(seq, dtype=object)


# -------------------------------
# Base VecEnv
# -------------------------------

class VecEnv(ABC):
    """Abstract vectorized environment (single-agent Gymnasium API)."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, num_envs: int, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False
        self.waiting = False

    @abstractmethod
    def reset(self, **kwargs):
        """Reset all envs. Returns stacked observations and a list of infos.

        Gymnasium API: each env returns (obs, info); we return (obs_batch, infos_list).
        """
        ...

    @abstractmethod
    def step_async(self, actions):
        ...

    @abstractmethod
    def step_wait(self):
        """Returns obs, rewards, dones, infos (Gymnasium done = terminated or truncated)."""
        ...

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        self.closed = True

    # Optional helpers
    def seed(self, seeds: T.Sequence[int] | int | None = None):
        """Implement in subclasses to propagate seeds to workers."""
        raise NotImplementedError

    def set_attr(self, name: str, value: T.Any):
        raise NotImplementedError

    def get_attr(self, name: str):
        raise NotImplementedError

    def call(self, name: str, *args, **kwargs):
        raise NotImplementedError

    # Rendering contract: only rgb_array returns frames
    def render(self, mode: str = "rgb_array"):
        raise NotImplementedError


# -------------------------------
# Subprocess worker
# -------------------------------

def _worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    need_squeeze_action = isinstance(env.action_space, Discrete)
    need_unsqueeze_obs = isinstance(env.observation_space, Discrete)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset(**(data or {}))
                remote.send((obs, info))
            elif cmd == "step":
                action = data
                if need_squeeze_action:
                    action = squeeze_scalar(action)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                if done:
                    old_obs = obs
                    old_info= info
                    obs, info = env.reset()
                    info["final_obs"] = old_obs
                    info["final_info"] = old_info
                # If episode ended, some Gymnasium envs put final_obs in info; no auto-reset here.
                remote.send((obs, reward, done, info))
            elif cmd == "render":
                frame = env.render(mode="rgb_array")
                remote.send(frame)
            elif cmd == "close":
                try:
                    env.close()
                finally:
                    remote.close()
                break
            elif cmd == "seed":
                env.reset(seed=data)
                remote.send(None)
            elif cmd == "set_attr":
                setattr(env, data["name"], data["value"])
                remote.send(None)
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "call":
                fn = getattr(env, data["name"])
                remote.send(fn(*data.get("args", ()), **data.get("kwargs", {})))
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(cmd)
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass


# -------------------------------
# SubprocVecEnv (async)
# -------------------------------

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns: T.Sequence[T.Callable[[], T.Any]], daemon: bool = True):
        assert len(env_fns) > 0, "Need at least one env_fn"
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in env_fns])
        self.ps = [Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(fn)))
                   for (work_remote, remote, fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = daemon
            p.start()
        for work_remote in self.work_remotes:
            work_remote.close()

        # Get spaces from the first env
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)

    # --- API ---
    def reset(self, **kwargs):
        for r in self.remotes:
            r.send(("reset", kwargs))
        results = [r.recv() for r in self.remotes]
        obs, infos = zip(*results)
        return _safe_stack(obs), list(infos)

    def step_async(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        self.waiting = True

    def step_wait(self):
        results = []
        for r in self.remotes:
            try:
                results.append(r.recv())
            except EOFError:
                # Worker died; surface a clear error
                raise RuntimeError("Subproc worker terminated unexpectedly.")
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        return _safe_stack(obs), np.asarray(rews, dtype=np.float32), np.asarray(dones, dtype=bool), list(infos)

    def close(self):
        if self.closed:
            return
        # Drain any pending results to avoid deadlocks
        if self.waiting:
            for r in self.remotes:
                if r.poll(0.05):
                    try:
                        _ = r.recv()
                    except EOFError:
                        pass
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.ps:
            p.join(timeout=1.0)
        self.closed = True
        super().close()

    # --- helpers ---
    def seed(self, seeds: T.Sequence[int] | int | None = None):
        if seeds is None:
            seeds = [None] * self.num_envs
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        for r, s in zip(self.remotes, seeds):
            r.send(("seed", s))
        for r in self.remotes:
            r.recv()

    def set_attr(self, name: str, value: T.Any):
        for r in self.remotes:
            r.send(("set_attr", {"name": name, "value": value}))
        for r in self.remotes:
            r.recv()

    def get_attr(self, name: str):
        for r in self.remotes:
            r.send(("get_attr", name))
        return [r.recv() for r in self.remotes]

    def call(self, name: str, *args, **kwargs):
        for r in self.remotes:
            r.send(("call", {"name": name, "args": args, "kwargs": kwargs}))
        return [r.recv() for r in self.remotes]

    # Rendering contract: only rgb_array supported (no blocking human viewer)
    def render(self, mode: str = "rgb_array"):
        assert mode == "rgb_array", "Only mode='rgb_array' is supported in vectorized render"
        for r in self.remotes:
            r.send(("render", None))
        frames = [r.recv() for r in self.remotes]
        return _safe_stack(frames)


# -------------------------------
# DummyVecEnv (sync)
# -------------------------------

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns: T.Sequence[T.Callable[[], T.Any]]):
        assert len(env_fns) > 0, "Need at least one env_fn"
        self.envs = [fn() for fn in env_fns]
        env0 = self.envs[0]
        super().__init__(len(self.envs), env0.observation_space, env0.action_space)
        self._pending_actions = None
        self.need_squeeze = isinstance(env0.action_space, Discrete)

    def reset(self, **kwargs):
        results = [env.reset(**kwargs) for env in self.envs]
        obs, infos = zip(*results)
        return _safe_stack(obs), list(infos)

    def step_async(self, actions):
        self._pending_actions = actions

    def step_wait(self):
        assert self._pending_actions is not None, "Must call step_async(actions) before step_wait()"
        results = [env.step(squeeze_scalar(a) if self.need_squeeze else a)  for env, a in zip(self.envs, self._pending_actions)]
        self._pending_actions = None
        observations, rews, terms, truncs, infos = zip(*results)
        observations=_safe_stack(observations)
        rews=np.asarray(rews, dtype=np.float32)
        infos = list(infos)
        dones=[]
        for env_idx,(t,u) in enumerate(zip(terms,truncs)):
            if t or u:
                final_obs, final_info = observations[env_idx], infos[env_idx]
                obs,infor = self.envs[env_idx].reset()
                observations[env_idx] = obs
                infos[env_idx] = infor
                infos[env_idx]["final_obs"] = final_obs
                infos[env_idx]["final_info"] = final_info
                dones.append(True)
            else:
                dones.append(False)
        dones=np.asarray(dones, dtype=bool)
        return observations,rews, dones, infos

    def close(self):
        if self.closed:
            return
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
        self.closed = True
        super().close()

    def seed(self, seeds: T.Sequence[int] | int | None = None):
        if seeds is None:
            seeds = [None] * self.num_envs
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        for env, s in zip(self.envs, seeds):
            env.reset(seed=s)

    def set_attr(self, name: str, value: T.Any):
        for env in self.envs:
            setattr(env, name, value)

    def get_attr(self, name: str):
        return [getattr(env, name) for env in self.envs]

    def call(self, name: str, *args, **kwargs):
        return [getattr(env, name)(*args, **kwargs) for env in self.envs]

    def render(self, mode: str = "rgb_array"):
        # assert mode == "rgb_array", "Only mode='rgb_array' is supported in vectorized render"
        frames = [env.render() for env in self.envs]
        return _safe_stack(frames)


# -------------------------------
# Example usage (remove in prod)
# -------------------------------
if __name__ == "__main__":
    import gymnasium as gym

    def make_env():
        return gym.make("CartPole-v1")

    print("\nDummyVecEnv quick test")
    v = DummyVecEnv([make_env for _ in range(4)])
    obs, infos = v.reset()
    acts = [v.action_space.sample() for _ in range(v.num_envs)]
    v.step_async(acts)
    obs, rew, done, infos = v.step_wait()
    print("obs shape:", np.shape(obs), "rew:", rew, "done:", done)
    v.close()

    print("\nSubprocVecEnv quick test")
    v = SubprocVecEnv([make_env for _ in range(4)])
    v.seed(0)
    obs, infos = v.reset()
    acts = [v.action_space.sample() for _ in range(v.num_envs)]
    v.step_async(acts)
    obs, rew, done, infos = v.step_wait()
    print("obs shape:", np.shape(obs), "rew:", rew, "done:", done)
    v.close()
