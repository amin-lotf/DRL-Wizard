import copy
import importlib
import math
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from pygame.transform import grayscale
from  stable_baselines3.common.atari_wrappers import WarpFrame
from drl_wizard.algorithms.utils.frame_wrapper import wrap_env, ImageToPyTorch, BufferWrapper
from drl_wizard.backend.services.training_service.algos import create_algo, AlgoState
from drl_wizard.backend.services.training_service.environments import EnvironmentState, create_environment
from drl_wizard.common.types import ActionType, AlgoType
from drl_wizard.algorithms.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from typing import List, Optional, Dict
import re
from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from drl_wizard.configs.app_cfg import AppConfig


_KNOWN_PLUGINS = [
    "ale_py"                # Atari via ALE
    # "minigrid",               # MiniGrid
    # "procgen",                # Procgen
    # "gymnasium_robotics",     # Robotics (if installed)
    # "dm_control"            # Usually not direct; use wrappers like dm_control2gym if needed
]

def auto_register_plugins(verbose: bool = False) -> None:
    """
    Attempt to import known plugin packages and register their environments
    into Gymnasium's registry. Silently skips anything not installed.
    Safe to call multiple times.
    """
    for mod_name in _KNOWN_PLUGINS:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(gym, "register_envs"):
                gym.register_envs(mod)
                if verbose:
                    print(f"[envs] Registered environments from {mod_name}")
        except ModuleNotFoundError:
            if verbose:
                print(f"[envs] {mod_name} not installed; skipping")
        except Exception as e:
            if verbose:
                print(f"[envs] Failed to register {mod_name}: {e}")


def discover_all_algos()->List[AlgoState]:
    algos = [
        create_algo(algo_id=AlgoType.PPO, algo_name='PPO',
                     action_type=[ActionType.CONTINUOUS, ActionType.DISCRETE, ActionType.MULTI_DISCRETE]),
        create_algo(algo_id=AlgoType.TRPO, algo_name='TRPO',
                    action_type=[ActionType.CONTINUOUS, ActionType.DISCRETE, ActionType.MULTI_DISCRETE]),
        create_algo(algo_id=AlgoType.SAC, algo_name='SAC',
                     action_type=[ActionType.CONTINUOUS, ActionType.DISCRETE, ActionType.MULTI_DISCRETE]),
        create_algo(algo_id=AlgoType.DQN, algo_name='DQN',
                    action_type=[ ActionType.DISCRETE, ActionType.MULTI_DISCRETE]),
        create_algo(algo_id=AlgoType.A2C, algo_name='A2C',
                    action_type=[ActionType.CONTINUOUS, ActionType.DISCRETE, ActionType.MULTI_DISCRETE]),
    ]
    return algos

def _parse_env_name(env_id: str) -> str:
    """'Ant-v5' -> 'Ant'"""
    return re.sub(r"-v\d+$", "", env_id)

def env_exists(env_id: str) -> bool:
    """Return True if Gymnasium knows this env_id."""
    try:
        gym.spec(env_id)  # raises if not found
        return True
    except Exception:
        return False

def _infer_action_type(space) -> "ActionType":
    """Map gym spaces to your ActionType."""
    if isinstance(space, Box):
        return ActionType.CONTINUOUS
    if isinstance(space, Discrete):
        return ActionType.DISCRETE
    if isinstance(space, (MultiDiscrete, MultiBinary)):
        return ActionType.MULTI_DISCRETE
    # Fallback: treat unknown spaces as discrete (safer for menus)
    raise ValueError(f"Unknown action space: {space}")

_version_re = re.compile(r"^(?P<base>.+)-v(?P<ver>\d+)$")

def _build_latest_version_map(all_env_ids: List[str]) -> Dict[str, int]:
    latest: Dict[str, int] = {}
    for env_id in all_env_ids:
        m = _version_re.match(env_id)
        if not m:
            continue
        base = m.group("base")
        ver = int(m.group("ver"))
        prev = latest.get(base)
        if prev is None or ver > prev:
            latest[base] = ver
    return latest

def _is_outdated_env_id(env_id: str, latest_versions: Dict[str, int]) -> bool:
    m = _version_re.match(env_id)
    if not m:
        return False  # no version suffix → can't be "out of date" in this sense

    base = m.group("base")
    ver = int(m.group("ver"))
    latest = latest_versions.get(base)
    if latest is None:
        return False  # no other versions known

    return ver < latest  # e.g. CartPole-v0 while CartPole-v1 exists

def get_env_info(env_id: str, origin: str = "gymnasium") -> Optional["EnvironmentState"]:
    """
    If env exists, instantiate, detect action space, and build EnvironmentState.
    Returns None if creation fails (missing deps, etc.).
    """
    if not env_exists(env_id):
        return None

    env = None
    try:
        # disable_env_checker=True speeds up and avoids strict validations
        env = gym.make(env_id, disable_env_checker=True)
        action_type = _infer_action_type(env.action_space)
        env_name = _parse_env_name(env_id)
        return create_environment(
            env_id=env_id,
            env_name=env_name,
            origin=origin,
            supported_action=action_type,
        )
    except Exception:
        # e.g., missing atari ROMs, mujoco license, etc.
        return None
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

buffered_envs: List["EnvironmentState"] = []

def discover_all_envs(origin: str = "gymnasium") -> List["EnvironmentState"]:
    """
    Discover all loadable Gymnasium environments, including external ones
    like ALE, Minigrid, etc., but skip outdated versions that Gym warns
    about (e.g., phys2d/CartPole-v0 when v1 exists).
    """
    if buffered_envs:
        return buffered_envs

    auto_register_plugins()

    # --- Get all env ids from the Gym registry ---
    try:
        all_env_ids = list(gym.envs.registry.keys())
    except Exception:
        from gymnasium.envs.registration import registry as _registry
        all_env_ids = list(_registry.keys())

    # Build a map base_name -> latest version number
    latest_versions = _build_latest_version_map(all_env_ids)

    out: List["EnvironmentState"] = []
    for env_id in sorted(all_env_ids):
        # Skip outdated envs like phys2d/CartPole-v0 if v1 exists
        if _is_outdated_env_id(env_id, latest_versions):
            continue

        info = get_env_info(env_id, origin=origin)
        if info is not None:
            out.append(info)

    buffered_envs.extend(out)
    return out




def make_train_env(cfg:AppConfig,is_eval=False,is_render=False):

    env_id=cfg.env_id
    seed=cfg.seed if not is_eval else cfg.seed+1000
    num_envs=cfg.n_envs if not is_eval else cfg.n_eval_envs
    rescale_frames=cfg.rescale_frames
    def get_env_fn(render_mode=None):
        def init_env():
            auto_register_plugins()
            env = gym.make(env_id, render_mode=render_mode)
            # --- Detect and wrap frame-based environments ---
            obs_space = env.observation_space
            if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
                if rescale_frames:
                    env = WarpFrame(env, width=84, height=84)
                env = ImageToPyTorch(env)
                env = BufferWrapper(env, n_steps=4)

            # --- Seed and return ---
            env.reset(seed=seed)
            return env

        return init_env
    if is_render:
        envs = DummyVecEnv([get_env_fn(render_mode="rgb_array")])
        envs.seed(seed)
        return envs
    if num_envs == 1:
        envs=DummyVecEnv([get_env_fn()])
        envs.seed(seed)
        return envs
    else:
        envs = SubprocVecEnv([get_env_fn() for i in range(num_envs)])
        envs.seed(seed)
        return envs


def tensor_to_numpy(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, num_clones):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_clones)])

def check(inputs, dtype: torch.dtype, device):
    if isinstance(inputs, np.ndarray):
        return torch.as_tensor(inputs, dtype=dtype, device=device)
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(dtype=dtype, device=device)
    else:
        raise TypeError(f"Unsupported type: {type(inputs)}")


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'Discrete':
        obs_shape = (1,)
    else:
        raise NotImplementedError
    return obs_shape


def get_len_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        num_actions = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        num_actions = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        num_actions = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        num_actions = act_space.shape[0]
    elif act_space.__class__.__name__ == 'list':
        num_actions=0
        for a_space in act_space:
            if a_space.__class__.__name__ == 'Discrete':
                num_actions += 1
            elif a_space.__class__.__name__ == "MultiDiscrete":
                num_actions += a_space.shape
            elif a_space.__class__.__name__ == "Box":
                num_actions += a_space.shape[0]
            elif a_space.__class__.__name__ == "MultiBinary":
                num_actions += a_space.shape[0]
    else:  # agar
        raise NotImplementedError
    return num_actions


def get_num_actions(action_space):
    action_type = action_space.__class__.__name__
    if action_type == "Discrete":
        action_dim = action_space.n
    elif action_type == "Box":
        action_dim = action_space.shape[0]
    elif action_type == "MultiBinary":
        action_dim = action_space.shape[0]
    elif action_type == "MultiDiscrete":
        action_dim = action_space.high - action_space.low + 1
    else:  # discrete + continuous
        continuous_dim = action_space[0].shape[0]
        discrete_dim = action_space[1].n
        action_dim=+continuous_dim+discrete_dim
    return action_dim