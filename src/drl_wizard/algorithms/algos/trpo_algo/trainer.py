import torch
import torch.nn as nn
from torch import autograd

from drl_wizard.algorithms.algos.trpo_algo.network.distributions import FixedCategorical, FixedBernoulli, FixedNormal
from drl_wizard.algorithms.algos.trpo_algo.policy import Policy
from drl_wizard.algorithms.buffers.trpo_buffer import TRPOBuffer
from drl_wizard.configs import TRPOConfig
from drl_wizard.algorithms.utils import check
from drl_wizard.configs.app_cfg import AppConfig



def _flat_params(params):
    return torch.cat([p.data.view(-1) for p in params])

def _set_params(params, flat):
    idx = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

def _flat_grad(y, params, retain_graph=False, create_graph=False):
    grads = autograd.grad(y, params, retain_graph=retain_graph, create_graph=create_graph)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).view(-1) for g, p in zip(grads, params)])

def _conjugate_gradients(Avp, b, iters, tol):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)
    for _ in range(iters):
        Avp_p = Avp(p)
        alpha = rr / (torch.dot(p, Avp_p) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp_p
        rr_new = torch.dot(r, r)
        if rr_new < tol:
            break
        beta = rr_new / (rr + 1e-8)
        p = r + beta * p
        rr = rr_new
    return x

def _kl_one(old_d, new_d):
    import torch
    # Categorical
    if isinstance(old_d, FixedCategorical):
        # KL(old||new) = sum p_old * (log p_old - log p_new)
        log_p_old = old_d.logits - old_d.logits.logsumexp(dim=-1, keepdim=True)
        log_p_new = new_d.logits - new_d.logits.logsumexp(dim=-1, keepdim=True)
        p_old = log_p_old.exp()
        return (p_old * (log_p_old - log_p_new)).sum(-1)

    # Bernoulli
    if isinstance(old_d, FixedBernoulli):
        p = old_d.probs.clamp(1e-8, 1-1e-8)
        q = new_d.probs.clamp(1e-8, 1-1e-8)
        return (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).sum(-1)

    # Normal (diag)
    if isinstance(old_d, FixedNormal):
        mu0, mu1 = old_d.mean, new_d.mean
        s0, s1 = old_d.stddev, new_d.stddev
        t1 = torch.log(s1 / s0).sum(-1)
        t2 = ((s0.pow(2) + (mu0 - mu1).pow(2)) / (2.0 * s1.pow(2))).sum(-1)
        t3 = 0.5 * mu0.size(-1)
        return t1 + t2 - t3

    raise NotImplementedError("Unsupported distribution types for KL.")

def _kl_divergence(old, new):
    # Handles CompositeDist by summing head-wise KL
    if hasattr(old, "dists") and hasattr(new, "dists"):
        assert len(old.dists) == len(new.dists)
        parts = [_kl_one(od, nd) for od, nd in zip(old.dists, new.dists)]
        return torch.stack(parts, dim=-1).sum(-1)
    else:
        return _kl_one(old, new)


class Trainer:
    def __init__(self, policy: Policy, cfg: AppConfig):
        self.device = cfg.resolved_device
        self.policy = policy
        self.algo_cfg: TRPOConfig = cfg.algo_cfg

    def cal_value_loss(self, values, value_predicts_batch, return_batch):
        value_pred_clipped = value_predicts_batch + (values - value_predicts_batch).clamp(-self.algo_cfg.clip_ratio,
                                                                                          self.algo_cfg.clip_ratio)
        value_loss_clipped = (return_batch - value_pred_clipped) ** 2 / 2
        value_loss_original = (return_batch - values) ** 2 / 2
        if self.algo_cfg.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original
        mean_value_loss = value_loss.mean()
        return mean_value_loss

    def trpo_update(self, sample):
        obs, shared_obs, actions, masked_actions, returns, value_predicts, advantages, old_log_probs = sample
        obs = check(obs, torch.float32, self.device)
        critic_input = check(shared_obs, torch.float32, self.device) if shared_obs is not None else obs
        actions = check(actions, torch.float32, self.device)
        advantages = check(advantages, torch.float32, self.device)
        value_predicts = check(value_predicts, torch.float32, self.device)
        returns = check(returns, torch.float32, self.device)
        old_log_probs = check(old_log_probs, torch.float32, self.device).mean(dim=-1, keepdim=True)
        if masked_actions is not None:
            masked_actions = check(masked_actions, torch.int64, self.device)

        # ——— Distributions (old/new) ———
        with torch.no_grad():
            dist_old = self.policy.get_dist(obs, masked_actions)
        dist_new = self.policy.get_dist(obs, masked_actions)  # current params

        # ——— Surrogate and entropy (for logging) ———
        logp = dist_new.log_prob(actions)
        if logp.dim() > 1:  # handle multi-dim actions (sum log-probs)
            logp = logp.sum(-1)
        ratio = torch.exp(logp - old_log_probs.squeeze(-1))
        surr = (ratio * advantages.squeeze(-1)).mean()
        entropy = dist_new.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(-1)
        entropy = entropy.mean()

        # ——— Policy gradient g ———
        actor_params = [p for p in self.policy.actor.parameters() if p.requires_grad]
        g = _flat_grad(surr, actor_params, retain_graph=True, create_graph=True)

        # We maximize surr, CG solves Hx = g. (H is Hessian of KL)
        # Define HVP via autograd on KL
        def hvp_func(v):
            dist_new_h = self.policy.get_dist(obs, masked_actions)  # re-compute with current graph
            kl = _kl_divergence(dist_old, dist_new_h).mean()
            grad_kl = _flat_grad(kl, actor_params, retain_graph=True, create_graph=True)
            kl_v = (grad_kl * v).sum()
            hvp = _flat_grad(kl_v, actor_params, retain_graph=True, create_graph=False)
            return hvp + self.algo_cfg.damping * v

        x = _conjugate_gradients(hvp_func, g, self.algo_cfg.cg_iters, self.algo_cfg.cg_residual_tol)

        # ——— Step size to satisfy KL ≤ delta ———
        xHx = (x * hvp_func(x)).sum()
        step_scale = torch.sqrt(2.0 * self.algo_cfg.max_kl / (xHx + 1e-8))
        full_step = step_scale * x

        # ——— Line search ———
        old_params = _flat_params(actor_params)
        def set_and_eval(step):
            new_params = old_params + step
            _set_params(actor_params, new_params)

            with torch.no_grad():
                d = self.policy.get_dist(obs, masked_actions)
                logp_new = d.log_prob(actions)
                if logp_new.dim() > 1:
                    logp_new = logp_new.sum(-1)
                r = torch.exp(logp_new - old_log_probs.squeeze(-1))
                surr_new = (r * advantages.squeeze(-1)).mean()
                kl_new = _kl_divergence(dist_old, d).mean()
            return surr_new, kl_new

        accepted = False
        step = full_step
        for _ in range(self.algo_cfg.backtrack_steps):
            surr_new, kl_new = set_and_eval(step)
            if (kl_new <= self.algo_cfg.max_kl) and (surr_new >= surr):
                accepted = True
                break
            step *= self.algo_cfg.backtrack_coeff

        if not accepted:
            # revert
            _set_params(actor_params, old_params)
            # we still compute logs from the reverted model
            with torch.no_grad():
                dn = self.policy.get_dist(obs, masked_actions)
                logp_cur = dn.log_prob(actions)
                if logp_cur.dim() > 1: logp_cur = logp_cur.sum(-1)
                ratio_mean = torch.exp(logp_cur - old_log_probs.squeeze(-1)).mean().item()
            policy_loss = -surr.item()
            # Critic update still happens below
        else:
            policy_loss = -surr_new.item()
            ratio_mean = torch.exp((logp if accepted else logp) - old_log_probs.squeeze(-1)).mean().item()

        # ——— Critic/value update (same as PPO) ———
        values = self.policy.get_values(critic_input)
        value_loss = self.cal_value_loss(values, value_predicts, returns)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.algo_cfg.vf_coef).backward()
        max_grad_norm = self.algo_cfg.max_grad_norm if self.algo_cfg.max_grad_norm else 1e6
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_grad_norm)
        self.policy.critic_optimizer.step()

        # Return metrics (actor_grad_norm not meaningful for TRPO step)
        return value_loss, critic_grad_norm, torch.tensor(policy_loss, device=self.device), entropy, torch.tensor(0.0), torch.tensor(ratio_mean)

    def train(self, buffer: TRPOBuffer):
        train_info = {
            'value_loss': 0,
            'policy_loss': 0,
            'dist_entropy': 0,
            'actor_grad_norm': 0,
            'critic_grad_norm': 0,
            'ratio': 0
        }
        tot_updates = 0
        for _ in range(self.algo_cfg.num_epochs):
            for sample in buffer.feed_forward_generator():
                vloss, cgnorm, ploss, dent, agn, ratio = self.trpo_update(sample)
                train_info['value_loss'] += vloss.item()
                train_info['policy_loss'] += ploss.item()
                train_info['dist_entropy'] += dent.item()
                train_info['actor_grad_norm'] += agn.item()
                train_info['critic_grad_norm'] += cgnorm.item()
                train_info['ratio'] += ratio.item()
                tot_updates += 1
        for k in train_info:
            train_info[k] /= max(tot_updates, 1)
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
