# composite_dist.py
import torch
from typing import List, Optional

class CompositeDist:
    """
    A simple joint (independent) distribution wrapper for multi-head action spaces.
    It mimics a torch.distributions API subset: log_prob, entropy, sample, mode (optional).
    For KL, you'll sum component KLs externally (old/new must both be CompositeDist
    with same structure).
    """
    def __init__(self, dists: List[torch.distributions.Distribution], kind: str = "mixed"):
        self.dists = dists
        self.kind = kind  # 'multi_discrete', 'mixed', etc.

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Expect actions concatenated along last dim for discrete/continuous blocks.
        # We split the actions per head using each head's event shape.
        idx = 0
        logps = []
        for d in self.dists:
            # Determine how many columns this head consumes
            if hasattr(d, "mean"):  # Normal-like → shape [..., act_dim]
                act_dim = d.mean.shape[-1]
                a_part = actions[..., idx:idx+act_dim]
                idx += act_dim
                lp = d.log_prob(a_part)
                if lp.dim() > 1:  # sum over action dims
                    lp = lp.sum(-1)
                logps.append(lp)
            else:
                # Categorical/Bernoulli: usually single column per head
                # but Categorical returns scalar event -> use int action
                a_part = actions[..., idx:idx+1].long()
                idx += 1
                lp = d.log_prob(a_part.squeeze(-1))
                # ensure 1D per-sample
                if lp.dim() > 1:
                    lp = lp.sum(-1)
                logps.append(lp)
        # sum over heads; return shape [batch]
        return torch.stack(logps, dim=-1).sum(-1)

    def entropy(self) -> torch.Tensor:
        ents = []
        for d in self.dists:
            e = d.entropy()
            if e.dim() > 1:
                e = e.sum(-1)
            ents.append(e)
        return torch.stack(ents, dim=-1).sum(-1)

    # Optional helpers
    @property
    def logits(self) -> Optional[torch.Tensor]:
        # Concatenate logits of categorical/bernoulli heads if present (None for pure Gaussian)
        logs = []
        for d in self.dists:
            if hasattr(d, "logits"):
                logs.append(d.logits)
        if logs:
            return torch.cat(logs, dim=-1)
        return None

    @property
    def mean(self) -> Optional[torch.Tensor]:
        mus = []
        for d in self.dists:
            if hasattr(d, "mean"):
                mus.append(d.mean)
        if mus:
            return torch.cat(mus, dim=-1)
        return None

    @property
    def stddev(self) -> Optional[torch.Tensor]:
        stds = []
        for d in self.dists:
            if hasattr(d, "stddev"):
                stds.append(d.stddev)
        if stds:
            return torch.cat(stds, dim=-1)
        return None
