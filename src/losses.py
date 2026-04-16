"""StableMax cross-entropy loss (Prieto et al., arXiv 2501.04697).

Replaces the exponential in softmax with a piecewise linear/rational function
to avoid absorption errors that cause softmax collapse at large logit magnitudes.
"""

import torch


def _stablemax_s(x: torch.Tensor) -> torch.Tensor:
    """s(x) = x+1 if x >= 0, else 1/(1-x).

    Clamp branch inputs to avoid inf/NaN in the non-selected branch of
    torch.where (autograd evaluates both branches, and 0 * inf = NaN).
    """
    safe_neg = x.clamp(max=0.0)
    return torch.where(x >= 0, x + 1.0, 1.0 / (1.0 - safe_neg))


def stablemax_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """StableMax CE: -log(StableMax(z_y)).

    Args:
        logits: (*, V) unnormalized scores
        targets: (*,) integer class labels
        reduction: "mean", "sum", or "none"
    """
    s_vals = _stablemax_s(logits)
    s_sum = s_vals.sum(dim=-1, keepdim=True)
    probs = s_vals / s_sum
    target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_probs = torch.log(target_probs.clamp(min=1e-30))
    loss = -log_probs

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
