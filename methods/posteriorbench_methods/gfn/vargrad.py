"""The VarGrad trajectory-balance loss (Richter et al., 2020; Zhang et al., 2023), an alternative to
subtrajectory balance beyond the paper (FOLLOWUPS.md).

Trajectory balance asks that log Z + log q(y) = log R(y) for every response y, with a learned
log Z. VarGrad replaces log Z by its estimate from a group of K responses to the same prompt,

    log Z^ = mean_k (log R(y_k) - log q(y_k)),

held fixed (no gradient), and the loss of the group is the sum of g(d_k) over its responses, with
the residual d_k = log Z^ + log q(y_k) - log R(y_k). With the square, g(d) = d^2, the gradient is
that of K times the variance of log R - log q over the group, whose minimum, at any distribution of
the responses, is a policy proportional to R. With Linex (`subtb.penalty`), a residual d < 0,
where the policy gives y less probability than the reward asks, is penalized more.

A response is a token sequence with its final stop token, if it has one; R(y) is then p of that
sequence times epsilon for a miss, and q(y) includes the stop token. A response at the response
limit ends there with probability one. Unlike subtrajectory balance, no prefix flows and no terms
between stop tokens enter.
"""

from __future__ import annotations

import torch

from posteriorbench_methods.gfn import subtb


def vargrad_terms(
    logq: torch.Tensor,
    log_rewards: torch.Tensor,
    *,
    loss_function: str = "square",
    loss_shape: float = 1.0,
) -> torch.Tensor:
    """The sum of g(d_k) over a group. logq, log_rewards: [K] log q(y_k) under the policy (with
    gradients) and log R(y_k) (without).

    Raises ValueError if the shapes differ or the group has fewer than two responses."""
    if logq.ndim != 1 or logq.shape != log_rewards.shape or len(logq) < 2:
        raise ValueError("a VarGrad group has at least two responses, each with log q and log R")
    log_ratio = log_rewards - logq
    log_z = log_ratio.detach().mean()
    residuals = log_z - log_ratio
    return subtb.penalty(residuals, loss_function=loss_function, shape=loss_shape).sum()
