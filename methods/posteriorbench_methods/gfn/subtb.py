"""The subtrajectory-balance loss of Hu et al. (2024) on responses that end with one of several
stop tokens or at the response limit.

A response y of N content tokens has the prefixes y_<=0, ..., y_<=N. The flow of a prefix is the
reward of the response that ends there, divided by the policy's probability of ending there:

    F(y_<=i) = R(y_<=i stop) / pi(stop | y_<=i).

Subtrajectory balance asks, for every pair of prefixes i < j, that

    log F(y_<=i) + sum_{t = i+1..j} log pi(y_t | y_<t) = log F(y_<=j),

and the loss of y is the sum of the squared differences between the two sides. With
`loss_function="linex"` (beyond the paper; FOLLOWUPS.md) the difference d, left side minus right
side, is penalized by Linex instead of its square (`penalty`).

Two things differ from a response with one end symbol. The tokenizer has K stop tokens, and a
response may end with any of them: R(y_<=i stop) and pi(stop | y_<=i) are then sums over the stop
tokens, and one more term for each prefix and each pair of stop tokens asks that the flow computed
from one stop token alone equal the flow computed from the other, which fixes how the policy
divides its probability of ending between them. And a response that reaches the response limit
ends there with probability one, so the flow of its last prefix is its reward.
"""

from __future__ import annotations

import math

import torch


def term_count(length: int, *, terminated: bool, stop_tokens: int) -> int:
    """The number of squared differences in the loss of a response of `length` tokens: one for
    each pair of its prefixes, and at each of its `length` positions one for each pair of the
    stop tokens."""
    content = length - int(terminated)
    return content * (content + 1) // 2 + length * (stop_tokens * (stop_tokens - 1) // 2)


def penalty(residual: torch.Tensor, *, loss_function: str, shape: float) -> torch.Tensor:
    """g(d) of each residual d = log F(y_<=i) + log pi(y_i+1..j) - log F(y_<=j).

    "square": d^2. "linex": 2 (exp(-a d) + a d - 1) / a^2 with a = `shape`, which equals d^2 to
    second order at 0, grows exponentially for d < 0, where the policy gives the segment less
    probability than the flows ask, and linearly for d > 0: it is zero-avoiding, like a forward
    KL, where the square treats both signs alike."""
    if loss_function == "square":
        return residual.square()
    if loss_function == "linex":
        scaled = -shape * residual
        return 2 * (torch.expm1(scaled) - scaled) / shape**2
    raise ValueError(f"unknown loss function {loss_function!r}")


def subtb_terms(
    token_logq: torch.Tensor,
    stop_logq: torch.Tensor,
    token_logp: torch.Tensor,
    stop_logp: torch.Tensor,
    prefix_hits: list[bool],
    *,
    epsilon: float,
    terminated: bool,
    loss_function: str = "square",
    loss_shape: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """The sum of the squared differences of one response, and their number (`term_count`).

    token_logq, token_logp: [L] the log-probability of each token of the response under the
        policy and under the original model; L includes the final stop token of a response that
        has one.
    stop_logq, stop_logp: [L, K] the log-probability of each of the K stop tokens at the same
        positions, that is, after each of the first L prefixes.
    prefix_hits: [N + 1] whether the text of each prefix is a hit, where N is L - 1 for a
        response that ends with a stop token (`terminated`) and L for one that reached the limit.
    epsilon: the reward of a miss is epsilon times its probability under the original model.
    loss_function, loss_shape: the penalty of a difference between prefixes (`penalty`); the
        differences between the flows of two stop tokens, which have no direction, are squared.

    Raises ValueError if the shapes do not fit together or epsilon is not in (0, 1)."""
    length = len(token_logq)
    content_length = length - int(terminated)
    if (
        length < 1
        or len(token_logp) != length
        or stop_logq.shape != stop_logp.shape
        or stop_logq.ndim != 2
        or stop_logq.shape[0] != length
        or len(prefix_hits) != content_length + 1
        or not 0 < epsilon < 1
    ):
        raise ValueError(
            "the scores of a response do not fit together, or epsilon is not in (0, 1)"
        )
    zero = token_logq.new_zeros(1)
    q_prefix = torch.cat((zero, token_logq[:content_length].cumsum(0)))
    p_prefix = torch.cat((zero, token_logp[:content_length].cumsum(0)))
    log_hit_factor = token_logq.new_tensor(
        [0.0 if hit else math.log(epsilon) for hit in prefix_hits]
    )
    # The prefixes after which the response could have ended with a stop token: all of them, but
    # for a response at the limit not the last one.
    endings = content_length + int(terminated)
    log_rewards = p_prefix[:endings, None] + stop_logp[:endings] + log_hit_factor[:endings, None]
    log_flow = torch.logsumexp(log_rewards, 1) - torch.logsumexp(stop_logq[:endings], 1)
    if not terminated:
        log_flow = torch.cat((log_flow, p_prefix[-1:] + log_hit_factor[-1:]))
    # The condition between prefixes i and j is a_i = a_j for a = log F - log pi(prefix). The sum
    # of (a_i - a_j)^2 over all pairs is n times the sum of squared deviations from the mean, so
    # no [n, n] tensor is built; subtracting a_0 first avoids cancellation in the mean.
    balance = log_flow - q_prefix
    states = len(balance)
    count = states * (states - 1) // 2
    if loss_function == "square":
        centered = balance - balance[0]
        total = states * (centered - centered.mean()).square().sum()
    else:
        # d_ij = a_i - a_j for every i < j.
        first, second = torch.triu_indices(states, states, offset=1, device=balance.device)
        residuals = balance[first] - balance[second]
        total = penalty(residuals, loss_function=loss_function, shape=loss_shape).sum()
    stop_flow = log_rewards - stop_logq[:endings]
    for first in range(stop_logq.shape[1]):
        for second in range(first + 1, stop_logq.shape[1]):
            total = total + (stop_flow[:, first] - stop_flow[:, second]).square().sum()
            count += endings
    if count == 0:
        return total * 0, 0
    return total, count
