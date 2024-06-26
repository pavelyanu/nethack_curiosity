from typing import Optional

import torch
from torch import Tensor


@torch.jit.script
def gae_advantages_single_batch(
    rewards: Tensor, dones: Tensor, values: Tensor, valids: Tensor, γ: float, λ: float
) -> Tensor:
    assert len(rewards) == len(dones)
    assert len(rewards) + 1 == len(values)

    deltas = (rewards - values[:-1]) * valids[:-1] + (1 - dones) * (
        γ * values[1:] * valids[1:]
    )

    advantages = calculate_discounted_sum_torch_single_batch(
        deltas, dones, valids[:-1], γ * λ
    )

    return advantages


@torch.jit.script
def calculate_discounted_sum_torch_single_batch(
    x: Tensor,
    dones: Tensor,
    valids: Tensor,
    discount: float,
    x_last: Optional[Tensor] = None,
) -> Tensor:
    if x_last is None:
        x_last = x[-1].clone().fill_(0.0)

    cumulative = x_last

    discounted_sum = torch.zeros_like(x)
    i = len(x) - 1
    while i >= 0:
        discount_valid = discount * valids[i] + (1 - valids[i])
        cumulative = x[i] + discount_valid * cumulative * (1.0 - dones[i])
        discounted_sum[i] = cumulative
        i -= 1

    return discounted_sum
