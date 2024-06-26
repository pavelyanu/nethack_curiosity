import pytest
import torch
from nethack_curiosity.intrinsic_reward.intrinsic_reward_algo.utils import (
    gae_advantages_single_batch,
    calculate_discounted_sum_torch_single_batch,
)
from sample_factory.algo.utils.rl_utils import (
    gae_advantages,
    calculate_discounted_sum_torch,
)


@pytest.mark.parametrize("E", [1, 10])
@pytest.mark.parametrize("T", [1, 10])
def test_gae_advantages_single_batch(E, T):

    rewards = torch.rand(E, T)
    dones = torch.randint(0, 2, (E, T))
    values = torch.rand(E, T + 1)
    valids = torch.ones(E, T + 1)
    γ = 0.99
    λ = 0.95

    advantages = gae_advantages(rewards, dones, values, valids, γ, λ)
    single_batch_advantages = torch.zeros_like(advantages)
    for e in range(E):
        single_batch_advantages[e] = gae_advantages_single_batch(
            rewards[e], dones[e], values[e], valids[e], γ, λ
        )

    assert torch.allclose(advantages, single_batch_advantages, atol=1e-7)
