import torch
from torch import Tensor

from sample_factory.utils.attr_dict import AttrDict

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)


class MockIntrinsicRewardModule(IntrinsicRewardModule):
    def get_intrinsic_rewards(self, mb: AttrDict) -> Tensor:
        return mb.rewards_cpu.new_zeros(mb.rewards_cpu.size())

    def loss(self, mb: AttrDict) -> Tensor:
        return torch.zeros(1, device=mb.rewards_cpu.device)
