from typing import Union

import torch
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)


class MockIntrinsicRewardModule(IntrinsicRewardModule):
    def compute_intrinsic_rewards(
        self, mb: Union[AttrDict | TensorDict], leading_dims: int = 1
    ) -> TensorDict:
        if isinstance(mb, AttrDict):
            ir_rewards = mb.rewards.new_zeros(mb.rewards.size())
        else:
            ir_rewards = mb["rewards"].new_zeros(mb["rewards"].size())
        return TensorDict(intrinsic_rewards=ir_rewards)

    def loss(self, mb: AttrDict) -> Tensor:
        return torch.zeros(torch.Size([]), device=mb.rewards_cpu.device)
