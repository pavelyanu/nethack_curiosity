from typing import Union

import torch
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)


class MockIntrinsicRewardModule(IntrinsicRewardModule):

    def forward(
        self, mb: Union[AttrDict | TensorDict], leading_dims: int = 1
    ) -> TensorDict:
        if isinstance(mb, AttrDict):
            ir_rewards = mb.rewards.new_zeros(mb.rewards.size(), device=self.device)
        else:
            ir_rewards = mb["rewards"].new_zeros(
                mb["rewards"].size(), device=self.device
            )
        return TensorDict(intrinsic_rewards=ir_rewards)

    def loss(self, mb: AttrDict) -> Tensor:
        return torch.zeros(torch.Size([]), device=self.device)

    def model_to_device(self, device: torch.device):
        self.device = device
        self.returns_normalizer.to(device)
