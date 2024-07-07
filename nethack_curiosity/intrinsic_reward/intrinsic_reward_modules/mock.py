from typing import Union, Dict

import torch
from gymnasium import Space
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)


class MockIntrinsicRewardModule(IntrinsicRewardModule):
    def __init__(self, cfg: AttrDict, obs_space: Space, action_space: Space):
        super().__init__(cfg, obs_space, action_space)
        self._summaries: Dict = {}
        if cfg.env_type == "nethack":
            self._summaries["max_dlvl"] = 1

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
        if self.cfg.env_type == "nethack":
            self.nethack_collect_summaries(mb)
        elif self.cfg.env_type == "minigrid":
            self.minigrid_collect_summaries(mb)
        return torch.zeros(torch.Size([]), device=self.device)

    def nethack_collect_summaries(self, mb: AttrDict):
        self._summaries["max_dlvl"] = torch.max(mb["normalized_obs"]["dlvl"]).item()

    def minigrid_collect_summaries(self, mb: AttrDict):
        pass

    def summaries(self):
        s = super().summaries()
        s.update(self._summaries)
        return s

    def model_to_device(self, device: torch.device):
        self.device = device
        self.returns_normalizer.to(device)
