from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.env_info import EnvInfo


class IntrinsicRewardModule(Module, ABC):

    def __init__(self, cfg: Config, env_info: EnvInfo):
        super().__init__()
        self.cfg = cfg
        self.env_info = env_info

    def get_intrinsic_rewards(
        self,
        mb: AttrDict,
    ) -> Tensor:
        pass

    def model_to_device(self, device: torch.device):
        pass

    def loss(self, mb: AttrDict) -> Tensor:
        pass
