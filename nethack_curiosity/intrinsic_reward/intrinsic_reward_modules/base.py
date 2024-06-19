from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from gymnasium.spaces import Space

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.env_info import EnvInfo


class IntrinsicRewardModule(Module, ABC):

    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__()
        self.cfg: Config = cfg
        self.obs_space: Space = obs_space

    def compute_intrinsic_rewards(
        self, mb: Union[AttrDict | TensorDict], leading_dims: int = 1
    ) -> TensorDict:
        pass

    def model_to_device(self, device: torch.device):
        pass

    def loss(self, mb: AttrDict) -> Tensor:
        pass
