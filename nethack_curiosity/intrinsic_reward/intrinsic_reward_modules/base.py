from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from gymnasium.spaces import Space

from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.env_info import EnvInfo


class IntrinsicRewardModule(Module, ABC):

    def __init__(self, cfg: Config, obs_space: Space, action_space: Space):
        super().__init__()
        self.device: torch.device = torch.device("cpu")
        self.cfg: Config = cfg
        self.obs_space: Space = obs_space
        self.action_space: Space = action_space
        self.returns_normalizer: RunningMeanStdInPlace = RunningMeanStdInPlace((1,))

    def forward(
        self, mb: Union[AttrDict | TensorDict], leading_dims: int = 1
    ) -> TensorDict:
        pass

    def model_to_device(self, device):
        self.device = device
        for module in self.children():
            if hasattr(module, "model_to_device"):
                module.model_to_device(device)
            else:
                module.to(device)

    def loss(self, mb: AttrDict) -> Tensor:
        pass

    def select_encoder_type(self, cfg: Config) -> type:
        if cfg.env_type == "minigrid":
            from nethack_curiosity.models.minigrid_models import (
                MinigridIntrinsicRewardEncoder,
            )

            return MinigridIntrinsicRewardEncoder
        else:
            raise NotImplementedError(f"Unknown env type: {cfg.env_type}")
