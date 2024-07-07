from abc import ABC
from typing import Union, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from gymnasium.spaces import Space

from sample_factory.algo.utils.running_mean_std import (
    RunningMeanStdInPlace,
    running_mean_std_summaries,
)
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config

from nethack_curiosity.models.nethack_models import NethackIntrinsicRewardEncoder
from nethack_curiosity.models.minigrid_models import MinigridIntrinsicRewardEncoder


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
        mapping = {
            "nethack": NethackIntrinsicRewardEncoder,
            "minigrid": MinigridIntrinsicRewardEncoder,
        }
        if cfg.env_type not in mapping:
            raise NotImplementedError(
                f"There is no encoder for env type: {cfg.encoder_type}"
            )
        return mapping[cfg.env_type]

    def summaries(self) -> Dict:
        # Can add more summaries here, like weights statistics
        s = {}
        if self.returns_normalizer is not None:
            for k, v in running_mean_std_summaries(self.returns_normalizer).items():
                s[f"intrinsic_returns_{k}"] = v
        return s

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain

        if hasattr(layer, "bias") and isinstance(
            layer.bias, torch.nn.parameter.Parameter
        ):
            layer.bias.data.fill_(0)

        if self.cfg.policy_initialization == "orthogonal":
            if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            else:
                # LSTMs and GRUs initialize themselves
                # should we use orthogonal/xavier for LSTM cells as well?
                # I never noticed much difference between different initialization schemes, and here it seems safer to
                # go with default initialization,
                pass
        elif self.cfg.policy_initialization == "xavier_uniform":
            if type(layer) is nn.Conv2d or type(layer) is nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
            else:
                pass
        elif self.cfg.policy_initialization == "torch_default":
            # do nothing
            pass
