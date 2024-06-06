from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from sample_factory.utils.attr_dict import AttrDict

class IntrinsicRewardModule(Module, ABC):
    def get_intrinsic_rewards(self, mb: AttrDict) -> Tensor:
        pass

    def loss(self, mb: AttrDict) -> Tensor:
        pass