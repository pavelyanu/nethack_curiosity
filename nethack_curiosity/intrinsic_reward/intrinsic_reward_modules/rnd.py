from typing import Iterator, Optional, Union

import torch
from gymnasium.spaces import Space
from torch import Tensor
from torch.nn import Parameter

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.encoder import Encoder
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config


class RNDIntrinsicRewardModule(IntrinsicRewardModule):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg, obs_space)
        self.random_encoder: Encoder
        self.target_encoder: Encoder
        encoder_type: type
        if cfg.env_type == "minigrid":
            from nethack_curiosity.models.minigrid_models import (
                MinigridIntrinsicRewardEncoder,
            )

            encoder_type = MinigridIntrinsicRewardEncoder
        else:
            raise NotImplementedError(f"Unknown env type: {cfg.env_type}")

        self.target_encoder = encoder_type(cfg, obs_space)
        self.predictor_encoder = encoder_type(cfg, obs_space)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def compute_intrinsic_rewards(
        self, mb: Union[AttrDict | TensorDict], leading_dims: int = 1
    ) -> TensorDict:
        normalized_obs: Tensor
        rewards: Tensor

        if isinstance(mb, AttrDict):
            normalized_obs = mb.normalized_obs
            rewards = mb.rewards
        else:
            normalized_obs = mb["normalized_obs"]
            rewards = mb["rewards"]

        reward_og_shape = rewards.shape

        if leading_dims > 1:
            normalized_obs = normalized_obs[:, :-1]
            normalized_obs = self.flatten(normalized_obs, leading_dims)

        target_features = self.target_encoder(normalized_obs)
        predictor_features = self.predictor_encoder(normalized_obs)
        intrinsic_rewards = (target_features - predictor_features).pow(2).sum(dim=1)

        if leading_dims > 1:
            intrinsic_rewards = intrinsic_rewards.view(*reward_og_shape)
        return TensorDict(intrinsic_rewards=intrinsic_rewards)

    def loss(self, mb: AttrDict) -> Tensor:
        intrinsic_rewards = mb.intrinsic_rewards
        return intrinsic_rewards.mean()

    def model_to_device(self, device: torch.device):
        self.target_encoder.model_to_device(device)
        self.predictor_encoder.model_to_device(device)

    def flatten(
        self, x: Union[TensorDict | Tensor], leading_dims: int
    ) -> Union[TensorDict | Tensor]:
        if isinstance(x, TensorDict):
            for key, value in x.items():
                x[key] = self.flatten(value, leading_dims)
            return x
        else:
            return torch.flatten(x, 0, leading_dims - 1)
