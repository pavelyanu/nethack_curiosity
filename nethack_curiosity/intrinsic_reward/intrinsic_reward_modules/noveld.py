from typing import Iterator, Optional, Union

import torch
import torch.nn as nn
from gymnasium.spaces import Space, Dict
from torch import Tensor

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.rnd import (
    RNDIntrinsicRewardModule,
)
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.encoder import Encoder
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config


def get_observation_space(cfg: Config, obs_space: Space) -> Dict:
    assert obs_space.__class__ == Dict
    # noinspection PyTypeChecker
    obs_space: Dict = obs_space
    for key in cfg.observation_keys:
        assert key in obs_space.keys(), f"Observation key {key} not in obs_space"
    assert (
        "visit_count" in obs_space.keys()
    ), "Observation key visit_count not in obs_space"
    return obs_space


class NovelDIntrindicRewardModule(IntrinsicRewardModule):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg, obs_space)
        self.device: torch.device = torch.device("cpu")

        self.observation_keys = cfg.observation_keys
        self.obs_space = get_observation_space(cfg, obs_space)

        self.rnd_module = RNDIntrinsicRewardModule(cfg, obs_space)

        self.returns_normalizer: RunningMeanStdInPlace = RunningMeanStdInPlace((1,))
        self.returns_normalizer = torch.jit.script(self.returns_normalizer)

    def forward(self, td: TensorDict, leading_dims: int = 1) -> TensorDict:
        if leading_dims == 1:
            return self.compute_intrinsic_reward_for_batch(td)

        # dims of td are [envs, time + 1, ...]
        # alloc output tensor with shape [envs, time]
        target_features = torch.empty(
            td["normalized_obs"]["image"].shape[0],
            td["normalized_obs"]["image"].shape[1] - 1,
            self.cfg.rnd_mlp_layers[-1],
            device=self.device,
        )
        predictor_features = torch.empty(
            td["normalized_obs"]["image"].shape[0],
            td["normalized_obs"]["image"].shape[1] - 1,
            self.cfg.rnd_mlp_layers[-1],
            device=self.device,
        )
        intrinsic_rewards = torch.empty(
            td["normalized_obs"]["image"].shape[0],
            td["normalized_obs"]["image"].shape[1] - 1,
            device=self.device,
        )
        for e in range(td["normalized_obs"]["image"].shape[0]):
            batch = td[e]
            batch_result = self.compute_intrinsic_reward_for_batch(batch)
            target_features[e] = batch_result["target_features"]
            predictor_features[e] = batch_result["predictor_features"]
            intrinsic_rewards[e] = batch_result["intrinsic_rewards"]
        return TensorDict(
            intrinsic_rewards=intrinsic_rewards,
            target_features=target_features,
            predictor_features=predictor_features,
        )

    def compute_intrinsic_reward_for_batch(self, batch: TensorDict) -> TensorDict:
        normalized_obs = batch["normalized_obs"]
        target_encoding = self.rnd_module.target_encoder(normalized_obs)
        if self.cfg.rnd_share_encoder:
            predictor_encoding = target_encoding
        else:
            predictor_encoding = self.rnd_module.predictor_encoder(normalized_obs)
        target_features = self.rnd_module.target_head(target_encoding)
        predictor_features = self.rnd_module.predictor_head(predictor_encoding)
        novelty = (
            (target_features[:-1].detach() - predictor_features.detach()[:-1])
            .pow(2)
            .sum(1)
        )
        novelty_next = (
            (target_features[1:].detach() - predictor_features.detach()[1:])
            .pow(2)
            .sum(1)
        )
        scaling_factor = 0.5  # Can be made a hyperparameter
        intrinsic_rewards = novelty_next - scaling_factor * novelty
        intrinsic_rewards = torch.clamp(intrinsic_rewards, min=0.0)
        visit_count = normalized_obs["visit_count"]
        visit_count = visit_count == 1
        if self.cfg.noveld_constant_novelty != 0.0:
            intrinsic_rewards = self.cfg.noveld_constant_novelty * visit_count[1:]
        else:
            intrinsic_rewards = intrinsic_rewards * visit_count[1:]
        return TensorDict(
            intrinsic_rewards=intrinsic_rewards,
            target_features=target_features[1:],
            predictor_features=predictor_features[1:],
        )

    def loss(self, mb: AttrDict) -> Tensor:
        target_features = mb.target_features
        predictor_features = mb.predictor_features
        return (target_features.detach() - predictor_features).pow(2).sum(dim=1).mean()

    def model_to_device(self, device: torch.device):
        self.device = device
        self.rnd_module.model_to_device(device)
        self.returns_normalizer.to(device)
        self.returns_normalizer = torch.jit.script(self.returns_normalizer)
