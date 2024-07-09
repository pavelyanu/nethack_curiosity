import typing
from typing import List

import torch
import torch.nn as nn
from gymnasium.spaces import Space, Dict
from torch import Tensor

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from nethack_curiosity.models.nethack_models import NethackRNDEncoder
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
    return obs_space


class RNDIntrinsicRewardModule(IntrinsicRewardModule):
    def __init__(self, cfg: Config, obs_space: Space, action_space: Space):
        super().__init__(cfg, obs_space, action_space)
        self.device: torch.device = torch.device("cpu")

        self.observation_keys = cfg.observation_keys
        self.obs_space = get_observation_space(cfg, obs_space)
        self.target_encoder: Encoder
        self.target_head: nn.Module
        self.predictor_encoder: Encoder
        self.predictor_head: nn.Module
        (
            self.target_encoder,
            self.target_head,
            self.predictor_encoder,
            self.predictor_head,
        ) = self.create_networks(cfg, obs_space)

        self.returns_normalizer: RunningMeanStdInPlace = RunningMeanStdInPlace((1,))
        self.returns_normalizer = torch.jit.script(self.returns_normalizer)

        self.apply(self.initialize_weights)

        self._summaries: typing.Dict = {}
        if cfg.env_type == "nethack":
            self._summaries["max_dlvl"] = 1

    def select_encoder_type(self, cfg: Config) -> type:
        if cfg.env_type == "nethack":
            return NethackRNDEncoder
        elif cfg.env_type == "minigrid":
            return super().select_encoder_type(cfg)

    def create_networks(
        self, cfg: Config, obs_space: Space
    ) -> tuple[Encoder, nn.Module, Encoder, nn.Module]:
        encoder_type = self.select_encoder_type(cfg)
        if cfg.rnd_share_encoder:
            target_encoder = encoder_type(cfg, obs_space)
            predictor_encoder = target_encoder
        else:
            target_encoder = encoder_type(cfg, obs_space)
            predictor_encoder = encoder_type(cfg, obs_space)

        predictor_mlp_layers: List[int] = cfg.rnd_predictor_mlp_layers
        target_mlp_layers: List[int] = cfg.rnd_target_mlp_layers
        predictor_mlp_layers.insert(0, predictor_encoder.get_out_size())
        target_mlp_layers.insert(0, target_encoder.get_out_size())
        target_head_layers = []
        predictor_head_layers = []

        for i in range(1, len(target_mlp_layers)):
            target_head_layers.append(
                nn.Linear(target_mlp_layers[i - 1], target_mlp_layers[i])
            )
            target_head_layers.append(nn.ReLU())
        target_head_layers.pop()

        for i in range(1, len(predictor_mlp_layers)):
            predictor_head_layers.append(
                nn.Linear(predictor_mlp_layers[i - 1], predictor_mlp_layers[i])
            )
            predictor_head_layers.append(nn.ReLU())
        predictor_head_layers.pop()

        target_head = nn.Sequential(*target_head_layers)
        predictor_head = nn.Sequential(*predictor_head_layers)

        return target_encoder, target_head, predictor_encoder, predictor_head

    def _forward(self, td: TensorDict, leading_dims: int = 1) -> TensorDict:

        if leading_dims == 1:
            return self.compute_intrinsic_rewards_for_batch(td)

        # dims of td are [envs, time, ...]
        normalized_obs = td["normalized_obs"]
        leading = normalized_obs[self.observation_keys[0]].shape[:leading_dims]
        target_features: Tensor = torch.zeros(
            leading + (self.cfg.rnd_mlp_layers[-1],), device=self.device
        )
        predictor_features: Tensor = torch.zeros(
            leading + (self.cfg.rnd_mlp_layers[-1],), device=self.device
        )
        for e in range(0, leading[0]):
            mb_td = td[e]
            td_e = self.compute_intrinsic_rewards_for_batch(mb_td)
            target_features[e] = td_e["target_features"]
            predictor_features[e] = td_e["predictor_features"]

        intrinsic_rewards = (
            (target_features.detach() - predictor_features.detach()).pow(2).sum(dim=-1)
        )

        intrinsic_rewards = intrinsic_rewards[:, 1:]
        target_features = target_features[:, 1:]
        predictor_features = predictor_features[:, 1:]

        if self.cfg.recompute_intrinsic_loss:
            return TensorDict(
                intrinsic_rewards=intrinsic_rewards,
            )
        else:
            return TensorDict(
                intrinsic_rewards=intrinsic_rewards,
                target_features=target_features,
                predictor_features=predictor_features,
            )

    def compute_intrinsic_rewards_for_batch(self, batch: TensorDict) -> TensorDict:
        normalized_obs = batch["normalized_obs"]

        if self.cfg.rnd_blank_obs:
            for key in self.observation_keys:
                normalized_obs[key] = torch.zeros_like(normalized_obs[key])
        if self.cfg.rnd_random_obs:
            for key in self.observation_keys:
                normalized_obs[key] = torch.rand_like(normalized_obs[key])

        target_encoding = self.target_encoder(normalized_obs)
        if self.cfg.rnd_share_encoder:
            predictor_encoding = target_encoding
        else:
            predictor_encoding = self.predictor_encoder(normalized_obs)
        target_features = self.target_head(target_encoding)
        predictor_features = self.predictor_head(predictor_encoding)
        intrinsic_rewards = (
            (target_features.detach() - predictor_features.detach()).pow(2).sum(dim=1)
        )
        return TensorDict(
            intrinsic_rewards=intrinsic_rewards,
            target_features=target_features,
            predictor_features=predictor_features,
        )

    def forward(self, td: TensorDict, leading_dims: int = 1) -> TensorDict:
        if leading_dims == 1:
            return self.compute_intrinsic_rewards_for_batch(td)

        normalized_obs = td["normalized_obs"]
        if self.cfg.rnd_blank_obs:
            for key in self.observation_keys:
                normalized_obs[key] = torch.ones_like(normalized_obs[key])
        if self.cfg.rnd_random_obs:
            for key in self.observation_keys:
                normalized_obs[key] = torch.rand_like(normalized_obs[key])

        normalized_obs_shape = normalized_obs[self.observation_keys[0]].shape
        E, T = normalized_obs_shape[:2]

        reshaped_obs = TensorDict()
        keys = normalized_obs.keys()
        for key in keys:
            new_shape = (E * T,) + normalized_obs[key].shape[2:]
            reshaped_obs[key] = normalized_obs[key].view(new_shape)

        target_encoding = self.target_encoder(reshaped_obs)
        if self.cfg.rnd_share_encoder:
            predictor_encoding = target_encoding
        else:
            predictor_encoding = self.predictor_encoder(reshaped_obs)
        target_features = self.target_head(target_encoding)
        predictor_features = self.predictor_head(predictor_encoding)
        target_features = target_features.view(E, T, -1)[:, 1:]

        if self.cfg.rnd_blank_target:
            target_features = torch.ones_like(target_features)

        predictor_features = predictor_features.view(E, T, -1)[:, 1:]
        intrinsic_rewards = (
            (target_features.detach() - predictor_features.detach()).pow(2).sum(dim=-1)
        )
        if self.cfg.recompute_intrinsic_loss:
            return TensorDict(
                intrinsic_rewards=intrinsic_rewards,
            )
        else:
            return TensorDict(
                intrinsic_rewards=intrinsic_rewards,
                target_features=target_features,
                predictor_features=predictor_features,
            )

    def nethack_collect_summaries(self, mb: AttrDict):
        self._summaries["max_dlvl"] = torch.max(mb["normalized_obs"]["dlvl"]).item()

    def minigrid_collect_summaries(self, mb: AttrDict):
        pass

    def loss(self, mb: AttrDict) -> Tensor:
        if self.cfg.env_type == "nethack":
            self.nethack_collect_summaries(mb)
        elif self.cfg.env_type == "minigrid":
            self.minigrid_collect_summaries(mb)
        if self.cfg.recompute_intrinsic_loss:
            obs = mb["normalized_obs"]
            if self.cfg.rnd_blank_obs:
                for key in self.observation_keys:
                    obs[key] = torch.zeros_like(obs[key])
            if self.cfg.rnd_random_obs:
                for key in self.observation_keys:
                    obs[key] = torch.rand_like(obs[key])
            target_encoding = self.target_encoder(obs)
            if self.cfg.rnd_share_encoder:
                predictor_encoding = target_encoding
            else:
                predictor_encoding = self.predictor_encoder(obs)
            target_features = self.target_head(target_encoding)

            if self.cfg.rnd_blank_target:
                target_features = torch.ones_like(target_features)

            predictor_features = self.predictor_head(predictor_encoding)
            return (
                (target_features.detach() - predictor_features).pow(2).sum(dim=1).mean()
            )
        else:
            target_features = mb.target_features
            predictor_features = mb.predictor_features
            return (
                (target_features.detach() - predictor_features).pow(2).sum(dim=1).mean()
            )

    def summaries(self):
        s = super().summaries()
        s.update(self._summaries)
        return s

    def model_to_device(self, device: torch.device):
        self.device = device
        self.target_encoder.model_to_device(device)
        self.target_head.to(device)
        self.predictor_encoder.model_to_device(device)
        self.predictor_head.to(device)
        self.returns_normalizer.to(device)
        self.returns_normalizer = torch.jit.script(self.returns_normalizer)
        # self.predictor_head.register_full_backward_hook(hook)


def hook(*args):
    print("hooked")
