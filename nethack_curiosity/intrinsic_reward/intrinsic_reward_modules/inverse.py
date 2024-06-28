from typing import List

import torch
import torch.nn as nn
from gymnasium.spaces import Space, Dict, Discrete
from torch import Tensor

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
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


def get_action_space(cfg: Config, action_space: Space) -> Discrete:
    assert action_space.__class__ == Discrete
    # noinspection PyTypeChecker
    action_space: Discrete = action_space
    return action_space


class InverseModelIntrinsicRewardModule(IntrinsicRewardModule):
    def __init__(self, cfg: Config, obs_space: Space, action_space: Space):
        super().__init__(cfg, obs_space, action_space)
        self.device: torch.device = torch.device("cpu")

        self.observation_keys = cfg.observation_keys
        self.obs_space: Dict = get_observation_space(cfg, obs_space)
        self.action_space: Discrete = get_action_space(cfg, action_space)

        self.returns_normalizer: RunningMeanStdInPlace = RunningMeanStdInPlace((1,))
        self.returns_normalizer = torch.jit.script(self.returns_normalizer)

        self.state_encoder: Encoder
        self.forward_dynamic_model: nn.Module
        self.inverse_dynamic_model: nn.Module
        (
            self.state_encoder,
            self.forward_dynamic_model,
            self.inverse_dynamic_model,
        ) = self.create_networks(cfg, self.obs_space, self.action_space)

    def create_networks(
        self, cfg: Config, obs_space: Dict, action_space: Discrete
    ) -> tuple[Encoder, nn.Module, nn.Module]:
        encoder_type = self.select_encoder_type(cfg)
        state_encoder = encoder_type(cfg, obs_space)
        forward_dynamic_model = nn.Sequential(
            nn.Linear(state_encoder.get_out_size() + action_space.n, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_encoder.get_out_size()),
        )
        inverse_dynamic_model = nn.Sequential(
            nn.Linear(state_encoder.get_out_size() * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n),
        )
        return state_encoder, forward_dynamic_model, inverse_dynamic_model

    def forward(self, td: TensorDict, leading_dims: int = 1) -> TensorDict:
        if leading_dims == 1:
            return self.compute_intrinsic_reward_for_batch(td)

        intrinsic_rewards = torch.empty(
            td["normalized_obs"]["visit_count"].shape[0],
            td["normalized_obs"]["visit_count"].shape[1] - 1,
            device=self.device,
        )
        encoding_current = torch.empty(
            td["normalized_obs"]["visit_count"].shape[0],
            td["normalized_obs"]["visit_count"].shape[1] - 1,
            self.state_encoder.get_out_size(),
            device=self.device,
        )
        encoding_next = torch.empty(
            td["normalized_obs"]["visit_count"].shape[0],
            td["normalized_obs"]["visit_count"].shape[1] - 1,
            self.state_encoder.get_out_size(),
            device=self.device,
        )
        forward_prediction = torch.empty(
            td["normalized_obs"]["visit_count"].shape[0],
            td["normalized_obs"]["visit_count"].shape[1] - 1,
            self.state_encoder.get_out_size(),
            device=self.device,
        )
        inverse_prediction = torch.empty(
            td["normalized_obs"]["visit_count"].shape[0],
            td["normalized_obs"]["visit_count"].shape[1] - 1,
            self.action_space.n,
            device=self.device,
        )
        for e in range(td["normalized_obs"]["visit_count"].shape[0]):
            batch = td[e]
            batch_result = self.compute_intrinsic_reward_for_batch(batch)
            intrinsic_rewards[e] = batch_result["intrinsic_rewards"]
            encoding_current[e] = batch_result["encoding_current"]
            encoding_next[e] = batch_result["encoding_next"]
            forward_prediction[e] = batch_result["forward_prediction"]
            inverse_prediction[e] = batch_result["inverse_prediction"]
        return TensorDict(
            intrinsic_rewards=intrinsic_rewards,
            encoding_current=encoding_current,
            encoding_next=encoding_next,
            forward_prediction=forward_prediction,
            inverse_prediction=inverse_prediction,
        )

    def compute_intrinsic_reward_for_batch(self, batch: TensorDict) -> TensorDict:
        normalized_obs = batch["normalized_obs"]
        visit_count = normalized_obs["visit_count"]
        encoding = self.state_encoder(normalized_obs)
        encoding_current = encoding[:-1]
        encoding_next = encoding[1:]

        if self.cfg.inverse_action_mode == "onehot":
            actions = batch["actions"]
            assert torch.all(actions == actions.long())
            actions = actions.long()
            actions = torch.nn.functional.one_hot(actions, self.action_space.n).float()
            actions = torch.squeeze(actions, dim=1)
        elif self.cfg.inverse_action_mode == "logits":
            actions = batch["action_logits"]
        elif self.cfg.inverse_action_mode == "logprobs":
            actions = batch["log_prob_actions"]
        else:
            raise NotImplementedError(
                f"Unknown inverse action mode: {self.cfg.inverse_action_mode}"
            )

        forward_input = torch.cat([encoding_current, actions], dim=-1)
        forward_prediction = self.forward_dynamic_model(forward_input)
        inverse_input = torch.cat([encoding_current, encoding_next], dim=-1)
        inverse_prediction = self.inverse_dynamic_model(inverse_input)

        if self.cfg.inverse_wiring == "icm":
            intrinsic_rewards = (forward_prediction - encoding_next).pow(2).sum(dim=-1)
        elif self.cfg.inverse_wiring == "ride":
            intrinsic_rewards = (encoding_next - encoding_current).pow(2).sum(dim=-1)
        else:
            raise NotImplementedError(
                f"Unknown inverse wiring: {self.cfg.inverse_wiring}"
            )

        if self.cfg.visit_count_weighting == "inverse_sqrt":
            intrinsic_rewards /= torch.sqrt(visit_count[1:])
        elif self.cfg.visit_count_weighting == "novel":
            intrinsic_rewards *= visit_count[1:] == 1
        else:
            assert (
                self.cfg.visit_count_weighting == "none"
                f"Unknown visit count weighting scheme: {self.cfg.visit_count_weighting}"
            )

        return TensorDict(
            intrinsic_rewards=intrinsic_rewards,
            encoding_current=encoding_current,
            encoding_next=encoding_next,
            forward_prediction=forward_prediction,
            inverse_prediction=inverse_prediction,
        )

    def loss(self, mb: AttrDict) -> Tensor:
        if not self.cfg.recompute_intrinsic_loss:
            raise NotImplementedError(
                "Not recomputing intrinsic loss is not implemented yet"
            )
        else:
            batch = TensorDict(
                normalized_obs=mb["normalized_obs"],
                actions=mb["actions"][:-1],
                action_logits=mb["action_logits"][:-1],
                action_log_probs=mb["log_prob_actions"][:-1],
            )
            batch_result = self.compute_intrinsic_reward_for_batch(batch)
            encoding_current = batch_result["encoding_current"]
            encoding_next = batch_result["encoding_next"]
            forward_prediction = batch_result["forward_prediction"]
            inverse_prediction = batch_result["inverse_prediction"]
            actions = torch.squeeze(batch["actions"], dim=1)
            assert torch.all(actions == actions.long())
            actions = actions.long()
            if self.cfg.inverse_wiring == "icm":
                forward_loss = (forward_prediction - encoding_next).pow(2).sum(dim=-1)
                forward_loss = torch.mean(forward_loss, dim=-1)
                inverse_losss = torch.nn.functional.cross_entropy(
                    inverse_prediction, actions
                )
            elif self.cfg.inverse_wiring == "ride":
                forward_loss = (encoding_next - encoding_current).pow(2).sum(dim=-1)
                forward_loss = torch.mean(forward_loss, dim=-1)
                inverse_losss = torch.nn.functional.cross_entropy(
                    inverse_prediction, actions
                )
            else:
                raise NotImplementedError(
                    f"Unknown inverse wiring: {self.cfg.inverse_wiring}"
                )
            return (
                self.cfg.inverse_loss_weight * inverse_losss
                + self.cfg.forward_loss_weight * forward_loss
            )
