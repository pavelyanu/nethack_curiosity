from __future__ import annotations

from typing import Dict, List, Tuple

from torch import Tensor, nn

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sample_factory.model.actor_critic import ActorCritic


class IntrinsicRewardActorCriticSharedWeights(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())

        self.decoder = model_factory.make_model_decoder_func(
            cfg, self.core.get_out_size()
        )
        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear = nn.Linear(decoder_out_size, 1)

        #################
        # MY CODE BLOCK #
        #################
        self.intrinsic_critic_linear = nn.Linear(decoder_out_size, 1)
        #################

        self.action_parameterization = self.get_action_parameterization(
            decoder_out_size
        )

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()

        #################
        # MY CODE BLOCK #
        #################
        intrinsic_values = self.intrinsic_critic_linear(decoder_output).squeeze()
        #################

        result = TensorDict(values=values, intrinsic_values=intrinsic_values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = (
            self.action_parameterization(decoder_output)
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result


def intrinsic_policy_output_shapes(
    num_actions, num_action_distribution_parameters
) -> List[Tuple[str, List]]:
    # policy outputs, this matches the expected output of the actor-critic
    policy_outputs = [
        ("actions", [num_actions]),
        ("action_logits", [num_action_distribution_parameters]),
        ("log_prob_actions", []),
        ("values", []),
        ("intrinsic_values", []),
        ("policy_version", []),
    ]
    return policy_outputs


def make_intrinsic_reward_actor_critic(
    cfg: Config, obs_space: ObsSpace, action_space: ActionSpace
) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()

    if cfg.actor_critic_share_weights:
        return IntrinsicRewardActorCriticSharedWeights(
            model_factory, obs_space, action_space, cfg
        )
    else:
        raise NotImplementedError("Separate weights not supported for intrinsic reward")
