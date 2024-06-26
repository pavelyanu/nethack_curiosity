from typing import Dict, Tuple, Optional

import numpy as np

import torch
from torch import Tensor

from sample_factory.algo.learning.rnn_utils import (
    build_rnn_inputs,
    build_core_out_from_seq,
)
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.optimizers import Lamb
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import to_scalar, synchronize, masked_select
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.timing import Timing
from sample_factory.algo.utils.action_distributions import (
    is_continuous_action_space,
    get_action_distribution,
)
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import (
    ActionDistribution,
    Config,
    PolicyID,
    InitModelData,
)
from sample_factory.utils.utils import log

from sample_factory.algo.learning.learner import (
    Learner,
    get_lr_scheduler,
    model_initialization_data,
)

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.make_intrinsic_reward_module import (
    make_intrinsic_reward_module,
)


class IntrinsicRewardLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.ir_module: Optional[IntrinsicRewardModule] = None
        self.ir_weight = cfg.intrinsic_reward_weight
        self.timing = Timing(name=f"IntrinsicRewardLearner {policy_id} profile")

    def init(self) -> InitModelData:
        if self.cfg.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr, valids, num_invalids: 0.0
        elif self.cfg.exploration_loss == "entropy":
            self.exploration_loss_func = self._entropy_exploration_loss
        elif self.cfg.exploration_loss == "symmetric_kl":
            self.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f"{self.cfg.exploration_loss} not supported!")

        if self.cfg.kl_loss_coeff == 0.0:
            if is_continuous_action_space(self.env_info.action_space):
                log.warning(
                    "WARNING! It is generally recommended to enable Fixed KL loss (https://arxiv.org/pdf/1707.06347.pdf) for continuous action tasks to avoid potential numerical issues. "
                    "I.e. set --kl_loss_coeff=0.1"
                )
            self.kl_loss_func = lambda action_space, action_logits, distribution, valids, num_invalids: (
                None,
                0.0,
            )
        else:
            self.kl_loss_func = self._kl_loss

        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # initialize device
        self.device = policy_device(self.cfg, self.policy_id)

        log.debug("Initializing actor-critic model on device %s", self.device)

        # trainable torch module
        self.actor_critic = create_actor_critic(
            self.cfg, self.env_info.obs_space, self.env_info.action_space
        )
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)

        #################
        # MY CODE BLOCK #
        #################
        log.debug("Initializing intrinsic reward module")
        self.ir_module = make_intrinsic_reward_module(self.cfg, self.env_info.obs_space)
        log.debug("Intrinsic reward module initialized")
        self.ir_module.model_to_device(self.device)
        #################

        def share_mem(t):
            if t is not None and not t.is_cuda:
                return t.share_memory_()
            return t

        # noinspection PyProtectedMember
        self.actor_critic._apply(share_mem)
        self.actor_critic.train()

        params = list(self.actor_critic.parameters())

        #################
        # MY CODE BLOCK #
        #################
        # noinspection PyProtectedMember
        self.ir_module._apply(share_mem)
        self.ir_module.train()

        params += list(self.ir_module.parameters())
        #################

        optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)
        if self.cfg.optimizer not in optimizer_cls:
            raise RuntimeError(f"Unknown optimizer {self.cfg.optimizer}")

        optimizer_cls = optimizer_cls[self.cfg.optimizer]
        log.debug(f"Using optimizer {optimizer_cls}")

        optimizer_kwargs = dict(
            lr=self.cfg.learning_rate,  # use default lr only in ctor, then we use the one loaded from the checkpoint
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        if self.cfg.optimizer in ["adam", "lamb"]:
            optimizer_kwargs["eps"] = self.cfg.adam_eps

        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

        self.load_from_checkpoint(self.policy_id)
        self.param_server.init(self.actor_critic, self.train_step, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        self.lr_scheduler = get_lr_scheduler(self.cfg)
        self.curr_lr = self.cfg.learning_rate if self.curr_lr is None else self.curr_lr
        self._apply_lr(self.curr_lr)

        self.is_initialized = True

        return model_initialization_data(
            self.cfg, self.policy_id, self.actor_critic, self.train_step, self.device
        )

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        #############
        # NO CHANGE #
        #############
        return Learner.checkpoint_dir(cfg, policy_id)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        #############
        # NO CHANGE #
        #############
        return Learner.get_checkpoints(checkpoints_dir, pattern)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        #############
        # NO CHANGE #
        #############
        return Learner.load_checkpoint(checkpoints, device)

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict["train_step"]
            self.env_steps = checkpoint_dict["env_steps"]
            self.best_performance = checkpoint_dict.get(
                "best_performance", self.best_performance
            )
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.curr_lr = checkpoint_dict.get("curr_lr", self.cfg.learning_rate)

        #################
        # MY CODE BLOCK #
        #################
        self.ir_module.load_state_dict(checkpoint_dict["ir_module"])
        #################

        log.info(f"Loaded experiment state at {self.train_step=}, {self.env_steps=}")

    def load_from_checkpoint(
        self, policy_id: PolicyID, load_progress: bool = True
    ) -> None:
        #############
        # NO CHANGE #
        #############
        super().load_from_checkpoint(policy_id, load_progress)

    def _should_save_summaries(self):
        #############
        # NO CHANGE #
        #############
        return super()._should_save_summaries()

    def _after_optimizer_step(self):
        #############
        # NO CHANGE #
        #############
        super()._after_optimizer_step()

    def _get_checkpoint_dict(self):
        checkpoint = {
            "train_step": self.train_step,
            "env_steps": self.env_steps,
            "best_performance": self.best_performance,
            "model": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "curr_lr": self.curr_lr,
            #################
            # MY CODE BLOCK #
            #################
            "ir_module": self.ir_module.state_dict(),
            #################
        }
        return checkpoint

    def _save_impl(
        self, name_prefix, name_suffix, keep_checkpoints, verbose=True
    ) -> bool:
        #############
        # NO CHANGE #
        #############
        return super()._save_impl(name_prefix, name_suffix, keep_checkpoints, verbose)

    def save(self) -> bool:
        #############
        # NO CHANGE #
        #############
        return super().save()

    def save_milestone(self):
        #############
        # NO CHANGE #
        #############
        super().save_milestone()

    def save_best(self, policy_id, metric, metric_value) -> bool:
        #############
        # NO CHANGE #
        #############
        return super().save_best(policy_id, metric, metric_value)

    def set_new_cfg(self, new_cfg: Dict) -> None:
        #############
        # NO CHANGE #
        #############
        super().set_new_cfg(new_cfg)

    def set_policy_to_load(self, policy_to_load: PolicyID) -> None:
        #############
        # NO CHANGE #
        #############
        super().set_policy_to_load(policy_to_load)

    def _maybe_update_cfg(self) -> None:
        #############
        # NO CHANGE #
        #############
        super()._maybe_update_cfg()

    def _maybe_load_policy(self) -> None:
        #############
        # NO CHANGE #
        #############
        super()._maybe_load_policy()

    @staticmethod
    def _policy_loss(
        ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids: int
    ):
        #############
        # NO CHANGE #
        #############
        return Learner._policy_loss(
            ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids
        )

    def _value_loss(
        self,
        new_values: Tensor,
        old_values: Tensor,
        target: Tensor,
        clip_value: float,
        valids: Tensor,
        num_invalids: int,
    ) -> Tensor:
        #############
        # NO CHANGE #
        #############
        return super()._value_loss(
            new_values, old_values, target, clip_value, valids, num_invalids
        )

    def _kl_loss(
        self,
        action_space,
        action_logits,
        action_distribution,
        valids,
        num_invalids: int,
    ) -> Tuple[Tensor, Tensor]:
        #############
        # NO CHANGE #
        #############
        return super()._kl_loss(
            action_space, action_logits, action_distribution, valids, num_invalids
        )

    def _entropy_exploration_loss(
        self, action_distribution, valids, num_invalids: int
    ) -> Tensor:
        #############
        # NO CHANGE #
        #############
        return super()._entropy_exploration_loss(
            action_distribution, valids, num_invalids
        )

    def _symmetric_kl_exploration_loss(
        self, action_distribution, valids, num_invalids: int
    ) -> Tensor:
        #############
        # NO CHANGE #
        #############
        return super()._symmetric_kl_exploration_loss(
            action_distribution, valids, num_invalids
        )

    def _optimizer_lr(self):
        #############
        # NO CHANGE #
        #############
        return super()._optimizer_lr()

    def _apply_lr(self, lr: float) -> None:
        #############
        # NO CHANGE #
        #############
        return super()._apply_lr(lr)

    def _get_minibatches(self, batch_size, experience_size):
        #############
        # NO CHANGE #
        #############
        return super()._get_minibatches(batch_size, experience_size)

    @staticmethod
    def _get_minibatch(buffer, indices):
        #############
        # NO CHANGE #
        #############
        return Learner._get_minibatch(buffer, indices)

    def _calculate_losses(self, mb: AttrDict, num_invalids: int) -> Tuple[
        ActionDistribution,
        Tensor,
        Tensor | float,
        Optional[Tensor],
        Tensor | float,
        Tensor,
        Tensor,
        Dict,
    ]:
        ################################################
        # I ADDED ADDITIONAL TENSOR TO THE RETURN TYPE #
        ################################################
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        # calculate policy head outside of recurrent loop
        with self.timing.add_time("forward_head"):
            head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = head_outputs.size(0)

        # initial rnn states
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                # this is the only way to stop RNNs from backpropagating through invalid timesteps
                # (i.e. experience collected by another policy)
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()
                head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                    head_outputs,
                    done_or_invalid,
                    mb.rnn_states,
                    recurrence,
                )
            else:
                rnn_states = mb.rnn_states[::recurrence]

        # calculate RNN outputs for each timestep in a loop
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    core_output_seq, _ = self.actor_critic.forward_core(
                        head_output_seq, rnn_states
                    )
                core_outputs = build_core_out_from_seq(
                    core_output_seq, inverted_select_inds
                )
                del core_output_seq
            else:
                core_outputs, _ = self.actor_critic.forward_core(
                    head_outputs, rnn_states
                )

            del head_outputs

        num_trajectories = minibatch_size // recurrence
        assert core_outputs.shape[0] == minibatch_size

        with self.timing.add_time("tail"):
            # calculate policy tail outside of recurrent loop
            result = self.actor_critic.forward_tail(
                core_outputs, values_only=False, sample_actions=False
            )
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = result["values"].squeeze()

            del core_outputs

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            #################
            # MY CODE BLOCK #
            #################
            if self.cfg.with_vtrace:
                raise NotImplementedError(
                    "V-trace is not supported for intrinsic reward learner"
                )
            adv = mb.advantages
            targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(
                adv_std, 1e-7
            )  # normalize advantage

            ir_adv = mb.intrinsic_advantages
            ir_targets = mb.intrinsic_returns

            ir_adv_std, ir_adv_mean = torch.std_mean(
                masked_select(ir_adv, valids, num_invalids)
            )
            ir_adv = (ir_adv - ir_adv_mean) / torch.clamp_min(
                ir_adv_std, 1e-7
            )  # normalize advantage

            pre_ir_adv = adv
            pre_ir_targets = targets

            adv = adv + self.ir_weight * ir_adv
            targets = targets + self.ir_weight * ir_targets
            #################

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(
                ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids
            )
            exploration_loss = self.exploration_loss_func(
                action_distribution, valids, num_invalids
            )
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space,
                mb.action_logits,
                action_distribution,
                valids,
                num_invalids,
            )
            old_values = mb["values"]
            value_loss = self._value_loss(
                values, old_values, targets, clip_value, valids, num_invalids
            )

        #################
        # MY CODE BLOCK #
        #################
        with self.timing.add_time("intrinsic_rewards_loss"):
            intrinsic_rewards_loss = self.ir_module.loss(mb)
        #################

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            #################
            # MY CODE BLOCK #
            #################
            intrinsic_rewards_loss=intrinsic_rewards_loss,
            intrinsic_adv=ir_adv,
            intrinsic_adv_std=ir_adv_std,
            intrinsic_adv_mean=ir_adv_mean,
            pre_ir_adv=pre_ir_adv,
            pre_ir_targets=pre_ir_targets,
            pre_ir_adv_mean=adv_mean,
            pre_ir_adv_std=adv_std,
            #################
        )

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            #################
            # MY CODE BLOCK #
            #################
            intrinsic_rewards_loss,
            #################
            loss_summaries,
        )

    def _train(
        self,
        gpu_buffer: TensorDict,
        batch_size: int,
        experience_size: int,
        num_invalids: int,
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"
                raise NotImplementedError(
                    "V-trace is not supported for intrinsic reward learner"
                )

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        #################
                        # MY CODE BLOCK #
                        #################
                        intrinsic_rewards_loss,
                        #################
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss

                    #################
                    # MY CODE BLOCK #
                    #################
                    loss += intrinsic_rewards_loss
                    #################

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                            #################
                            # MY CODE BLOCK #
                            #################
                            to_scalar(intrinsic_rewards_loss),
                            #################
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(
                            old_action_distribution
                        )
                        kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(
                            f"KL-divergence is very high: {kl_old.max().item():.4f}"
                        )

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    #################
                    # MY CODE BLOCK #
                    #################
                    for p in self.ir_module.parameters():
                        p.grad = None
                    #################

                    loss.backward()

                    #######################
                    # ORIGINAL CODE BLOCK #
                    #######################
                    # if self.cfg.max_grad_norm > 0.0:
                    #     with timing.add_time("clip"):
                    #         torch.nn.utils.clip_grad_norm_(
                    #             self.actor_critic.parameters(), self.cfg.max_grad_norm
                    #         )
                    #######################
                    # MY CODE BLOCK       #
                    #######################
                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            # combine iterable of parameters from both models
                            params = list(self.actor_critic.parameters()) + list(
                                self.ir_module.parameters()
                            )
                            torch.nn.utils.clip_grad_norm_(
                                params, self.cfg.max_grad_norm
                            )
                    #######################

                    curr_policy_version = (
                        self.train_step
                    )  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = (
                            self.curr_lr
                            * (experience_size - num_invalids)
                            / experience_size
                        )
                    self._apply_lr(actual_lr)

                    ###################################################
                    # TO MAKE PBT WORK WITH INTRINSIC REWARD LEARNER  #
                    # WE WOULD NEED TO ALSO LOCK IR_MODULE PARAMETERS #
                    ###################################################
                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(
                            self.curr_lr, recent_kls
                        )

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= (
                        epoch == summaries_epoch and batch_num == summaries_batch
                    )
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(
                            AttrDict(summary_vars)
                        )
                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        ###############
        # ALL MY CODE #
        ###############
        stats = super()._record_summaries(train_loop_vars)
        stats.intrinsic_rewards_loss = train_loop_vars.intrinsic_rewards_loss.detach()
        intrinsic_rewards = train_loop_vars.mb.intrinsic_rewards.detach()
        stats.intrinsic_rewards_mean = intrinsic_rewards.mean()
        stats.intrinsic_rewards_max = intrinsic_rewards.max()
        stats.intrinsic_rewards_min = intrinsic_rewards.min()
        stats.pre_intrinsic_adv_mean = train_loop_vars.pre_ir_adv_mean.detach()
        stats.pre_intrinsic_adv_std = train_loop_vars.pre_ir_adv_std.detach()
        stats.intrinsic_adv_mean = train_loop_vars.intrinsic_adv_mean.detach()
        stats.intrinsic_adv_std = train_loop_vars.intrinsic_adv_std.detach()
        stats.adv_mean = stats.intrinsic_adv_mean + stats.pre_intrinsic_adv_mean
        stats.adv_std = stats.intrinsic_adv_std + stats.pre_intrinsic_adv_std
        return stats

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (
                curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag
            )
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]

            # ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self.actor_critic(
                normalized_last_obs, buff["rnn_states"][:, -1], values_only=True
            )["values"]
            buff["values"][:, -1] = next_values

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values = buff[
                    "values"
                ].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(
                    denormalized_values, denormalize=True
                )
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]

        #################
        # MY CODE BLOCK #
        #################
        with self.timing.add_time("intrinsic_rewards computation"):
            if not self.ir_module.training:
                self.ir_module.train()
            intrinsic_output = self.ir_module(buff, leading_dims=2)
            intrinsic_rewards = intrinsic_output["intrinsic_rewards"]
            # buff["rewards"] += (
            #     intrinsic_rewards.to(buff["rewards"].device) * self.ir_weight
            # )
            for key, value in intrinsic_output.items():
                buff[key] = value
        #################
        with torch.no_grad():

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards"].add_(
                    self.cfg.gamma
                    * denormalized_values[:, :-1]
                    * buff["time_outs"]
                    * buff["dones"]
                )

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
                buff["advantages"] = gae_advantages(
                    buff["rewards"],
                    buff["dones"],
                    denormalized_values,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns"] = (
                    buff["advantages"]
                    + buff["valids"][:, :-1] * denormalized_values[:, :-1]
                )

                #################
                # MY CODE BLOCK #
                #################
                buff["intrinsic_advantages"] = gae_advantages(
                    intrinsic_rewards,
                    buff["dones"],
                    denormalized_values,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )

                buff["intrinsic_returns"] = (
                    buff["intrinsic_advantages"]
                    + buff["valids"][:, :-1] * denormalized_values[:, :-1]
                )
                #################

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to(
                "cpu", copy=True, dtype=torch.float, non_blocking=True
            )
            buff["rewards_cpu"] = buff["rewards"].to(
                "cpu", copy=True, dtype=torch.float, non_blocking=True
            )

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])  # in-place

            #################
            # MY CODE BLOCK #
            #################
            buff["intrinsic_returns_pre_normalization"] = buff[
                "intrinsic_returns"
            ].clone()
            if self.cfg.normalize_intrinsic_returns:
                self.ir_module.returns_normalizer(buff["intrinsic_returns"])
            #################

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(
                        f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples"
                    )

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][
                    invalid_indices
                ] = -1  # -1 seems like a safe value

        #################
        # MY CODE BLOCK #
        #################
        for key, value in intrinsic_output.items():
            buff[key] = value.reshape((dataset_size,) + tuple(value.shape[2:]))
        return buff, dataset_size, num_invalids
        #################

    def train(self, batch: TensorDict) -> Optional[Dict]:
        #############
        # NO CHANGE #
        #############
        return super().train(batch)
