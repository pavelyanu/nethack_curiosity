from typing import Dict, Tuple, Optional

import numpy as np

import torch
from torch import Tensor

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.optimizers import Lamb
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import to_scalar, synchronize, masked_select
from sample_factory.model.actor_critic import create_actor_critic
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
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.mock import (
    MockIntrinsicRewardModule,
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
        self.ir_module = make_intrinsic_reward_module(self.cfg, self.env_info)
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
        return super().checkpoint_dir(cfg, policy_id)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        #############
        # NO CHANGE #
        #############
        return super().get_checkpoints(checkpoints_dir, pattern)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        #############
        # NO CHANGE #
        #############
        return super().load_checkpoint(checkpoints, device)

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
        return super()._policy_loss(
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
        return super()._get_minibatch(buffer, indices)

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
        ###########################################
        # Added another tensor to the return type #
        ###########################################

        #########################
        # EVERYTHING IS MY CODE #
        #########################
        intrinsic_rewards_loss = self.ir_module.loss(mb)
        (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        ) = super()._calculate_losses(mb, num_invalids)
        loss_summaries["intrinsic_rewards_loss"] = intrinsic_rewards_loss.item()
        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            intrinsic_rewards_loss,
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

                #################
                # MY CODE BLOCK #
                #################
                with timing.add_time("intrinsic_rewards computation"):
                    intrinsic_rewards = self.ir_module.get_intrinsic_rewards(mb)
                rewards_before_intrinsic_max = mb.rewards_cpu.max()
                rewards_before_intrinsic_min = mb.rewards_cpu.min()
                mb.rewards_cpu += intrinsic_rewards
                #################

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
        stats.intrinsic_rewards_loss = train_loop_vars.intrinsic_rewards_loss
        stats.rewards_before_intrinsic_max = (
            train_loop_vars.rewards_before_intrinsic_max
        )
        stats.rewards_before_intrinsic_min = (
            train_loop_vars.rewards_before_intrinsic_min
        )
        stats.intrinsic_rewards_max = train_loop_vars.intrinsic_rewards.max()
        stats.intrinsic_rewards_min = train_loop_vars.intrinsic_rewards.min()
        return stats

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        #############
        # NO CHANGE #
        #############
        return super()._prepare_batch(batch)

    def train(self, batch: TensorDict) -> Optional[Dict]:
        #############
        # NO CHANGE #
        #############
        return super().train(batch)

    def _make_intrinsic_reward_module(self, cfg: Config) -> IntrinsicRewardModule:
        module_name = cfg.intrinsic_reward_module
        if module_name == "mock":
            return MockIntrinsicRewardModule(cfg, self.env_info)
        else:
            raise NotImplementedError(f"Unknown intrinsic reward module: {module_name}")