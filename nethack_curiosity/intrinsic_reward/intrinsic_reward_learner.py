from typing import Dict, Tuple, Optional

import numpy as np

import torch
from torch import Tensor

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.optimizers import Lamb
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.timing import Timing
from sample_factory.algo.utils.action_distributions import is_continuous_action_space
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
        self._ir_module: Optional[IntrinsicRewardModule] = None
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
        # MY CODE START #
        #################

        log.debug("Initializing intrinsic reward module")
        self._ir_module = make_intrinsic_reward_module(self.cfg, self.env_info)
        log.debug("Intrinsic reward module initialized")
        self._ir_module.model_to_device(self.device)

        #################
        #  MY CODE END  #
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
        # MY CODE START #
        #################

        # noinspection PyProtectedMember
        self._ir_module._apply(share_mem)
        self._ir_module.train()

        params += list(self._ir_module.parameters())

        #################
        #  MY CODE END  #
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

    def _make_intrinsic_reward_module(self, cfg: Config) -> IntrinsicRewardModule:
        module_name = cfg.intrinsic_reward_module
        if module_name == "mock":
            return MockIntrinsicRewardModule(cfg, self.env_info)
        else:
            raise NotImplementedError(f"Unknown intrinsic reward module: {module_name}")

    def _calculate_losses(self, mb: AttrDict, num_invalids: int) -> Tuple[
        ActionDistribution,
        Tensor,
        Tensor | float,
        Optional[Tensor],
        Tensor | float,
        Tensor,
        Dict,
    ]:
        intrinsic_rewards = self._ir_module.get_intrinsic_rewards(mb)
        mb.rewards_cpu = mb.rewards_cpu + intrinsic_rewards
        return super()._calculate_losses(mb, num_invalids)
