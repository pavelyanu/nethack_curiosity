from __future__ import annotations

from threading import Thread
from typing import Optional

from signal_slot.signal_slot import EventLoop, Timer
from torch import Tensor

from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.algo.learning.learner_worker import LearnerWorker

from nethack_curiosity.intrinsic_reward.intrinsic_reward_learner import IntrinsicRewardLearner


class IntrinsicRewardLearnerWorker(LearnerWorker):
    
    def __init__(
        self,
        evt_loop: EventLoop,
        cfg: Config,
        env_info: EnvInfo,
        buffer_mgr: BufferMgr,
        batcher: Batcher,
        policy_id: PolicyID,
    ):
        Configurable.__init__(self, cfg)

        unique_name = f"{LearnerWorker.__name__}_p{policy_id}"
        HeartbeatStoppableEventLoopObject.__init__(self, evt_loop, unique_name, cfg.heartbeat_interval)

        self.batcher: Batcher = batcher
        self.batcher_thread: Optional[Thread] = None

        policy_versions_tensor: Tensor = buffer_mgr.policy_versions
        self.param_server = ParameterServer(policy_id, policy_versions_tensor, cfg.serial_mode)
        self.learner: IntrinsicRewardLearner = IntrinsicRewardLearner(cfg, env_info, policy_versions_tensor, policy_id, self.param_server)

        # total number of full training iterations (potentially multiple minibatches/epochs per iteration)
        self.training_iteration_since_resume: int = 0

        self.cache_cleanup_timer = Timer(self.event_loop, 30)
        self.cache_cleanup_timer.timeout.connect(self._cleanup_cache)