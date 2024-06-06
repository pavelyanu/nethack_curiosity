from typing import Dict, Tuple, Any, Optional
from sample_factory.utils.attr_dict import AttrDict
from torch import Tensor

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.utils.typing import ActionDistribution, Config, PolicyID

from sample_factory.algo.learning.learner import Learner

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import IntrinsicRewardModule

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
        self._ir_module : IntrinsicRewardModule = self.get_intrinsic_reward_module(cfg)

    def get_intrinsic_reward_module(self, cfg: Config) -> IntrinsicRewardModule:
        ...

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        intrinsic_rewards = self._ir_module.get_intrinsic_rewards(mb)
        mb.rewards_cpu = mb.rewards_cpu + intrinsic_rewards