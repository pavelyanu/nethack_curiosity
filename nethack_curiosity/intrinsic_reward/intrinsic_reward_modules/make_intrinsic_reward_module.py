from gymnasium.spaces import Space

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from sample_factory.utils.typing import Config


def make_intrinsic_reward_module(
    cfg: Config, obs_space: Space
) -> IntrinsicRewardModule:
    if cfg.intrinsic_reward_module == "mock":
        from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.mock import (
            MockIntrinsicRewardModule,
        )

        return MockIntrinsicRewardModule(cfg, obs_space)
    elif cfg.intrinsic_reward_module == "rnd":
        from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.rnd import (
            RNDIntrinsicRewardModule,
        )

        return RNDIntrinsicRewardModule(cfg, obs_space)
    elif cfg.intrinsic_reward_module == "noveld":
        from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.noveld import (
            NovelDIntrindicRewardModule,
        )

        return NovelDIntrindicRewardModule(cfg, obs_space)
    else:
        raise NotImplementedError(
            f"Unknown intrinsic reward module: {cfg.intrinsic_reward_module}"
        )
