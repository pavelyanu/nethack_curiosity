from gymnasium.spaces import Space

from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.noveld import (
    NovelDIntrindicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.rnd import (
    RNDIntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.inverse import (
    InverseModelIntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.mock import (
    MockIntrinsicRewardModule,
)
from sample_factory.utils.typing import Config


def reward_module_class(name: str) -> type:
    mapping = {
        "mock": MockIntrinsicRewardModule,
        "rnd": RNDIntrinsicRewardModule,
        "noveld": NovelDIntrindicRewardModule,
        "inverse": InverseModelIntrinsicRewardModule,
    }
    if name not in mapping:
        raise ValueError(f"Unknown intrinsic reward module: {name}")
    return mapping[name]


def make_intrinsic_reward_module(
    cfg: Config, obs_space: Space, action_space: Space
) -> IntrinsicRewardModule:
    return reward_module_class(cfg.intrinsic_reward_module)(
        cfg, obs_space, action_space
    )
