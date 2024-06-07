from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)


def make_intrinsic_reward_module(cfg, env_info) -> IntrinsicRewardModule:
    if cfg.intrinsic_reward_module == "mock":
        from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.mock import (
            MockIntrinsicRewardModule,
        )

        return MockIntrinsicRewardModule(cfg, env_info)
    else:
        raise NotImplementedError(
            f"Unknown intrinsic reward module: {cfg.intrinsic_reward_module}"
        )
