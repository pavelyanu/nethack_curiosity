_constant_params = {
    "batch_size": 2048,
    "num_batches_per_epoch": 4,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "rollout": 128,
    "rnd_share_encoder": True,
    "rnd_mlp_layers": 512,
}

_setups = [
    {
        "intrinsic_reward_weight": 0.05,
        "intrinsic_reward_module": "noveld",
        "version": 0,
    },
    {
        "intrinsic_reward_weight": 0.05,
        "intrinsic_reward_module": "noveld",
        "version": 1,
    },
    {"intrinsic_reward_weight": 0.1, "intrinsic_reward_module": "noveld", "version": 0},
    {"intrinsic_reward_weight": 0.1, "intrinsic_reward_module": "noveld", "version": 1},
    {"intrinsic_reward_weight": 0.0, "version": 0},
    {"intrinsic_reward_weight": 0.0, "version": 1},
    {"intrinsic_reward_weight": 0.05, "intrinsic_reward_module": "rnd", "version": 0},
    {"intrinsic_reward_weight": 0.05, "intrinsic_reward_module": "rnd", "version": 1},
    {"intrinsic_reward_weight": 0.1, "intrinsic_reward_module": "rnd", "version": 0},
    {"intrinsic_reward_weight": 0.1, "intrinsic_reward_module": "rnd", "version": 1},
]

for setup in _setups:
    setup.update(_constant_params)


def get_setups():
    return _setups
