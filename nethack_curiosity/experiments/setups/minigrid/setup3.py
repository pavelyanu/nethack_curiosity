_constant_params = {
    "batch_size": 2048,
    "num_batches_per_epoch": 4,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "rollout": 128,
}

_setups = [
    {"intrinsic_reward_weight": 0.001},
    {"intrinsic_reward_weight": 0.005},
    {"intrinsic_reward_weight": 0.01},
    {"intrinsic_reward_weight": 0.05},
    {"intrinsic_reward_weight": 0.1},
    {"intrinsic_reward_weight": 0.5},
    {"intrinsic_reward_weight": 0.2},
    {"intrinsic_reward_weight": 1},
    {"intrinsic_reward_weight": 5},
    {"intrinsic_reward_weight": 0.0},
    {"intrinsic_reward_weight": 0.11},
    {"intrinsic_reward_weight": 0.13},
    {"intrinsic_reward_weight": 0.16},
    {"intrinsic_reward_weight": 0.18},
]

for setup in _setups:
    setup.update(_constant_params)


def get_setups():
    return _setups
