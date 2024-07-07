_constant_params = {
    "batch_size": 4096,
    "num_batches_per_epoch": 1,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "rollout": 32,
    "rnd_share_encoder": True,
    "rnd_mlp_layers": 2048,
    "recompute_intrinsic_loss": True,
    "noveld_constant_novelty": 0.0,
    "inverse_wiring": "ride",
    "visit_count_weighting": "inverse_sqrt",
    "train_for_env_steps": 10_000_000,
}

_setups = [
    {"intrinsic_reward_module": "rnd", "version": 1, "intrinsic_reward_weight": 0.1},
    {"intrinsic_reward_module": "mock", "version": 1, "intrinsic_reward_weight": 0.0},
    {"intrinsic_reward_module": "rnd", "version": 1, "intrinsic_reward_weight": 0.25},
    {"intrinsic_reward_module": "rnd", "version": 1, "intrinsic_reward_weight": 0.5},
    #
    {"intrinsic_reward_module": "rnd", "version": 2, "intrinsic_reward_weight": 0.1},
    {"intrinsic_reward_module": "mock", "version": 2, "intrinsic_reward_weight": 0.0},
    {"intrinsic_reward_module": "rnd", "version": 2, "intrinsic_reward_weight": 0.25},
    {"intrinsic_reward_module": "rnd", "version": 2, "intrinsic_reward_weight": 0.5},
    #
    {"intrinsic_reward_module": "rnd", "version": 3, "intrinsic_reward_weight": 0.1},
    {"intrinsic_reward_module": "mock", "version": 3, "intrinsic_reward_weight": 0.0},
    {"intrinsic_reward_module": "rnd", "version": 3, "intrinsic_reward_weight": 0.25},
    {"intrinsic_reward_module": "rnd", "version": 3, "intrinsic_reward_weight": 0.5},
    #
    {"intrinsic_reward_module": "rnd", "version": 4, "intrinsic_reward_weight": 0.1},
    {"intrinsic_reward_module": "mock", "version": 4, "intrinsic_reward_weight": 0.0},
    {"intrinsic_reward_module": "rnd", "version": 4, "intrinsic_reward_weight": 0.25},
    {"intrinsic_reward_module": "rnd", "version": 4, "intrinsic_reward_weight": 0.5},
]

for setup in _setups:
    setup.update(_constant_params)


def get_setups():
    return _setups
