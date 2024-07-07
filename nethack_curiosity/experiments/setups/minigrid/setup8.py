_constant_params = {
    "batch_size": 2048,
    "num_batches_per_epoch": 4,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "rollout": 128,
    "rnd_share_encoder": True,
    "rnd_mlp_layers": 2048,
    "recompute_intrinsic_loss": True,
    "noveld_constant_novelty": 0.0,
    "inverse_wiring": "ride",
    "visit_count_weighting": "inverse_sqrt",
    "train_for_env_steps": 1000000,
}

_setups = [
    {
        "intrinsic_reward_module": "mock",
    },
    {
        "intrinsic_reward_module": "rnd",
    },
    {
        "intrinsic_reward_module": "noveld",
    },
    {
        "intrinsic_reward_module": "inverse",
    },
]

for setup in _setups:
    setup.update(_constant_params)


def get_setups():
    return _setups
