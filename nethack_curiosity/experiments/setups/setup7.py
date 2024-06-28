_constant_params = {
    "batch_size": 2048,
    "num_batches_per_epoch": 4,
    "num_workers": 16,
    "num_envs_per_worker": 32,
    "rollout": 128,
    "rnd_share_encoder": True,
    "rnd_mlp_layers": 2048,
}

_setups = [
    {
        "intrinsic_reward_weight": 0.1,
        "intrinsic_reward_module": "noveld",
        "noveld_constant_novelty": 0.0,
        "version": 0,
    },
    {
        "intrinsic_reward_weight": 0.1,
        "intrinsic_reward_module": "noveld",
        "noveld_constant_novelty": 0.5,
        "version": 0,
    },
    {
        "intrinsic_reward_weight": 0.1,
        "intrinsic_reward_module": "rnd",
        "noveld_constant_novelty": 0.0,
        "version": 0,
    },
    {
        "intrinsic_reward_weight": 0.25,
        "intrinsic_reward_module": "rnd",
        "noveld_constant_novelty": 0.0,
        "version": 0,
    },
    {
        "intrinsic_reward_weight": 0.5,
        "intrinsic_reward_module": "rnd",
        "noveld_constant_novelty": 0.0,
        "version": 0,
    },
    {
        "intrinsic_reward_weight": 0.0,
        "intrinsic_reward_module": "mock",
        "noveld_constant_novelty": 0.0,
        "version": 0,
    },
]

num_versions = 4
new_setups = _setups.copy()
for i in range(1, num_versions):
    for setup in _setups:
        new_setup = setup.copy()
        new_setup["version"] = i
        new_setups.append(new_setup)

_setups = new_setups

for setup in _setups:
    setup.update(_constant_params)


def get_setups():
    return _setups
