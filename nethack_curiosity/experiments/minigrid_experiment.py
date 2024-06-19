from sample_factory.launcher.run_description import (
    Experiment,
    ParamGrid,
    RunDescription,
    ParamList,
)

# _params = ParamGrid(
#     [
#         ("env", ["keycorridor"]),
#         ("minigrid_room_size", [3]),
#         ("minigrid_row_num", [1]),
#         ("train_for_env_steps", [20_000_000]),
#         ("max_policy_lag", [64]),
#         ("recurrence", [-1]),
#         ("batch_size", [64, 128, 256, 512, 1014]),
#         ("rollout", [32]),
#         ("with_vtrace", [True, False]),
#         ("normalize_returns", [True, False]),
#         ("intrinsic_reward_weight", [0.1, 0.01, 0.005, 0.0]),
#     ]
# )

constant_params = {
    "env": "keycorridor",
    "minigrid_room_size": 3,
    "minigrid_row_num": 3,
    "train_for_env_steps": 5_000_000,
    "recurrence": -1,
    "rollout": 32,
    "intrinsic_reward_weight": 0.01,
    "intrinsic_reward_module": "rnd",
}

# rule of thumb:
# batch_size * num_batches_per_epoch ~ num_workers * num_envs_per_worker * rollout
# for 8 workers and 16 envs per worker right side is 8 * 16 * 64 = 8192
# for 16 workers and 32 envs per worker right side is 16 * 32 * 64 = 32768

setups = [
    # Batch size 256
    {
        "batch_size": 256,
        "num_batches_per_epoch": 8,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 256,
        "num_batches_per_epoch": 4,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 256,
        "num_batches_per_epoch": 2,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    # Batch size 512
    # Num workers 8
    {
        "batch_size": 512,
        "num_batches_per_epoch": 8,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 512,
        "num_batches_per_epoch": 4,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 512,
        "num_batches_per_epoch": 2,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    # Num workers 8
    {
        "batch_size": 512,
        "num_batches_per_epoch": 8,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 512,
        "num_batches_per_epoch": 4,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 512,
        "num_batches_per_epoch": 2,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    # Batch size 1024
    # Num workers 8
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 8,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 4,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 2,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 1,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    # Num workers 16
    # Num envs per worker 16
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 8,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 4,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 2,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    # Num envs per worker 32
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 8,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 4,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
    {
        "batch_size": 1024,
        "num_batches_per_epoch": 2,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
    # Batch size 2048
    # Num workers 8
    # Num envs per worker 16
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 8,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 4,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 2,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 1,
        "num_workers": 8,
        "num_envs_per_worker": 16,
    },
    # Num workers 16
    # Num envs per worker 16
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 8,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 4,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 2,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 1,
        "num_workers": 16,
        "num_envs_per_worker": 16,
    },
    # Num envs per worker 32
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 8,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 4,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 2,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
    {
        "batch_size": 2048,
        "num_batches_per_epoch": 1,
        "num_workers": 16,
        "num_envs_per_worker": 32,
    },
]

for setup in setups:
    setup.update(constant_params)

# _params = ParamList(setups)
_params = setups

# _params = _params.generate_params(randomize=False)
print("Total number of experiments:", len(list(_params)))
print("Params:")
for p in _params:
    print(p)

_experiment = Experiment(
    "minigrid",
    "python -m nethack_curiosity.experiments.run_minigrid",
    _params,
)

description = RunDescription("minigrid_bs_bpe_grid_at_rnd_0.01", [_experiment])

RUN_DESCRIPTION = description
