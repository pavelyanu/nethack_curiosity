_constant_params = {
    "rollout": 128,
    "intrinsic_reward_weight": 0.01,
}

_setups = [
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

for setup in _setups:
    setup.update(_constant_params)


def get_setups():
    return _setups
