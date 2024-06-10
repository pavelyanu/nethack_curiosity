# fmt: off
argv = [
    "--experiment", "unittest_experiment",
    # "--train_dir", "/tmp/unittest_train_dir",
    "--restart_behavior", "overwrite",
    "--device", "cpu",
    "--seed", "42",
    "--num_policies", "1",
    "--serial_mode", "True",
    "--async_rl", "False",
    "--num_workers", "1",
    "--num_envs_per_worker", "2",
    "--shuffle_minibatches", "False",
    "--normalize_input", "True",
    "--normalize_returns", "True",
    # "--log_to_file", "False",
    "--train_for_env_steps", "1000",
    "--train_for_seconds", "5",
    "--save_every_sec", "3600",
    "--keep_checkpoints", "0",
    "--load_checkpoint_kind", "latest"
]
# fmt: on
