from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from nethack_curiosity.envs.minigrid.minigrid_env_params import (
    minigrid_env_override_defaults,
    add_minigrid_env_args,
)
from sample_factory.utils.typing import Config

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
    "--train_for_seconds", "10",
    "--save_every_sec", "3600",
    "--keep_checkpoints", "0",
    "--load_checkpoint_kind", "latest",
    "--normalize_input", "True",
    "--normalize_input_keys", "image",
    "--batch_size", "64",
    "--observation_keys", "image",
]
# fmt: on


def make_minigrid_cfg(env: str) -> Config:
    args = ["--env", env]
    args.extend(args)
    parser, partial_cfg = parse_sf_args(args)
    add_minigrid_env_args("minigrid-empty", parser)
    minigrid_env_override_defaults("minigrid-empty", parser)
    final_cfg = parse_full_cfg(parser, args)
    return final_cfg
