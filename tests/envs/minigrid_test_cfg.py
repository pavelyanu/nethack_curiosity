from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from nethack_curiosity.envs.minigrid.minigrid_env_params import (
    minigrid_env_override_defaults,
    add_minigrid_env_args,
)
from sample_factory.utils.typing import Config

from tests.envs.base_test_input import base_argv

# fmt: off
minigrid_argv = [
    "--normalize_input", "True",
    "--normalize_input_keys", "image",
    "--batch_size", "64",
    "--observation_keys", "image",
]
# fmt: on
minigrid_argv.extend(base_argv)


def make_minigrid_cfg(env: str) -> Config:
    args = ["--env", env]
    args.extend(args)
    parser, partial_cfg = parse_sf_args(args)
    add_minigrid_env_args("minigrid-empty", parser)
    minigrid_env_override_defaults("minigrid-empty", parser)
    final_cfg = parse_full_cfg(parser, args)
    return final_cfg
