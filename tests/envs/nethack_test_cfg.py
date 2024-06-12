from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from nethack_curiosity.envs.nethack.nethack_env_params import (
    nethack_env_override_defaults,
    add_nethack_env_args,
)
from sample_factory.utils.typing import Config

from tests.envs.base_test_input import base_argv

# fmt: off
nethack_arvg = [

]
# fmt: on
nethack_arvg.extend(base_argv)


def make_nethack_cfg(env: str) -> Config:
    args = ["--env", env]
    args.extend(nethack_arvg)
    parser, partial_cfg = parse_sf_args(args)
    add_nethack_env_args(env, parser, testing=True)
    nethack_env_override_defaults(env, parser, testing=True)
    final_cfg = parse_full_cfg(parser, args)
    return final_cfg
