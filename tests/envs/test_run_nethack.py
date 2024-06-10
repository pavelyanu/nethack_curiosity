import pytest

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.misc import ExperimentStatus

from nethack_curiosity.envs.nethack.make_nethack import make_nethack
from nethack_curiosity.envs.nethack.nethack_env_params import (
    nethack_env_override_defaults,
    add_nethack_env_args,
)
from tests.envs.minigrid_argv import argv as nethack_argv


def test_run_nethack(evaluation=False):
    argv = ["--env", "nethack-score"]
    argv.extend(nethack_argv)
    register_env("nethack-score", make_nethack)
    parser, partial_cfg = parse_sf_args(argv, evaluation=evaluation)
    add_nethack_env_args("nethack-score", parser)
    nethack_env_override_defaults("nethack-score", parser)
    final_cfg = parse_full_cfg(parser, argv)
    status = run_rl(final_cfg)
    assert status == ExperimentStatus.SUCCESS
