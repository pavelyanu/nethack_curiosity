import pytest

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.misc import ExperimentStatus

from nethack_curiosity.envs.minigrid.make_minigrid import make_minigrid
from nethack_curiosity.envs.minigrid.minigrid_env_params import (
    minigrid_env_override_defaults,
    add_minigrid_env_args,
)

from tests.envs.minigrid_argv import argv as minigrid_argv


def test_run_minigrid(evaluation=False):
    argv = ["--env", "minigrid-empty"]
    argv.extend(minigrid_argv)
    register_env("minigrid-empty", make_minigrid)
    parser, partial_cfg = parse_sf_args(argv, evaluation=evaluation)
    add_minigrid_env_args("minigrid-empty", parser)
    minigrid_env_override_defaults("minigrid-empty", parser)
    final_cfg = parse_full_cfg(parser, argv)
    status = run_rl(final_cfg)
    assert status == ExperimentStatus.SUCCESS
