import pytest

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.context import global_model_factory

from nethack_curiosity.envs.nethack.make_nethack import make_nethack
from nethack_curiosity.envs.nethack.nethack_env_params import (
    nethack_env_override_defaults,
    add_nethack_env_args,
)
from tests.envs.minigrid_test_cfg import argv as nethack_argv


def register_model_components():
    pass


def parse_args():
    argv = ["--env", "nethack-score"]
    argv.extend(nethack_argv)
    parser, partial_cfg = parse_sf_args(argv)
    add_nethack_env_args("nethack-score", parser)
    nethack_env_override_defaults("nethack-score", parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def test_run_nethack():
    register_model_components()
    register_env("nethack-score", make_nethack)
    cfg = parse_args()
    status = run_rl(cfg)
    assert status == ExperimentStatus.SUCCESS
