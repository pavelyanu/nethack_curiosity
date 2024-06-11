import pytest

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.context import global_model_factory

from nethack_curiosity.envs.minigrid.make_minigrid import make_minigrid
from nethack_curiosity.envs.minigrid.minigrid_env_params import (
    minigrid_env_override_defaults,
    add_minigrid_env_args,
)
from nethack_curiosity.models.minigrid_models import MinigridEncoder

from tests.envs.minigrid_test_cfg import minigrid_argv as minigrid_argv


def register_model_components():
    global_model_factory().register_encoder_factory(MinigridEncoder)


def parse_args():
    argv = ["--env", "minigrid-empty"]
    argv.extend(minigrid_argv)
    parser, partial_cfg = parse_sf_args(argv)
    add_minigrid_env_args("minigrid-empty", parser)
    minigrid_env_override_defaults("minigrid-empty", parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def test_run_minigrid():
    register_model_components()
    register_env("minigrid-empty", make_minigrid)
    cfg = parse_args()
    status = run_rl(cfg)
    assert status == ExperimentStatus.SUCCESS
