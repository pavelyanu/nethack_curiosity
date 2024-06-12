import pytest

from sample_factory.algo.utils.misc import ExperimentStatus

from tests.train.empty_argv import argv
from nethack_curiosity.intrinsic_reward.train_ir import (
    make_ir_runner,
    run_ir_rl,
    ir_parse_full_cfg,
)
from tests.train.register_test_environments import register_empty_env


def test_parse_ir_args():
    evaluation = False
    cfg = ir_parse_full_cfg(argv, evaluation)
    assert cfg is not None
    assert cfg.intrinsic_reward_module == "mock"


def test_make_ir_runner():
    cfg = ir_parse_full_cfg(argv)
    cfg, runner = make_ir_runner(cfg)
    assert runner is not None


def test_ir_runner_init():
    register_empty_env()
    cfg = ir_parse_full_cfg(argv)
    cfg, runner = make_ir_runner(cfg)
    status = runner.init()
    assert status == ExperimentStatus.SUCCESS


# def test_run_ir_rl():
#     register_empty_env()
#     cfg = parse_ir_args(argv)
#     cfg, runner = make_ir_runner(cfg)
#     status = runner.init()
#     if status == ExperimentStatus.SUCCESS:
#         status = runner.run()
#     assert status == ExperimentStatus.SUCCESS
