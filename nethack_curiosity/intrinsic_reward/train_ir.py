from argparse import ArgumentParser
from typing import Tuple
from types import MethodType
import sys

from nethack_curiosity.intrinsic_reward.intrinsic_reward_algo.intrinsic_reward_buffer_mgr import (
    IntrinsicRewardBufferMgr,
    alloc_trajectory_tensors,
    policy_device,
)
import sample_factory.algo.utils.shared_buffers

shared_buffers = sys.modules["sample_factory.algo.utils.shared_buffers"]
shared_buffers.BufferMgr = IntrinsicRewardBufferMgr
shared_buffers.alloc_trajectory_tensors = alloc_trajectory_tensors
shared_buffers.policy_device = policy_device

from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.runners.runner import Runner
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg

from nethack_curiosity.intrinsic_reward.intrinsic_reward_algo.intrinsic_reward_learner_worker import (
    IntrinsicRewardLearnerWorker,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_params import (
    add_intrinsic_reward_args,
    intrinsic_reward_override_defaults,
)


def ir_parse_full_cfg(argv=None, evaluation=False):
    parser: ArgumentParser
    partial_cfg: Config
    parser, partial_cfg = parse_sf_args(argv, evaluation)
    add_intrinsic_reward_args(partial_cfg.env, parser)
    intrinsic_reward_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def parse_ir_args(argv=None):
    parser: ArgumentParser
    partial_cfg: Config
    parser, partial_cfg = parse_sf_args(argv)
    add_intrinsic_reward_args(partial_cfg.env, parser)
    intrinsic_reward_override_defaults(partial_cfg.env, parser)
    return parser, partial_cfg


def _make_learner(self, event_loop, policy_id: PolicyID, batcher: Batcher):
    return IntrinsicRewardLearnerWorker(
        event_loop,
        self.cfg,
        self.env_info,
        self.buffer_mgr,
        batcher,
        policy_id=policy_id,
    )


def make_ir_runner(cfg: Config) -> Tuple[Config, Runner]:

    cfg: Config
    runner: Runner
    cfg, runner = make_runner(cfg)

    assert (
        cfg.num_policies == 1
    ), "Intrinsic reward learning only supports a single policy"

    runner._make_learner = MethodType(_make_learner, runner)

    return cfg, runner


def run_ir_rl(cfg: Config):
    cfg, runner = make_ir_runner(cfg)
    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    return status
