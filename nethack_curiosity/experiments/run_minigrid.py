from nethack_curiosity.intrinsic_reward.train_ir import run_ir_rl, parse_ir_args
from sample_factory.cfg.arguments import parse_full_cfg
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.context import global_model_factory

from nethack_curiosity.envs.minigrid.make_minigrid import make_minigrid
from nethack_curiosity.envs.minigrid.minigrid_env_params import (
    minigrid_env_override_defaults,
    add_minigrid_env_args,
)
from nethack_curiosity.models.minigrid_models import MinigridEncoder


def register_model_components():
    global_model_factory().register_encoder_factory(MinigridEncoder)


def parse_args(argv=None):
    parser, partial_cfg = parse_ir_args(argv)
    add_minigrid_env_args("minigrid", parser, testing=False)
    minigrid_env_override_defaults("minigrid", parser, testing=False)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    register_model_components()
    register_env("keycorridor", make_minigrid)
    register_env("empty", make_minigrid)
    register_env("multiroom", make_minigrid)
    register_env("lockedroom", make_minigrid)
    register_env("obstructedmaze_2dlh", make_minigrid)
    cfg = parse_args()
    status = run_ir_rl(cfg)
    assert status == ExperimentStatus.SUCCESS


if __name__ == "__main__":
    main()
