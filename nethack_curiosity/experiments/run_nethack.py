from nethack_curiosity.intrinsic_reward.train_ir import run_ir_rl, parse_ir_args
from nethack_curiosity.models.intrinsic_reward_actor_critic import (
    make_intrinsic_reward_actor_critic,
)
from sample_factory.cfg.arguments import parse_full_cfg
from sample_factory.envs.env_utils import register_env
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.train import run_rl

from nethack_curiosity.envs.nethack.make_nethack import make_nethack
from nethack_curiosity.envs.nethack.nethack_env_params import (
    nethack_env_override_defaults,
    add_nethack_env_args,
)
from nethack_curiosity.models.nethack_models import NethackEncoder


def register_model_components():
    global_model_factory().register_actor_critic_factory(
        make_intrinsic_reward_actor_critic
    )
    global_model_factory().register_encoder_factory(NethackEncoder)


def parse_args(argv=None):
    parser, partial_cfg = parse_ir_args(argv)
    add_nethack_env_args("nethack", parser, testing=False)
    nethack_env_override_defaults("nethack", parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    register_model_components()
    env_list = [
        "score",
        "staircase",
        "staircase_pet",
        "oracle",
        "gold",
        "eat",
        "scout",
        "challenge",
    ]
    for env_name in env_list:
        register_env(env_name, make_nethack)
    cfg = parse_args()
    status = run_ir_rl(cfg)
    # status = run_rl(cfg)
    assert status == ExperimentStatus.SUCCESS


if __name__ == "__main__":
    main()
