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
