from sample_factory.envs.env_utils import register_env

from nethack_curiosity.envs.minigrid.make_minigrid import make_minigrid


def register_empty_env():
    register_env("minigrid-empty", make_minigrid)
