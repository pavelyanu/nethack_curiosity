from typing import Optional

from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.minigrid_env import MiniGridEnv

from nethack_curiosity.envs.minigrid.wrappers import __required__, __global_order__
from sample_factory.utils.typing import Config


def make_empty(render_mode: Optional[str] = None) -> MiniGridEnv:
    return EmptyEnv(render_mode=render_mode)


def make_multiroom(n: int, s: int, render_mode: Optional[str] = None) -> MiniGridEnv:
    return MultiRoomEnv(
        minNumRooms=n, maxNumRooms=n, maxRoomSize=s, render_mode=render_mode
    )


def make_minigrid(
    full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None
) -> MiniGridEnv:
    env = _make_minigrid(full_env_name)
    if cfg is not None:
        env = apply_required_wrappers(env, cfg)
    return env


def _make_minigrid(name: str) -> MiniGridEnv:
    name = name.lower()
    env: MiniGridEnv

    if "multi" in name:
        try:
            n, s = parse_multiroom_env_name(name)
        except Exception:
            raise ValueError(
                f"Invalid environment name: {name}. Must be in the format 'multiroom-N#-S#'"
            )
        env = make_multiroom(n, s)
    elif "empty" in name:
        env = make_empty()
    else:
        raise NotImplementedError(f"Unknown environment: {name}")

    return env


def apply_required_wrappers(env: MiniGridEnv, cfg: Config) -> MiniGridEnv:
    for wrapper in __global_order__:
        if wrapper in __required__:
            env = wrapper(env, cfg)
    return env


def parse_multiroom_env_name(full_env_name: str) -> tuple[int, int]:
    full_env_name = full_env_name.lower()
    n = get_split_containing_string(full_env_name, "n")
    n = n.replace("n", "")
    n = int(n)
    s = get_split_containing_string(full_env_name, "s")
    s = s.replace("s", "")
    s = int(s)
    return n, s


def get_split_containing_string(full_env_name: str, string: str) -> str:
    splits = full_env_name.split("-")
    for split in splits:
        if string in split:
            return split
