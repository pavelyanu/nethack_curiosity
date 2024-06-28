from typing import Optional

from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.keycorridor import KeyCorridorEnv
from minigrid.envs.lockedroom import LockedRoomEnv
from minigrid.envs.obstructedmaze import ObstructedMaze_2Dlh
from minigrid.minigrid_env import MiniGridEnv

from nethack_curiosity.envs.minigrid.wrappers import __required__, __global_order__
from nethack_curiosity.envs.minigrid.wrappers.minigrid_visit_count import (
    MinigridVisitCountWrapper,
)
from sample_factory.utils.typing import Config


def make_empty(
    size: int = 5, max_steps: int = 100, render_mode: Optional[str] = None
) -> MiniGridEnv:
    return EmptyEnv(size=size, render_mode=render_mode)


def make_multiroom(
    n: int, s: int, max_steps: int = 100, render_mode: Optional[str] = None
) -> MiniGridEnv:
    return MultiRoomEnv(
        minNumRooms=n, maxNumRooms=n, maxRoomSize=s, render_mode=render_mode
    )


def make_keycorridor(
    s: int, r: int, max_steps: int = 100, render_mode: Optional[str] = None
) -> MiniGridEnv:
    return KeyCorridorEnv(room_size=s, num_rows=r, render_mode=render_mode)


def make_lockedroom(
    s: int = 19, max_steps: int = 100, render_mode: Optional[str] = None
) -> MiniGridEnv:
    return LockedRoomEnv(size=s, max_steps=max_steps, render_mode=render_mode)


def make_obstructedmaze_2dlh(render_mode: Optional[str] = None) -> MiniGridEnv:
    return ObstructedMaze_2Dlh(render_mode=render_mode)


def make_minigrid(
    full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None
) -> MiniGridEnv:
    env = _make_minigrid(
        full_env_name,
        cfg.minigrid_room_size,
        cfg.minigrid_room_num,
        cfg.minigrid_row_num,
        cfg.minigrid_max_steps,
    )
    if (
        cfg.intrinsic_reward_module == "noveld"
        or cfg.intrinsic_reward_module == "inverse"
    ):
        __required__.append(MinigridVisitCountWrapper)
    if cfg is not None:
        env = apply_required_wrappers(env, cfg)
    return env


def _make_minigrid(
    name: str,
    room_size: int = 5,
    room_num: int = 5,
    row_num: int = 5,
    max_steps: int = 100,
) -> MiniGridEnv:
    name = name.lower()
    env: MiniGridEnv
    if "multi" in name:
        n = room_num
        s = room_size
        env = make_multiroom(n, s, max_steps=max_steps)
    elif "empty" in name:
        n = room_size
        env = make_empty(n, max_steps=max_steps)
    elif "keycorridor" in name:
        s = room_size
        r = row_num
        env = make_keycorridor(s, r, max_steps=max_steps)
    elif "lockedroom" in name:
        s = room_size
        env = make_lockedroom(s, max_steps=max_steps)
    elif "obstructedmaze_2dlh" in name:
        env = make_obstructedmaze_2dlh()
    else:
        raise NotImplementedError(f"Unknown environment: {name}")

    return env


def apply_required_wrappers(env: MiniGridEnv, cfg: Config) -> MiniGridEnv:
    for wrapper in __global_order__:
        if wrapper in __required__:
            env = wrapper(env, cfg)
    return env
