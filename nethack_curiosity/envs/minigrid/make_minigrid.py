from typing import Optional

from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.minigrid_env import MiniGridEnv


def make_empty(render_mode: Optional[str] = None) -> MiniGridEnv:
    return EmptyEnv(render_mode=render_mode)


def make_multiroom(n: int, s: int, render_mode: Optional[str] = None) -> MiniGridEnv:
    return MultiRoomEnv(
        minNumRooms=n, maxNumRooms=n, maxRoomSize=s, render_mode=render_mode
    )


def make_minigrid(
    full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None
) -> MiniGridEnv:
    full_env_name = full_env_name.lower()
    if "multi" in full_env_name:
        n, s = parse_multiroom_env_name(full_env_name)
        return make_multiroom(n, s)
    elif "empty" in full_env_name:
        return make_empty()
    else:
        raise NotImplementedError(f"Unknown environment: {full_env_name}")


def parse_multiroom_env_name(full_env_name: str) -> tuple[int, int]:
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
