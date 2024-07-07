from typing import Dict as DictType
from collections import namedtuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete

from sample_factory.utils.typing import Config

BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)


class NethackDlvlWrapper(gym.Wrapper):
    def __init__(self, env, cfg: Config):
        super().__init__(env)

        assert env.observation_space.__class__ == Dict
        # noinspection PyTypeChecker
        old_space: Dict = env.observation_space
        assert "blstats" in old_space.spaces

        new_space: DictType = {"dlvl": Discrete(100)}
        new_space.update([(k, old_space[k]) for k in old_space.keys()])
        self.observation_space = Dict(new_space)

    def state_hash(self, obs) -> str:
        blstats = BLStats(*obs["blstats"])
        state = [getattr(blstats, attr) for attr in self.blstats]
        return "_".join(map(str, state))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["dlvl"] = np.array([BLStats(*obs["blstats"]).depth], dtype=np.uint64)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["dlvl"] = np.array([BLStats(*obs["blstats"]).depth], dtype=np.uint64)
        return obs, reward, terminated, truncated, info
