from typing import Dict as DictType, List
from collections import namedtuple

import gymnasium as gym
from gymnasium.spaces import Dict, Discrete
import numpy as np

from sample_factory.utils.typing import Config

BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)


class NethackVisitCountWrapper(gym.Wrapper):
    def __init__(self, env, cfg: Config):
        super().__init__(env)

        assert env.observation_space.__class__ == Dict
        # noinspection PyTypeChecker
        old_space: Dict = env.observation_space
        assert "blstats" in old_space.spaces

        new_space: DictType = {"visit_count": Discrete(10000)}
        new_space.update([(k, old_space[k]) for k in old_space.keys()])
        self.observation_space = Dict(new_space)

        self.blstats: List[str] = cfg.visit_count_blstats

        self.visit_counts = {}
        self.cfg = cfg

    def state_hash(self, obs) -> str:
        blstats = BLStats(*obs["blstats"])
        state = [getattr(blstats, attr) for attr in self.blstats]
        hash = "_".join(map(str, state))
        if obs.keys().__contains__("inv_letters") and obs.keys().__contains__(
            "inv_oclasses"
        ):
            letters = map(chr, obs["inv_letters"])
            oclasses = map(str, obs["inv_oclasses"])
            inventory = list(zip(letters, oclasses))
            inventory = sorted(inventory, key=lambda x: x[1])
            inventory = "".join(["".join(pair) for pair in inventory])
            hash += "_" + inventory
        return hash

    def reset(self, **kwargs):
        self.visit_counts = {}
        obs, info = self.env.reset(**kwargs)
        obs["visit_count"] = np.array([1], dtype=np.uint64)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state_hash = self.state_hash(obs)
        obs["visit_count"] = np.array(
            [self.visit_counts.get(state_hash, 1)], dtype=np.uint64
        )
        self.visit_counts[state_hash] = obs["visit_count"][0] + 1
        return obs, reward, terminated, truncated, info
