from typing import Dict as DictType

import gymnasium as gym
from gymnasium.spaces import Dict, Discrete
import numpy as np

from sample_factory.utils.typing import Config


class MinigridVisitCountWrapper(gym.Wrapper):
    def __init__(self, env, cfg: Config):
        super().__init__(env)

        assert env.observation_space.__class__ == Dict
        # noinspection PyTypeChecker
        old_space: Dict = env.observation_space

        new_space: DictType = {"visit_count": Discrete(10000)}
        new_space.update([(k, old_space[k]) for k in old_space.keys()])
        self.observation_space = Dict(new_space)

        self.hash = self.define_hash()

        self.visit_counts = {}

    def define_hash(self):

        # noinspection PyTypeChecker
        obs_spaces: Dict = self.observation_space
        if "image" in obs_spaces.keys():
            return lambda obs: obs["image"].tobytes()
        else:
            raise ValueError(
                f"No hash function defined for observation space. Spaces are: {' '.join(obs_spaces.keys())}"
            )

    def reset(self, **kwargs):
        self.visit_counts = {}
        obs, info = self.env.reset(**kwargs)
        obs["visit_count"] = np.array([1], dtype=np.uint64)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state_hash = self.hash(obs)
        obs["visit_count"] = np.array(
            [self.visit_counts.get(state_hash, 1)], dtype=np.uint64
        )
        self.visit_counts[state_hash] = obs["visit_count"][0] + 1
        return obs, reward, terminated, truncated, info
