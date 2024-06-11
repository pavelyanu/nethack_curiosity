from minigrid.wrappers import ObservationWrapper
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np

from minigrid.core.mission import MissionSpace

from sample_factory.utils.typing import Config


class MinigridTypeWrapper(ObservationWrapper):
    def __init__(self, env, cfg: Config):
        super().__init__(env)

        assert cfg.__contains__("observation_keys")
        self.observation_keys = cfg.observation_keys

        assert self.env.observation_space.__class__ == Dict
        # noinspection PyTypeChecker
        old_space: Dict = self.env.observation_space

        if "image" in old_space.keys():
            assert "image" in old_space.keys()
            assert old_space["image"].__class__ == Box

        if "direction" in old_space.keys():
            assert "direction" in old_space.keys()
            assert old_space["direction"].__class__ == Discrete

        if "mission" in old_space.keys():
            assert "mission" in old_space.keys()
            assert old_space["mission"].__class__ == MissionSpace

        for key in self.observation_keys:
            assert key in old_space.keys()

        new_space = {}
        for key in self.observation_keys:
            new_space[key] = old_space[key]

        self.observation_space = Dict(new_space)

    def observation(self, obs):
        new_obs = {}
        for key in self.observation_keys:
            if key == "direction":
                new_obs[key] = np.array([obs[key]], dtype=np.int64)
            else:
                new_obs[key] = obs[key]
        return new_obs
