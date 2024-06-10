from minigrid.wrappers import ObservationWrapper
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np

from minigrid.core.mission import MissionSpace


class MinigridTypeWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert self.env.observation_space.__class__ == Dict
        # noinspection PyTypeChecker
        old_space: Dict = self.env.observation_space

        assert "image" in old_space.keys()
        assert old_space["image"].__class__ == Box

        assert "direction" in old_space.keys()
        assert old_space["direction"].__class__ == Discrete

        assert "mission" in old_space.keys()
        assert old_space["mission"].__class__ == MissionSpace

        self.observation_space = Dict(
            {"image": old_space["image"], "direction": old_space["direction"]}
        )

    def observation(self, obs):
        return {
            "image": obs["image"],
            "direction": obs["direction"],
        }
