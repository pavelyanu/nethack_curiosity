import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np

class MinigridVisitCountWrapper(gym.Wrapper):
    def __init__(self, env, hash=None):
        super().__init__(env)
        obs_spaces = {"visit_count": spaces.Discrete(10000)}
        obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
        self.observation_space = spaces.Dict(obs_spaces)
        self.hash = hash if hash is not None else self.define_hash()
        self.visit_counts = {}

    def define_hash(self):
        obs_spaces = self.env.observation_space
        try:
            shape = obs_spaces["image"].shape
            return lambda obs: obs["image"].tobytes()
        except:
            raise ValueError(f"No hash function defined for observation space. Spaces are: {' '.join(obs_spaces.keys())}")

    def reset(self, **kwargs):
        self.visit_counts = {}
        obs = self.env.reset(**kwargs)
        obs["visit_count"] = np.array([0], dtype=np.uint64)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        hash = self.hash(obs)
        obs["visit_count"] = np.array([self.visit_counts.get(hash, 0)], dtype=np.uint64)
        self.visit_counts[hash] = obs["visit_count"][0] + 1
        return obs, reward, done, info