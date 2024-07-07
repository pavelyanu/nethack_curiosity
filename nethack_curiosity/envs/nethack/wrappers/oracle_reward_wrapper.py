from collections import namedtuple

import gymnasium as gym

from sample_factory.utils.typing import Config

BLStats = namedtuple(
    "BLStats",
    "x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask align_bits",
)


class OracleRewardWrapper(gym.Wrapper):
    def __init__(self, env, cfg: Config):
        self.reward_win = cfg.oracle_reward_win
        self.reward_loss = cfg.oracle_reward_loss
        self.reward_penalty = cfg.oracle_reward_penalty
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        won = reward.__float__() > 0.0

        done = terminated or truncated

        if done:
            if won:
                reward = self.reward_win
            else:
                reward = self.reward_loss
        else:
            reward = self.reward_penalty

        return obs, reward, terminated, truncated, info
