import torch
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.base import (
    IntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.mock import (
    MockIntrinsicRewardModule,
)
from nethack_curiosity.intrinsic_reward.intrinsic_reward_modules.rnd import (
    RNDIntrinsicRewardModule,
)

from nethack_curiosity.envs.minigrid.make_minigrid import make_empty
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config


def test_rnd_parameters():
    env = make_empty()
    obs_space = env.observation_space
    cfg = AttrDict({"env_type": "minigrid", "observation_keys": ["image"]})
    rnd = RNDIntrinsicRewardModule(cfg, obs_space)
    params = list(rnd.parameters())
    assert len(params) != 0


def test_rnd_learning():
    env = make_empty(5)
    obs_space = env.observation_space
    cfg = AttrDict({"env_type": "minigrid", "observation_keys": ["image"]})
    rnd = RNDIntrinsicRewardModule(cfg, obs_space)
    optimizer = torch.optim.Adam(rnd.parameters(), lr=1e-3)
    fake_obs = torch.rand(32, 5, 5, 3)
    fake_obs = TensorDict({"image": fake_obs})
    with torch.no_grad():
        old_target_embedding = rnd.target(fake_obs)
    for _ in range(1000):
        fake_rewards = torch.rand(32)
        mb = AttrDict(
            {
                "normalized_obs": fake_obs,
                "rewards_cpu": fake_rewards,
            }
        )
        optimizer.zero_grad()
        target_embedding = rnd.target(fake_obs)
        predictor_embedding = rnd.predictor(fake_obs)
        error = target_embedding - predictor_embedding
        loss = error.pow(2).sum(1).mean()
        loss.backward()
        optimizer.step()
        assert torch.allclose(target_embedding, old_target_embedding)
