import math

import torch
from torch import nn
from gymnasium.spaces import Dict, Box, Discrete, Space

from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.utils.typing import Config


class MinigridEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg)

        assert cfg.__contains__("observation_keys")
        self.observation_keys = cfg.observation_keys

        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        self.encoder_dict = {}

        for key in self.observation_keys:
            assert key in obs_space.keys()
            if key == "image":
                self.encoder_dict[key] = MinigridImageHead(cfg, obs_space)
            elif key == "direction":
                self.encoder_dict[key] = MinigridDirectionHead(cfg, obs_space)
            elif key == "mission":
                self.encoder_dict[key] = MinigridMissionHead(cfg, obs_space)
            else:
                raise NotImplementedError

        encoder_out_size = sum(
            [encoder.get_out_size() for encoder in self.encoder_dict.values()]
        )

        self.fc = nn.Sequential(
            nn.Linear(encoder_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = torch.cat(
            [encoder(obs[key]) for key, encoder in self.encoder_dict.items()], dim=1
        )
        return self.fc(x)

    def get_out_size(self) -> int:
        return 1024

    def model_to_device(self, device):
        for encoder in self.encoder_dict.values():
            encoder.to(device)
        self.fc.to(device)


class MinigridImageHead(Encoder):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg)

        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        assert "image" in obs_space.keys()
        assert obs_space["image"].__class__ == Box
        # noinspection PyTypeChecker
        image: Box = obs_space["image"]

        self.H, self.W, self.C = image.shape

        self.fc = nn.Sequential(
            nn.Conv2d(image.shape[-1], 32, 3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 128, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 512, 3, stride=2, padding=1),
            nn.ELU(),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.fc(x)
        x = x.reshape(x.size(0), -1)
        return x

    def get_out_size(self) -> int:
        # conv2d output shape: floor((H + 2P - K) / S + 1)
        # we have 1 conv2d with stride 1 and 2 conv2d with stride 2
        out_h = math.floor((self.H + 2 * 1 - 3) / 1 + 1)
        out_h = math.floor((out_h + 2 * 1 - 3) / 2 + 1)
        out_h = math.floor((out_h + 2 * 1 - 3) / 2 + 1)

        out_w = out_h

        return out_h * out_w * 512


class MinigridDirectionHead(Encoder):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg)

        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        assert "direction" in obs_space.keys()
        assert obs_space["direction"].__class__ == Discrete
        # noinspection PyTypeChecker
        direction: Discrete = obs_space["direction"]

        self.fc = nn.Embedding(direction.n, 32)

    def forward(self, x):
        x = x.long()
        return self.fc(x)

    def get_out_size(self) -> int:
        return 32


class MinigridMissionHead(Encoder):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg)
        raise NotImplementedError


class MinigridDecoder(Decoder):
    def __init__(self, cfg: Config, action_space: Space):
        super().__init__(cfg)

        assert action_space.__class__ == Discrete
        # noinspection PyTypeChecker
        action_space: Discrete = action_space

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n),
        )

    def forward(self, x):
        return self.fc(x)
