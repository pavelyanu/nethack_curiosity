from gymnasium import Space
from gymnasium.spaces import Dict
from torch import nn
import torch

from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config
from sf_examples.nethack.models.chaotic_dwarf import (
    MessageEncoder,
    BLStatsEncoder,
    TopLineEncoder,
    BottomLinesEncoder,
    InverseModel,
    ScreenEncoder,
    ChaoticDwarvenGPT5,
    calc_num_elements,
)

NethackEncoder = ChaoticDwarvenGPT5
NethackIntrinsicRewardEncoder = ChaoticDwarvenGPT5


class NethackRNDEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__(cfg)

        assert cfg.__contains__("observation_keys")
        self.observation_keys = cfg.observation_keys
        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        self.encoder_dict = nn.ModuleDict()

        # screen encoder needs only tty_chars and tty_colors keys
        self.encoder_dict["screen"] = RNDScreenEncoder(cfg, obs_space)
        self.encoder_dict["topline"] = MessageEncoder()
        self.encoder_dict["bottomline"] = BLStatsEncoder()
        topline_shape = obs_space["message"].shape
        bottomline_shape = obs_space["blstats"].shape

        if self.cfg.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.prev_actions_dim = 0
            self.num_actions = None

        assert obs_space.keys().__contains__("inv_letters")
        assert obs_space.keys().__contains__("inv_oclasses")
        self.inv_letters_dim = obs_space["inv_letters"].shape[0]
        self.inv_oclasses_dim = obs_space["inv_oclasses"].shape[0]

        self.encoder_dict["inv_letters"] = InvLettersEncoder(cfg, obs_space)
        self.encoder_dict["inv_oclasses"] = InvOClassesEncoder(cfg, obs_space)

        self.encoder_out_size = sum(
            [
                self.encoder_dict["screen"].get_out_size(),
                calc_num_elements(self.encoder_dict["topline"], topline_shape),
                calc_num_elements(self.encoder_dict["bottomline"], bottomline_shape),
                self.encoder_dict["inv_letters"].get_out_size(),
                self.encoder_dict["inv_oclasses"].get_out_size(),
                self.prev_actions_dim,
            ]
        )

    def forward(self, obs_dict):
        B, C, H, W = obs_dict["screen_image"].shape

        topline = obs_dict["message"]
        bottom_line = obs_dict["blstats"]
        inv_letters = obs_dict["inv_letters"]
        inv_oclasses = obs_dict["inv_oclasses"]

        encodings = [
            self.encoder_dict["topline"](
                topline.float(memory_format=torch.contiguous_format).view(B, -1)
            ),
            self.encoder_dict["bottomline"](
                bottom_line.float(memory_format=torch.contiguous_format).view(B, -1)
            ),
            self.encoder_dict["screen"](obs_dict),
            self.encoder_dict["inv_letters"](
                inv_letters.float(memory_format=torch.contiguous_format).view(B, -1)
            ),
            self.encoder_dict["inv_oclasses"](
                inv_oclasses.float(memory_format=torch.contiguous_format).view(B, -1)
            ),
        ]

        if self.cfg.use_prev_action:
            prev_actions = obs_dict["prev_actions"].long().view(B)
            encodings.append(
                torch.nn.functional.one_hot(prev_actions, self.num_actions)
            )

        return torch.cat(encodings, dim=1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class RNDScreenEncoder(nn.Module):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__()

        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        assert "tty_chars" in obs_space.keys()
        assert "tty_colors" in obs_space.keys()
        assert "tty_cursor" in obs_space.keys()

        self.h = obs_space["tty_chars"].shape[0]
        self.w = obs_space["tty_chars"].shape[1]

        self.tty_dim = self.h * self.w

        self.tty_cursor_dim = obs_space["tty_cursor"].shape[0]

        self.hidden_size = 512

        self.tty_char_range = 128.0
        self.tty_char_start = 32.0
        self.tty_color_range = 16.0

        self.fc = nn.Sequential(
            nn.Linear(
                self.tty_dim + self.tty_dim + self.tty_cursor_dim,
                self.hidden_size,
            ),
            nn.ReLU(),
        )

    def forward(self, obs_dict):

        B, H, W = obs_dict["tty_chars"].shape

        tty_chars = obs_dict["tty_chars"]
        tty_chars = tty_chars.view(B, H * W)
        tty_chars = tty_chars - self.tty_char_start
        tty_chars = tty_chars / self.tty_char_range

        tty_colors = obs_dict["tty_colors"]
        tty_colors = tty_colors.view(B, H * W)
        tty_colors = tty_colors / self.tty_color_range

        tty_cursor = obs_dict["tty_cursor"]

        x = torch.cat([tty_chars, tty_colors, tty_cursor], dim=1)

        x = self.fc(x)

        return x

    def get_out_size(self) -> int:
        return self.hidden_size


class InvLettersEncoder(nn.Module):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__()

        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        assert "inv_letters" in obs_space.keys()
        self.inv_letters_dim = obs_space["inv_letters"].shape[0]

        self.hidden_size = 512

        self.inv_letters_range = 128.0

        self.fc = nn.Sequential(
            nn.Linear(self.inv_letters_dim, self.hidden_size),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = obs / self.inv_letters_range
        x = self.fc(x)
        return x

    def get_out_size(self) -> int:
        return self.hidden_size


class InvOClassesEncoder(nn.Module):
    def __init__(self, cfg: Config, obs_space: Space):
        super().__init__()

        assert obs_space.__class__ == Dict
        # noinspection PyTypeChecker
        obs_space: Dict = obs_space

        assert "inv_oclasses" in obs_space.keys()
        self.inv_oclasses_dim = obs_space["inv_oclasses"].shape[0]

        self.hidden_size = 512

        self.oclasses_range = 18.0

        self.fc = nn.Sequential(
            nn.Linear(self.inv_oclasses_dim, self.hidden_size),
            nn.ReLU(),
        )

    def forward(self, obs):
        x = obs / self.oclasses_range
        x = self.fc(x)
        return x

    def get_out_size(self) -> int:
        return self.hidden_size
