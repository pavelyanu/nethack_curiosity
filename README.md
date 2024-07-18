# Nethack Curiosity

This project provides the code to obtain the experimental results for my
Bachelor's thesis "Enhancing PPO with Intrinsic Rewards: A Study in the NetHack
Environment".

## Installation

- clone the repository with the submodules:
```
git clone --recursive https://github.com/pavelyanu/nethack_curiosity.git
```
- install nle dependencies
```
apt-get install build-essential python3-dev python3-pip python3-numpy autoconf libtool pkg-config libbz2-dev
conda install cmake flex bison lit
conda install -c conda-forge pybind11
```
- Install NLE as editable. From inside `externals/nle` run:
```
pip install -e .
```
- Install Sample Factory as editable. From inside `externals/sample-factory`
  run:
```
pip install -e .[nethack]
pip install -e sf_examples/nethack/nethack_render_utils
```
- Install nethack-curiosity as editable. From inside the root of the project
  run:
```
pip install -e .
```

## Running experiments

Custom experiments can be run through `nethack_curiosity/experiments/nethack/run_nethack.py`
python module as follows:
```
python3 nethack_curiosity/experiments/nethack/run_nethack.py --env score
```

## Replicating results reported in Bachelor's thesis

Directory `runs` contains several presets for the experiments in NetHack Score
and Oracle tasks. The results reported in "Enhancing PPO with Intrinsic Rewards:
A Study in the NetHack Environment" are obtained from
runs 0, 2, 4, 6, 8, 10, 12, 14, 16 and 19.

To run a specific configuration from `runs`  set the value in `ID` file inside the root of the project to the
id of the run and run `sweep.sh` from the root of the project.

#$ NetHack Environment Arguments

- `--rnd_encoder_type`: Type of the RND encoder. Choices: ["dwarven", "linear"]. Default: "dwarven"
- `--observation_keys`: Keys to use when creating the observation. Default: ["blstats", "message", "inv_letters", "inv_oclasses", "tty_chars", "tty_colors", "tty_cursor"]
- `--actions`: List of actions. If None, the full action space will be used.
- `--options`: List of game options to initialize Nethack. If None, Nethack will be initialized with default options.
- `--wizard`: Activate wizard mode. Default: False
- `--allow_all_yn_questions`: If set to True, no y/n questions in step() are declined. Default: False
- `--allow_all_modes`: If set to True, do not decline menus, text input or auto 'MORE'. Default: False
- `--spawn_monsters`: If False, disables normal NetHack behavior to randomly create monsters. Default: True
- `--oracle_reward_win`: Reward for winning the game. Default: 10.0
- `--oracle_reward_loss`: Reward for losing the game. Default: -10.0
- `--oracle_reward_penalty`: Reward for penalty. Default: 0.0
- `--character`: Name of character. Default: "mon-hum-neu-mal"
- `--max_episode_steps`: Maximum amount of steps allowed before the game is forcefully quit. Default: 100000
- `--penalty_step`: Constant applied to amount of frozen steps. Default: -0.01
- `--penalty_time`: Constant applied to amount of frozen steps. Default: 0.0
- `--penalty_mode`: Mode for calculating the time step penalty. Choices: ["constant", "exp", "square", "linear", "always"]. Default: "constant"
- `--savedir`: Path to save ttyrecs (game recordings) into. Default: None
- `--save_ttyrec_every`: Save a ttyrec every Nth episode. Default: 0
- `--add_image_observation`: If True, additional wrapper will render screen image. Default: True
- `--crop_dim`: Crop image around the player. Default: 18
- `--pixel_size`: Rescales each character to size of (pixel_size, pixel_size). Default: 6
- `--use_prev_action`: If True, the model will use previous action. Default: False
- `--use_tty_only`: If True, the model will use tty_chars for the topline and bottomline. Default: False
- `--h_dim`: Hidden dim for encoders. Default: 1738
- `--msg_hdim`: Hidden dim for message encoder. Default: 64
- `--color_edim`: Color Embedding Dim. Default: 16
- `--char_edim`: Char Embedding Dim. Default: 16
- `--use_crop`: Do we want to add additional embedding with cropped screen. Default: True
- `--use_crop_norm`: Do we want to use BatchNorm2d when processing cropped screen. Default: True
- `--screen_kernel_size`: Kernel size for screen convolutional encoder. Default: 3
- `--no_max_pool`: Do we want to use max pool in ResNet. Default: False
- `--screen_conv_blocks`: Number of blocks in ResNet. Default: 2
- `--blstats_hdim`: Hidden dim for blstats encoder. Default: 512
- `--fc_after_cnn_hdim`: Hidden dim for screen encoder. Default: 512
- `--use_resnet`: Do we want to use ResNet in screen encoder. Default: False
- `--visit_count_blstats`: List of blstats to use for visit count. Default: ["x", "y", "depth"]
- `--model`: Name of the model. Default: "ChaoticDwarvenGPT5"
- `--add_stats_to_info`: If True, adds wrapper which logs additional statistics. Default: True

## Intrinsic Reward Arguments

- `--intrinsic_reward_module`: Intrinsic reward module to use. Choices: ["mock", "none", "count", "curiosity", "rnd", "ride", "noveld", "inverse"]. Default: "rnd"
- `--rnd_share_encoder`: Share the encoder between the RND target and predictor networks. Default: False
- `--rnd_target_mlp_layers`: Number of hidden layers in the RND target head. Default: [1024, 1024, 1024, 1024]
- `--rnd_predictor_mlp_layers`: Number of hidden layers in the RND predictor head. Default: [1024, 1024, 1024]
- `--recompute_intrinsic_loss`: Recompute the intrinsic loss instead of using the stored value. Default: True
- `--rnd_blank_obs`: Blank out the observation before passing it to the RND module. Default: False
- `--rnd_random_obs`: Randomize the observation before passing it to the RND module. Default: False
- `--rnd_blank_target`: Blank out the target before passing it to the RND module. Default: False
- `--noveld_novelty_module`: Novelty module to use. Choices: ["rnd"]. Default: "rnd"
- `--noveld_constant_novelty`: Constant novelty instead of using the novelty module. Default: 0.0
- `--force_intrinsic_reward_components`: Force the use of intrinsic reward components even if the intrinsic reward module is mock or none. Default: False
- `--inverse_wiring`: Wiring of the inverse exploration model. Choices: ["icm", "ride"]. Default: "icm"
- `--visit_count_weighting`: Scheme to weight the intrinsic rewards based on visit count. Choices: ["none", "inverse_sqrt", "novel"]. Default: "none"
- `--inverse_action_mode`: Action encoding mode for the inverse model. Choices: ["onehot", "logits", "logprobs"]. Default: "onehot"
- `--inverse_loss_weight`: Weight of the inverse loss. Default: 1.0
- `--forward_loss_weight`: Weight of the forward loss. Default: 1.0
- `--env_type`: Environment type. Choices: ["nethack", "minigrid"]. Default: "nethack"
- `--intrinsic_reward_weight`: Weight of the intrinsic reward. Default: 0.5
- `--normalize_intrinsic_returns`: Normalize intrinsic returns. Default: True
- `--version`: Version (purely for logging purposes). Default: 1
- `--force_vanilla`: Force vanilla RL without intrinsic reward. Default: False

## Sample Factory Arguments

Refer to [Sample Factory Documentation](https://www.samplefactory.dev/02-configuration/cfg-params/).