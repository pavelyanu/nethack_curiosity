from argparse import ArgumentParser
from sample_factory.utils.utils import str2bool


def add_nethack_env_args(env: str, parser: ArgumentParser, testing: bool = False):
    add_my_nethack_env_args(env, parser, testing)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_general(parser)


def add_my_nethack_env_args(env: str, parser: ArgumentParser, testing: bool = False):
    parser.add_argument(
        "--rnd_encoder_type",
        type=str,
        default="dwarven",
        choices=["dwarven", "linear"],
        help="Type of the RND encoder. Defaults to `dwarven`.",
    )
    parser.add_argument(
        "--observation_keys",
        type=str,
        nargs="+",
        default=[
            # "glyphs",
            # "chars",
            # "colors",
            # "specials",
            "blstats",
            "message",
            # "inv_glyphs",
            # "inv_strs",
            "inv_letters",
            "inv_oclasses",
            # "screen_descriptions",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
        ],
        help="Keys to use when creating the observation. Defaults to all.",
    )
    parser.add_argument(
        "--actions",
        type=str,
        nargs="+",
        default=None,
        help="List of actions. If None, the full action space will be used.",
    )
    parser.add_argument(
        "--options",
        type=str,
        nargs="+",
        default=None,
        help="List of game options to initialize Nethack. If None, Nethack will be initialized with default options.",
    )
    parser.add_argument(
        "--wizard", action="store_true", help="Activate wizard mode. Defaults to False."
    )
    parser.add_argument(
        "--allow_all_yn_questions",
        action="store_true",
        help="If set to True, no y/n questions in step() are declined. Defaults to False.",
    )
    parser.add_argument(
        "--allow_all_modes",
        action="store_true",
        help="If set to True, do not decline menus, text input or auto 'MORE'. Defaults to False.",
    )
    parser.add_argument(
        "--spawn_monsters",
        type=str2bool,
        default=True,
        help="If False, disables normal NetHack behavior to randomly create monsters.",
    )
    parser.add_argument(
        "--oracle_reward_win",
        type=float,
        default=1.0,
        help="Reward for winning the game. Defaults to 1.0.",
    )
    parser.add_argument(
        "--oracle_reward_loss",
        type=float,
        default=-1.0,
        help="Reward for losing the game. Defaults to -1.0.",
    )
    parser.add_argument(
        "--oracle_reward_penalty",
        type=float,
        default=0.0,
        help="Reward for penalty. Defaults to 0.0.",
    )


def add_extra_params_nethack_env(parser):
    """
    Specify any additional command line arguments for NetHack environments.
    """
    p = parser
    p.add_argument(
        "--character",
        type=str,
        default="mon-hum-neu-mal",
        help="name of character. Defaults to 'mon-hum-neu-mal'.",
    )
    p.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum amount of steps allowed before the game is forcefully quit. In such cases, `info 'end_status']` will be equal to `StepStatus.ABORTED`",
    )
    p.add_argument(
        "--penalty_step",
        type=float,
        default=0.0,
        help="constant applied to amount of frozen steps. Defaults to 0.0.",
    )
    p.add_argument(
        "--penalty_time",
        type=float,
        default=0.0,
        help="constant applied to amount of frozen steps. Defaults to 0.0.",
    )
    p.add_argument(
        "--penalty_mode",
        type=str,
        default="constant",
        help="name of the mode for calculating the time step penalty. Can be `constant`, `exp`, `square`, `linear`, or `always`. Defaults to `constant`.",
    )
    p.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Path to save ttyrecs (game recordings) into, if save_ttyrec_every is nonzero. If nonempty string, interpreted as a path to a new or existing directory. If "
        " (empty string) or None, NLE choses a unique directory name.Defaults to `None`.",
    )
    p.add_argument(
        "--save_ttyrec_every",
        type=int,
        default=0,
        help="Integer, if 0, no ttyrecs (game recordings) will be saved. Otherwise, save a ttyrec every Nth episode.",
    )
    p.add_argument(
        "--add_image_observation",
        type=str2bool,
        default=True,
        help="If True, additional wrapper will render screen image. Defaults to `True`.",
    )
    p.add_argument(
        "--crop_dim",
        type=int,
        default=18,
        help="Crop image around the player. Defaults to `18`.",
    )
    p.add_argument(
        "--pixel_size",
        type=int,
        default=6,
        help="Rescales each character to size of `(pixel_size, pixel_size). Defaults to `6`.",
    )


def add_extra_params_model(parser):
    """
    Specify any additional command line arguments for NetHack models.
    """
    p = parser
    p.add_argument(
        "--use_prev_action",
        type=str2bool,
        default=False,
        help="If True, the model will use previous action. Defaults to `True`",
    )
    p.add_argument(
        "--use_tty_only",
        type=str2bool,
        default=True,
        help="If True, the model will use tty_chars for the topline and bottomline. Defaults to `True`",
    )
    # parameters specific to ScaledNet
    p.add_argument(
        "--h_dim",
        type=int,
        default=1738,
        help="Hidden dim for encoders. Defaults to `1738`",
    )
    p.add_argument(
        "--msg_hdim",
        type=int,
        default=64,
        help="Hidden dim for message encoder. Defaults to `64`",
    )
    p.add_argument(
        "--color_edim",
        type=int,
        default=16,
        help="Color Embedding Dim. Defaults to `16`",
    )
    p.add_argument(
        "--char_edim", type=int, default=16, help="Char Embedding Dim. Defaults to `16`"
    )
    p.add_argument(
        "--use_crop",
        type=str2bool,
        default=True,
        help="Do we want to add additional embedding with cropped screen. Defaults to `True`",
    )
    p.add_argument(
        "--use_crop_norm",
        type=str2bool,
        default=True,
        help="Do we want to use BatchNorm2d when processing cropped screen. Defaults to `True`",
    )
    p.add_argument(
        "--screen_kernel_size",
        type=int,
        default=3,
        help="Kernel size for screen convolutional encoder. Defaults to `3`",
    )
    p.add_argument(
        "--no_max_pool",
        type=str2bool,
        default=False,
        help="Do we want to use max pool in ResNet. Defaults to `False`",
    )
    p.add_argument(
        "--screen_conv_blocks",
        type=int,
        default=2,
        help="Number of blocks in ResNet. Defaults to `2`",
    )
    p.add_argument(
        "--blstats_hdim",
        type=int,
        default=512,
        help="Hidden dim for blstats encoder. Defaults to `512`",
    )
    p.add_argument(
        "--fc_after_cnn_hdim",
        type=int,
        default=512,
        help="Hidden dim for screen encoder. Defaults to `512`",
    )
    p.add_argument(
        "--use_resnet",
        type=str2bool,
        default=False,
        help="Do we want to use ResNet in screen encoder. Defaults to `False`",
    )
    p.add_argument(
        "--visit_count_blstats",
        type=str,
        nargs="+",
        default=["x", "y", "depth"],
        help="List of blstats to use for visit count. Defaults to ['x', 'y', 'depth']",
    )


def add_extra_params_general(parser):
    """
    Specify any additional command line arguments for NetHack.
    """
    p = parser
    p.add_argument(
        "--model",
        type=str,
        default="ChaoticDwarvenGPT5",
        help="Name of the model. Defaults to `ChaoticDwarvenGPT5`.",
    )
    p.add_argument(
        "--add_stats_to_info",
        type=str2bool,
        default=True,
        help="If True, adds wrapper which loggs additional statisics. Defaults to `True`.",
    )


def nethack_env_override_defaults(_env, parser):
    """RL params specific to NetHack envs."""
    # set hyperparameter values to the same as in d&d
    parser.set_defaults(
        env_type="nethack",
        use_record_episode_statistics=False,
        gamma=0.999,
        num_workers=16,
        num_envs_per_worker=32,
        worker_num_splits=2,
        train_for_env_steps=2_000_000_000,
        nonlinearity="relu",
        use_rnn=True,
        rnn_type="lstm",
        actor_critic_share_weights=True,
        policy_initialization="orthogonal",
        policy_init_gain=1.0,
        adaptive_stddev=False,  # True only for continous action distributions
        reward_scale=1.0,
        reward_clip=10.0,
        batch_size=2048,
        rollout=128,
        max_grad_norm=4,
        num_epochs=1,
        num_batches_per_epoch=4,  # can be used for increasing the batch_size for SGD
        ppo_clip_ratio=0.1,
        ppo_clip_value=1.0,
        value_loss_coeff=1.0,
        exploration_loss="entropy",
        exploration_loss_coeff=0.001,
        learning_rate=0.0001,
        gae_lambda=1.0,
        with_vtrace=False,  # in d&d they've used vtrace
        normalize_input=True,  # turn off for now and use normalization from d&d
        normalize_input_keys=["screen", "tty_cursor"],
        normalize_returns=True,
        async_rl=True,
        experiment_summaries_interval=50,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-7,
        seed=22,
        save_every_sec=120,
    )
