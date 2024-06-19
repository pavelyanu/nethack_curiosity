from argparse import ArgumentParser


def nethack_env_override_defaults(
    env: str, parser: ArgumentParser, testing: bool = False
):
    parser.set_defaults(
        env_type="nethack",
    )


def add_nethack_env_args(env: str, parser: ArgumentParser, testing: bool = False):
    parser.add_argument(
        "--save_ttyrec_every",
        type=int,
        default=0,
        help="Integer, if 0, no ttyrecs (game recordings) will be saved. Otherwise, save a ttyrec every Nth episode.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default=None,
        help="Path to save ttyrecs (game recordings) into, if save_ttyrec_every is nonzero.",
    )
    parser.add_argument(
        "--character",
        type=str,
        default="mon-hum-neu-mal",
        help="Name of character. Defaults to 'mon-hum-neu-mal'.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=5000,
        help="Maximum amount of steps allowed before the game is forcefully quit. Defaults to 5000.",
    )
    parser.add_argument(
        "--observation_keys",
        type=str,
        nargs="+",
        default=[
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "screen_descriptions",
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
        type=bool,
        default=True,
        help="If False, disables normal NetHack behavior to randomly create monsters.",
    )
    parser.add_argument(
        "--penalty_mode",
        type=str,
        default="constant",
        choices=["constant", "exp", "square", "linear", "always"],
        help="Name of the mode for calculating the time step penalty. Defaults to 'constant'.",
    )
    parser.add_argument(
        "--penalty_step",
        type=float,
        default=-0.01,
        help="Constant applied to amount of frozen steps. Defaults to -0.01.",
    )
    parser.add_argument(
        "--penalty_time",
        type=float,
        default=-0.0,
        help="Constant applied to amount of frozen steps. Defaults to -0.0.",
    )
