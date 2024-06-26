from argparse import ArgumentParser


def minigrid_env_override_defaults(
    env: str, parser: ArgumentParser, testing: bool = False
):
    if testing:
        return
    parser.set_defaults(
        train_for_env_steps=10_000_000,
        env_type="minigrid",
        normalize_input_keys=["image"],
    )


def add_minigrid_env_args(env: str, parser: ArgumentParser, testing: bool = False):
    parser.add_argument(
        "--observation_keys",
        type=str,
        nargs="+",
        default=["image"],
        choices=["image", "direction"],
    )

    parser.add_argument(
        "--minigrid_reproduce",
        type=str,
        default="vanilla",
        choices=["vanilla", "count", "curiosity", "rnd", "ride", "bebold"],
    )

    parser.add_argument(
        "--minigrid_room_size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minigrid_room_num",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minigrid_row_num",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--image_head",
        type=str,
        default="conv",
        choices=["conv", "flat"],
    )

    parser.add_argument(
        "--ir_image_head",
        type=str,
        default="conv",
        choices=["conv", "flat"],
    )

    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=256,
        help="Hidden size of the encoder",
    )
