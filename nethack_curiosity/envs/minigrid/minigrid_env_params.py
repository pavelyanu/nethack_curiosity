from argparse import ArgumentParser


def minigrid_env_override_defaults(
    env: str, parser: ArgumentParser, testing: bool = False
):
    if testing:
        return
    parser.set_defaults(
        # train_for_env_steps=50000000,
        num_workers=16,
        num_envs_per_worker=32,
        worker_num_splits=2,
        env_type="minigrid",
        # with_wandb=True,
        # wandb_user="pyanushonak",
        # wandb_project="ppo_curiosity",
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
        default=5,
    )

    parser.add_argument(
        "--minigrid_room_num",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--minigrid_row_num",
        type=int,
        default=5,
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
