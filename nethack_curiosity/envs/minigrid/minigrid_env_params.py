from argparse import ArgumentParser


def minigrid_env_override_defaults(env: str, parser: ArgumentParser):
    pass


def add_minigrid_env_args(env: str, parser: ArgumentParser):
    parser.add_argument(
        "--observation_keys",
        type=str,
        nargs="+",
        default=["image"],
    )
