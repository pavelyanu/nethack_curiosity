from argparse import ArgumentParser


def intrinsic_reward_override_defaults(env: str, parser: ArgumentParser):
    pass


def add_intrinsic_reward_args(env: str, parser: ArgumentParser):
    parser.add_argument(
        "--intrinsic_reward_module",
        type=str,
        default="mock",
        help="Intrinsic reward module to use",
        choices=["mock", "none", "count", "curiosity", "rnd", "ride", "bebold"],
    )

    parser.add_argument(
        "--force_intrinsic_reward_components",
        action="store_true",
        help="Force the use of intrinsic reward components even if the intrinsic reward module is mock or none",
    )