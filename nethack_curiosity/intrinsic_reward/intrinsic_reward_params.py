from argparse import ArgumentParser


def intrinsic_reward_override_defaults(env: str, parser: ArgumentParser):
    pass


def add_intrinsic_reward_args(env: str, parser: ArgumentParser):
    parser.add_argument(
        "--intrinsic_reward_module",
        type=str,
        default="rnd",
        help="Intrinsic reward module to use",
        choices=["mock", "none", "count", "curiosity", "rnd", "ride", "noveld"],
    )

    parser.add_argument(
        "--rnd_share_encoder",
        type=bool,
        default=False,
        help="Share the encoder between the RND target and predictor networks",
    )

    parser.add_argument(
        "--rnd_mlp_layers",
        type=int,
        nargs="+",
        default=[32],
        help="Number of hidden layers in the RND MLP",
    )

    parser.add_argument(
        "--rnd_recompute_loss",
        type=bool,
        default=True,
        help="Recompute the loss for the RND module instead of using the one computed during the forward pass",
    )

    parser.add_argument(
        "--rnd_blank_obs",
        type=bool,
        default=False,
        help="Blank out the observation before passing it to the RND module. For debugging purposes.",
    )

    parser.add_argument(
        "--noveld_novelty_module",
        type=str,
        default="rnd",
        help="Novelty module to use",
        choices=["rnd"],
    )

    parser.add_argument(
        "--noveld_constant_novelty",
        type=float,
        default=0.0,
        help="Constant novelty instead of using the novelty module. Set to 0 to use the novelty module.",
    )

    parser.add_argument(
        "--force_intrinsic_reward_components",
        action="store_true",
        help="Force the use of intrinsic reward components even if the intrinsic reward module is mock or none",
    )

    parser.add_argument(
        "--env_type",
        type=str,
        default="minigrid",
        choices=["nethack", "minigrid"],
    )

    parser.add_argument(
        "--intrinsic_reward_weight",
        type=float,
        default=1.0,
        help="Weight of the intrinsic reward",
    )

    parser.add_argument(
        "--normalize_intrinsic_returns",
        type=bool,
        default=True,
        help="Normalize intrinsic returns",
    )

    parser.add_argument(
        "--version", type=int, default=1, help="Version. Purely for logging purposes."
    )
