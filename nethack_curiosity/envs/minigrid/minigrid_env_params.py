from argparse import ArgumentParser


def minigrid_env_override_defaults(env: str, parser: ArgumentParser):
    parser.set_defaults(
        num_workers=20,
        num_envs_per_worker=2,
        # train_for_env_steps=500000000,
        train_for_env_steps=1000000,
        batch_size=32,
        rollout=100,
        max_grad_norm=40,
        exploration_loss="entropy",
        exploration_loss_coeff=0.001,
        value_loss_coeff=0.5,
        gamma=0.99,
        learning_rate=0.0001,
    )


def add_minigrid_env_args(env: str, parser: ArgumentParser):
    parser.add_argument(
        "--observation_keys",
        type=str,
        nargs="+",
        default=["image"],
    )
