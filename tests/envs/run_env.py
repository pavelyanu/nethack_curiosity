from gymnasium import Env


def run_for_n_few_steps(env: Env, n: int = 10):
    env.reset()
    for _ in range(n):
        env.step(env.action_space.sample())
    env.close()
