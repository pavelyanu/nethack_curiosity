import yaml
import itertools


def create_config(params):
    config = {
        "restart_behavior": "resume",
        "env": params["env"],
        "max_episode_steps": params["max_episode_steps"],
        "intrinsic_reward_module": params["intrinsic_reward_module"],
        "character": params["character"],
        "intrinsic_reward_weight": params["intrinsic_reward_weight"],
    }

    if params["intrinsic_reward_module"] == "inverse":
        config["inverse_wiring"] = params["inverse_wiring"]
        config["visit_count_weighting"] = params["visit_count_weighting"]

    return config


def get_filename(params):
    parts = [
        f"env-{params['env']}",
        f"ep-{params['max_episode_steps']}",
        f"ir-{params['intrinsic_reward_module']}",
        f"ch-{params['character'].split('-')[1]}",
    ]

    if params["intrinsic_reward_module"] == "inverse":
        parts.append(f"iw-{params['inverse_wiring']}")
        parts.append(f"vcw-{params['visit_count_weighting']}")

    return "_".join(parts) + ".yaml"


grid = {
    "env": ["oracle", "score"],
    "max_episode_steps": [50000, 100000],
    "intrinsic_reward_module": ["mock", "rnd", "inverse", "noveld"],
    "inverse_wiring": ["icm", "ride"],
    "visit_count_weighting": ["none", "inverse_sqrt"],
    "character": ["mon-hum-neu-mal", "val-dwa-law-fem"],
}

combinations = list(itertools.product(*grid.values()))

for i in range(len(combinations) - 1, -1, -1):
    if combinations[i] in combinations[:i]:
        del combinations[i]


files = {}

for combo in combinations:
    params = dict(zip(grid.keys(), combo))

    if params["env"] == "oracle":
        params["intrinsic_reward_weight"] = 1.0
    if params["env"] == "score":
        params["intrinsic_reward_weight"] = 0.5
    if params["intrinsic_reward_module"] == "mock":
        params["intrinsic_reward_weight"] = 0.0
    if params["env"] == "oracle" and params["max_episode_steps"] != 50000:
        continue
    if params["env"] == "score" and params["max_episode_steps"] != 100000:
        continue
    if params["intrinsic_reward_module"] != "inverse":
        params["inverse_wiring"] = None
        params["visit_count_weighting"] = None
    if params["intrinsic_reward_module"] == "inverse":
        if (
            params["inverse_wiring"] == "icm"
            and params["visit_count_weighting"] != "none"
        ):
            continue
        if (
            params["inverse_wiring"] == "ride"
            and params["visit_count_weighting"] == "none"
        ):
            continue

    config = create_config(params)
    filename = get_filename(params)
    files[filename] = config

counter = 0

for filename, config in files.items():
    filename = f"{counter:02}_{filename}"
    counter += 1
    with open(filename, "w") as f:
        yaml.dump(config, f)


print("YAML files created successfully.")
