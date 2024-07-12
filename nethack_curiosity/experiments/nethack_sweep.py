import os
import yaml
from typing import Dict, List
from sample_factory.launcher.run_description import (
    Experiment,
    RunDescription,
)

YAML_DIR = "/home/pavel/nethack_curiosity/runs"
ID_PATH = "/home/pavel/nethack_curiosity/ID"

with open(ID_PATH, "r") as f:
    MACHINE_ID = int(f.read().strip())

yaml_files = [f for f in os.listdir(YAML_DIR) if f.endswith(".yaml")]
target_prefix = f"{MACHINE_ID:02d}_"  # Create a two-digit prefix
yaml_file = next((f for f in yaml_files if f.startswith(target_prefix)), None)

if yaml_file is None:
    raise ValueError(f"No YAML file found for Machine ID {MACHINE_ID}")

with open(os.path.join(YAML_DIR, yaml_file), "r") as f:
    setup: Dict = yaml.safe_load(f)

ir_module: str = setup["intrinsic_reward_module"]
env: str = setup["env"]

seeds = [40, 41, 42]

params: List[Dict] = []
for seed in seeds:
    param = setup.copy()
    param["seed"] = seed
    params.append(param)

cmd = "python -m nethack_curiosity.experiments.run_nethack"

experiment = Experiment(name=ir_module, cmd=cmd, param_generator=params)

description = RunDescription(run_name=env, experiments=[experiment])

RUN_DESCRIPTION = description
