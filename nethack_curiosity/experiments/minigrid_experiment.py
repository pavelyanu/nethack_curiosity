from sample_factory.launcher.run_description import (
    Experiment,
    ParamGrid,
    RunDescription,
    ParamList,
)

from nethack_curiosity.experiments.setups.setup5 import get_setups


_params = get_setups()

print("Total number of experiments:", len(list(_params)))
print("Params:")
for p in _params:
    print(p)

_experiment = Experiment(
    "rnd_vs_vanilla_vs_noveld",
    "python -m nethack_curiosity.experiments.run_minigrid --env keycorridor --minigrid_room_size 5 --minigrid_row_num 3 --train_for_env_steps 30000000",
    _params,
)

description = RunDescription("kcS5R3", [_experiment])

RUN_DESCRIPTION = description
