from sample_factory.launcher.run_description import (
    Experiment,
    RunDescription,
)

from nethack_curiosity.experiments.setups.minigrid.setup9 import get_setups


_params = get_setups()

print("Total number of experiments:", len(list(_params)))
print("Params:")
for p in _params:
    print(p)

_experiment = Experiment(
    "easy_rnd_vs_mock",
    "python -m nethack_curiosity.experiments.run_minigrid --env keycorridor --minigrid_room_size 3 --minigrid_row_num 3",
    _params,
)

description = RunDescription("keycorridor", [_experiment])

RUN_DESCRIPTION = description
