from sample_factory.launcher.run_description import (
    Experiment,
    ParamGrid,
    RunDescription,
    ParamList,
)

from nethack_curiosity.experiments.setups.setup7 import get_setups


_params = get_setups()

print("Total number of experiments:", len(list(_params)))
print("Params:")
for p in _params:
    print(p)

_experiment = Experiment(
    "nd_vs_ndc_vs_rnd_vs_mock",
    "python -m nethack_curiosity.experiments.run_minigrid --env obstructedmaze_2dlh --train_for_env_steps 30000000",
    _params,
)

description = RunDescription("maze", [_experiment])

RUN_DESCRIPTION = description
