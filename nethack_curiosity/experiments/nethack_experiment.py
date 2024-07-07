from sample_factory.launcher.run_description import (
    Experiment,
    RunDescription,
)

from nethack_curiosity.experiments.setups.nethack.setup2 import get_setups


_params = get_setups()

print("Total number of experiments:", len(list(_params)))
print("Params:")
for p in _params:
    print(p)

_experiment = Experiment(
    "rnd_irw_grid",
    "python -m nethack_curiosity.experiments.run_nethack --env score",
    _params,
)

description = RunDescription("score", [_experiment])

RUN_DESCRIPTION = description
