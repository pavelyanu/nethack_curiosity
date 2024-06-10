from nethack_curiosity.envs.minigrid.wrappers.minigrid_type import MinigridTypeWrapper
from nethack_curiosity.envs.minigrid.wrappers.minigrid_visit_count import (
    MinigridVisitCountWrapper,
)

__required__ = [MinigridTypeWrapper]

__global_order__ = [MinigridTypeWrapper, MinigridVisitCountWrapper]
