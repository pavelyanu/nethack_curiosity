from minigrid.envs.multiroom import MultiRoomEnv

def make_multiroom(N: int, S: int) -> MultiRoomEnv:
    return MultiRoomEnv(minNumRooms=N, maxNumRooms=N, maxRoomSize=S)
