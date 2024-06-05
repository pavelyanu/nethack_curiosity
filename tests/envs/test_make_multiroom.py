import pytest

from nethack_curiosity.envs.minigrid.make_multiroom import make_multiroom

def test_make_multiroom():
    for i in range(1, 100):
        for j in range(4, 100):
            make_multiroom(i, j)

    # invalid room size
    for i in range(0, 4):
        with pytest.raises(AssertionError):
            make_multiroom(1, i)
