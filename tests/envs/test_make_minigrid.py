import pytest

from nethack_curiosity.envs.minigrid.make_minigrid import (
    make_minigrid,
    make_multiroom,
    make_empty,
    parse_multiroom_env_name,
)


def test_make_multiroom():
    for i in range(1, 100):
        for j in range(4, 100):
            make_multiroom(i, j)

    # invalid room size
    for i in range(0, 4):
        with pytest.raises(AssertionError):
            make_multiroom(1, i)


def test_make_empty():
    make_empty()


def test_make_minigrid():
    name = "multiroom-1n-4s"
    n, s = parse_multiroom_env_name(name)
    assert n == 1
    assert s == 4
    make_minigrid(name)

    name = "multiroom-N2-S4-V0"
    n, s = parse_multiroom_env_name(name)
    assert n == 2
    assert s == 4
    make_minigrid(name)

    name = "empty"
    make_minigrid(name)

    with pytest.raises(NotImplementedError):
        make_minigrid("unknown")
