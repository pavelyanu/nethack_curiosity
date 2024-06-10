import pytest

from tests.envs.run_env import run_for_n_few_steps
from nethack_curiosity.envs.minigrid.make_minigrid import (
    make_minigrid,
    _make_minigrid,
    make_multiroom,
    make_empty,
    parse_multiroom_env_name,
    __required__,
    __global_order__,
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
    _make_minigrid(name, False)

    name = "multiroom-N2-S4-V0"
    n, s = parse_multiroom_env_name(name)
    assert n == 2
    assert s == 4
    _make_minigrid(name, False)

    name = "empty"
    _make_minigrid(name, False)

    with pytest.raises(NotImplementedError):
        _make_minigrid("unknown", False)


def test_with_required_wrappers():
    env = _make_minigrid("multiroom-1n-4s", False)
    for wrapper in __required__:
        env = wrapper(env)
    run_for_n_few_steps(env)

    env = _make_minigrid("empty", False)
    for wrapper in __required__:
        env = wrapper(env)
    run_for_n_few_steps(env)


def test_with_optional_wrappers():
    if len(__global_order__) == len(__required__):
        pytest.skip("No optional wrappers to test")

    env = _make_minigrid("multiroom-1n-4s", False)
    for wrapper in __global_order__:
        env = wrapper(env)
        run_for_n_few_steps(env)

    env = _make_minigrid("empty", False)
    for wrapper in __global_order__:
        env = wrapper(env)
        run_for_n_few_steps(env)
