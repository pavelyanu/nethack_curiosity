import pytest

from tests.envs.run_env import run_for_n_few_steps
from tests.envs.minigrid_test_cfg import make_minigrid_cfg
from nethack_curiosity.envs.minigrid.make_minigrid import (
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
    _make_minigrid(name)

    name = "multiroom-N2-S4-V0"
    n, s = parse_multiroom_env_name(name)
    assert n == 2
    assert s == 4
    _make_minigrid(name)

    name = "empty"
    _make_minigrid(name)

    with pytest.raises(NotImplementedError):
        _make_minigrid("unknown")


def test_with_required_wrappers():
    cfg = make_minigrid_cfg("multiroom-1n-4s")
    env = _make_minigrid("multiroom-1n-4s")
    for wrapper in __required__:
        env = wrapper(env, cfg)
    run_for_n_few_steps(env)

    cfg = make_minigrid_cfg("empty")
    env = _make_minigrid("empty")
    for wrapper in __required__:
        env = wrapper(env, cfg)
    run_for_n_few_steps(env)


def test_with_optional_wrappers():
    if len(__global_order__) == len(__required__):
        pytest.skip("No optional wrappers to test")

    # Test multiroom-1n-4s
    cfg = make_minigrid_cfg("multiroom-1n-4s")
    env = _make_minigrid("multiroom-1n-4s")

    for wrapper in __global_order__:
        if wrapper in __required__:
            env = wrapper(env, cfg)

    for wrapper in __global_order__:
        env = wrapper(env, cfg)
        run_for_n_few_steps(env)

    # Test empty
    cfg = make_minigrid_cfg("empty")
    env = _make_minigrid("empty")

    for wrapper in __global_order__:
        if wrapper in __required__:
            env = wrapper(env, cfg)

    for wrapper in __global_order__:
        env = wrapper(env, cfg)
        run_for_n_few_steps(env)
