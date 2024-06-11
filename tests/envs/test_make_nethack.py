import pytest
from nethack_curiosity.envs.nethack.make_nethack import _make_nethack


def test_make_nethack():
    # Test with default parameters
    env = _make_nethack("NetHackScore")
    assert env is not None
    assert env.penalty_mode == "constant"
    assert env.penalty_step == -0.01
    assert env.penalty_time == 0.0

    # Test with custom parameters
    env = _make_nethack(
        "NetHackScore",
        penalty_mode="exp",
        penalty_step=-0.02,
        penalty_time=0.1,
        add_required_wrappers=False,
    )
    assert env.penalty_mode == "exp"
    assert env.penalty_step == -0.02
    assert env.penalty_time == 0.1

    # Test with invalid task name
    with pytest.raises(ValueError):
        _make_nethack("InvalidTaskName")


class TestMakeNetHack:
    def test_name_parameter(self):
        score_names = ["score", "nethack_score", "NetHackScore"]
        staircase_pet_names = [
            "staircase_pet",
            "nethack_staircase_pet",
            "NetHackStaircasePet",
        ]
        staircase_names = ["staircase", "nethack_staircase", "NetHackStaircase"]
        oracle_names = ["oracle", "nethack_oracle", "NetHackOracle"]
        gold_names = ["gold", "nethack_gold", "NetHackGold"]
        eat_names = ["eat", "nethack_eat", "NetHackEat"]
        scout_names = ["scout", "nethack_scout", "NetHackScout"]
        challenge_names = ["challenge", "nethack_challenge", "NetHackChallenge"]

        for name in score_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackScore"

        for name in staircase_pet_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackStaircasePet"

        for name in staircase_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackStaircase"

        for name in oracle_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackOracle"

        for name in gold_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackGold"

        for name in eat_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackEat"

        for name in scout_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackScout"

        for name in challenge_names:
            env = _make_nethack(name)
            assert env is not None
            assert env.__class__.__name__ == "NetHackChallenge"

    def test_invalid_name_parameter(self):
        with pytest.raises(ValueError):
            _make_nethack("InvalidTaskName")
