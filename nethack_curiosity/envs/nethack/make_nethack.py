from typing import Optional

import gymnasium as gym

from nethack_curiosity.envs.nethack.wrappers.nethack_dlvl_wrapper import (
    NethackDlvlWrapper,
)
from nethack_curiosity.envs.nethack.wrappers.oracle_reward_wrapper import (
    OracleRewardWrapper,
)
from nle.env.tasks import (
    NetHackScore,
    NetHackStaircase,
    NetHackStaircasePet,
    NetHackOracle,
    NetHackGold,
    NetHackEat,
    NetHackScout,
    NetHackChallenge,
)
from nle.env.base import NLE

from nethack_curiosity.envs.nethack.wrappers.nethack_visit_count import (
    NethackVisitCountWrapper,
)
from sample_factory.utils.typing import Config

from nethack_curiosity.envs.nethack.wrappers.sf_wrappes import (
    BlstatsInfoWrapper,
    NLETimeLimit,
    PrevActionsWrapper,
    RenderCharImagesWithNumpyWrapperV2,
    SeedActionSpaceWrapper,
    TaskRewardsInfoWrapper,
)


# make liberal patterns for matching task names to classes
# for example I want 'score', 'nethack_score', 'NetHackScore' to all match to NetHackScore
def match_name_to_class(name: str) -> type:
    if "score" in name.lower():
        return NetHackScore
    if "staircase" in name.lower() and "pet" in name.lower():
        return NetHackStaircasePet
    if "staircase" in name.lower():
        return NetHackStaircase
    if "oracle" in name.lower():
        return NetHackOracle
    if "gold" in name.lower():
        return NetHackGold
    if "eat" in name.lower():
        return NetHackEat
    if "scout" in name.lower():
        return NetHackScout
    if "challenge" in name.lower():
        return NetHackChallenge
    else:
        raise ValueError(f"Task {name} not found in available nethack tasks.")


def make_nethack(
    full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None
) -> NLE:
    assert cfg is not None, "cfg must be provided to make_nethack."
    assert render_mode in [None, "human", "ansi", "full"]
    env = _make_nethack(
        name=full_env_name,
        save_ttyrec_every=cfg.save_ttyrec_every,
        savedir=cfg.savedir,
        character=cfg.character,
        max_episode_steps=cfg.max_episode_steps,
        observation_keys=cfg.observation_keys,
        actions=cfg.actions,
        options=cfg.options,
        wizard=cfg.wizard,
        allow_all_yn_questions=cfg.allow_all_yn_questions,
        allow_all_modes=cfg.allow_all_modes,
        spawn_monsters=cfg.spawn_monsters,
        render_mode=render_mode,
        penalty_mode=cfg.penalty_mode,
        penalty_step=cfg.penalty_step,
        penalty_time=cfg.penalty_time,
    )

    if cfg.add_image_observation:
        env = RenderCharImagesWithNumpyWrapperV2(
            env,
            crop_size=cfg.crop_dim,
            rescale_font_size=(cfg.pixel_size, cfg.pixel_size),
        )
    if cfg.use_prev_action:
        env = PrevActionsWrapper(env)

    if cfg.add_stats_to_info:
        env = BlstatsInfoWrapper(env)
        env = TaskRewardsInfoWrapper(env)

    if (
        cfg.intrinsic_reward_module == "noveld"
        or cfg.intrinsic_reward_module == "inverse"
    ):
        env = NethackVisitCountWrapper(env, cfg)

    env = NethackDlvlWrapper(env, cfg)

    if "oracle" in full_env_name:
        env = OracleRewardWrapper(env, cfg)

    return env


def _make_nethack(
    name: str,
    # below are for base
    save_ttyrec_every=0,
    savedir=None,
    character: str = "mon-hum-neu-mal",
    max_episode_steps: int = 5000,
    observation_keys: list = (
        "glyphs",
        "chars",
        "colors",
        "specials",
        "blstats",
        "message",
        "inv_glyphs",
        "inv_strs",
        "inv_letters",
        "inv_oclasses",
        "screen_descriptions",
        "tty_chars",
        "tty_colors",
        "tty_cursor",
    ),
    actions: list = None,
    options: list = None,
    wizard: bool = False,
    allow_all_yn_questions: bool = False,
    allow_all_modes: bool = False,
    spawn_monsters: bool = True,
    render_mode: str = "human",
    # below are for tasks
    penalty_mode: str = "constant",
    penalty_step: float = -0.01,
    penalty_time: float = -0.0,
) -> NLE:
    """Constructs a new NetHack environment.

    Args:
        name (str): name of the task. One of:
            "score", "staircase", "staircase_pet", "oracle", "gold", "eat",
            "scout", "challenge".
        save_ttyrec_every: Integer, if 0, no ttyrecs (game recordings) will
            be saved. Otherwise, save a ttyrec every Nth episode.
        savedir (str or None): Path to save ttyrecs (game recordings) into,
            if save_ttyrec_every is nonzero. If nonempty string, interpreted
            as a path to a new or existing directory.
            If "" (empty string) or None, NLE choses a unique directory name.
        character (str): name of character. Defaults to "mon-hum-neu-mal".
        max_episode_steps (int): maximum amount of steps allowed before the
            game is forcefully quit. In such cases, ``info["end_status"]``
            will be equal to ``StepStatus.ABORTED``. Defaults to 5000.
        observation_keys (list): keys to use when creating the observation.
            Defaults to all.
        actions (list): list of actions. If None, the full action space will
            be used, i.e. ``nle.nethack.ACTIONS``. Defaults to None.
        options (list): list of game options to initialize Nethack. If None,
            Nethack will be initialized with the options found in
            ``nle.nethack.NETHACKOPTIONS`. Defaults to None.
        wizard (bool): activate wizard mode. Defaults to False.
        allow_all_yn_questions (bool):
            If set to True, no y/n questions in step() are declined.
            If set to False, only elements of SKIP_EXCEPTIONS are not declined.
            Defaults to False.
        allow_all_modes (bool):
            If set to True, do not decline menus, text input or auto 'MORE'.
            If set to False, only skip click through 'MORE' on death.
        spawn_monsters: If False, disables normal NetHack behavior to randomly
            create monsters.
        render_mode (str): mode used to render the screen. One of
            "human" | "ansi" | "full".
            Defaults to "human", i.e. what a human would see playing the game.
        penalty_mode (str): name of the mode for calculating the time step
            penalty. Can be ``constant``, ``exp``, ``square``, ``linear``, or
            ``always``. Defaults to ``constant``.
        penalty_step (float): constant applied to amount of frozen steps.
            Defaults to -0.01.
        penalty_time (float): constant applied to amount of frozen steps.
            Defaults to -0.0.
    """
    task_class = match_name_to_class(name)
    env: NLE

    if task_class == NetHackChallenge:
        env = task_class(
            save_ttyrec_every=save_ttyrec_every,
            savedir=savedir,
            character=character,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
            options=options,
            wizard=wizard,
            allow_all_yn_questions=allow_all_yn_questions,
            allow_all_modes=allow_all_modes,
            spawn_monsters=spawn_monsters,
            render_mode=render_mode,
            penalty_mode=penalty_mode,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
        )
    else:
        env = task_class(
            save_ttyrec_every=save_ttyrec_every,
            savedir=savedir,
            character=character,
            max_episode_steps=max_episode_steps,
            observation_keys=observation_keys,
            actions=actions,
            options=options,
            wizard=wizard,
            allow_all_yn_questions=allow_all_yn_questions,
            allow_all_modes=allow_all_modes,
            spawn_monsters=spawn_monsters,
            render_mode=render_mode,
            penalty_mode=penalty_mode,
            penalty_step=penalty_step,
            penalty_time=penalty_time,
        )

    return env


# def apply_required_wrappers(env: NLE, cfg: Config) -> NLE:
#     for wrapper in __global_order__:
#         if wrapper in __required__:
#             env = wrapper(env, cfg)
#     return env
