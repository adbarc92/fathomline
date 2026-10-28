"""`Env` and `Policy` over LeRobot 0.6.1: SmolVLA on LIBERO `libero_object` task 0 (D1).

D3: fathomline owns the step loop. This module builds the same objects
`lerobot-eval` builds (`make_env`, `make_policy`, both processor pairs, the
`rename_map`) and exposes one step at a time; `lerobot_eval.rollout` is never
called. The split follows what each half owns: the environment's processors
run in `LeRobotEnv`, the policy's in `LeRobotPolicy`, so an observation crossing
the seam is the LeRobot-format batch any LeRobot policy reads, and an action is
the plain float tuple the core hashes.

Linux only, and only in the container: it needs `--extra lerobot` and the
snapshots `warm.py` places at build time. It is omitted from the unit job's
coverage gate (D11) and exercised by the CI `container` job instead.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from typing import Any

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs import preprocess_observation as _preprocess_observation
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

from fathomline.adapters.lerobot import pins
from fathomline.core.types import Action, ArtifactRef, Observation, Transition

__all__ = ["LeRobotEnv", "LeRobotPolicy", "load_fixture"]


class LeRobotEnv:
    """One LIBERO task behind a one-wide `SyncVectorEnv`, as `lerobot-eval` builds it.

    The episode's initial state is a function of the seed alone, `seed % n`, over
    the task's `n` fixed benchmark init states. LeRobot instead advances a
    counter on every reset, which would make an episode's outcome depend on how
    many episodes ran before it in the same process. With the default seed set
    (1000, 1001, ...) and `libero_object`'s 50 init states, the two agree.
    """

    def __init__(
        self,
        vec_env: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        task_description: str,
        ref: ArtifactRef,
    ) -> None:
        self._vec_env = vec_env
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._task_description = task_description
        self._ref = ref

    @property
    def ref(self) -> ArtifactRef:
        return self._ref

    @property
    def max_steps(self) -> int:
        """LeRobot's per-suite episode limit (280 for `libero_object`)."""
        return int(self._vec_env.call("_max_episode_steps")[0])

    def reset(self, seed: int) -> Observation:
        sub_env = self._vec_env.envs[0]
        init_states = sub_env._init_states
        if init_states is not None:
            sub_env.init_state_id = seed % len(init_states)
        raw, _info = self._vec_env.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
        return self._observe(raw)

    def step(self, action: Action) -> Transition:
        batch = {ACTION: torch.tensor([action], dtype=torch.float32)}
        sent = self._postprocessor(batch)[ACTION].to("cpu").numpy()
        raw, reward, terminated, truncated, info = self._vec_env.step(sent)
        return Transition(
            observation=self._observe(raw),
            reward=float(reward[0]),
            done=bool(terminated[0] or truncated[0]),
            success=_is_success(info),
        )

    def close(self) -> None:
        self._vec_env.close()

    def _observe(self, raw: Any) -> Observation:
        observation = _preprocess_observation(raw)
        observation["task"] = [self._task_description]
        processed: dict[str, object] = self._preprocessor(observation)
        return processed


def _is_success(info: Mapping[str, Any]) -> bool:
    """Read success the way `lerobot_eval.rollout` does, for a one-wide vector env."""
    final_info = info.get("final_info")
    if isinstance(final_info, Mapping):
        return bool(final_info.get("is_success", [False])[0])
    if "is_success" in info:
        return bool(info["is_success"][0])
    return False


class LeRobotPolicy:
    """A pretrained LeRobot policy with its own pre- and post-processors.

    `reset(seed)` reseeds Python, NumPy and torch, so the flow-matching noise an
    episode samples depends on its seed and not on what ran before it.
    """

    def __init__(
        self, policy: Any, preprocessor: Any, postprocessor: Any, *, ref: ArtifactRef
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._ref = ref

    @property
    def ref(self) -> ArtifactRef:
        return self._ref

    def reset(self, seed: int) -> None:
        set_seed(seed)
        self._policy.reset()

    def act(self, observation: Observation) -> Action:
        # A shallow copy: the observation is also the one monitors are handed.
        batch = self._preprocessor(dict(observation))
        with torch.inference_mode():
            action = self._policy.select_action(batch)
        values = self._postprocessor(action).to("cpu").numpy()
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError(f"expected one action of shape (1, dim), got {values.shape}")
        return tuple(float(value) for value in values[0])


def _installed(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def load_fixture() -> tuple[LeRobotEnv, LeRobotPolicy, int]:
    """Build the D1 fixture from the pinned, digest-checked snapshots in the image.

    Returns the env, the policy and the episode step limit. Runs offline: every
    path it reads is local, and the image sets `HF_HUB_OFFLINE=1`, so anything
    unpinned that tries the network fails instead of drifting.
    """
    lerobot_version = _installed("lerobot")
    if lerobot_version != pins.LEROBOT_VERSION:
        raise RuntimeError(
            f"lerobot {lerobot_version} is installed; the fixture pins {pins.LEROBOT_VERSION}"
        )

    dirs = pins.artifact_dirs()
    for pin, directory in dirs.items():
        pins.verify(pin, directory)
    policy_dir, vlm_dir = dirs[pins.POLICY], dirs[pins.VLM]

    env_cfg = make_env_config("libero", task=pins.SUITE, task_ids=[pins.TASK_ID])
    policy_cfg = PreTrainedConfig.from_pretrained(
        policy_dir,
        cli_overrides=[
            "--device=cpu",
            "--use_amp=false",
            # The pinned local snapshot instead of the repo id the checkpoint names.
            f"--vlm_model_name={vlm_dir}",
            f"--n_action_steps={pins.N_ACTION_STEPS}",
        ],
    )
    policy_cfg.pretrained_path = policy_dir

    vec_env = make_env(env_cfg, n_envs=1, use_async_envs=False)[pins.SUITE][pins.TASK_ID]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=pins.RENAME_MAP)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={
            # The checkpoint ships `device: cuda`.
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": pins.RENAME_MAP},
            "tokenizer_processor": {"tokenizer_name": str(vlm_dir)},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    env = LeRobotEnv(
        vec_env,
        env_preprocessor,
        env_postprocessor,
        task_description=str(vec_env.call("task_description")[0]),
        ref=ArtifactRef(
            name=f"libero/{pins.SUITE}/{pins.TASK_ID}",
            revision=(
                f"lerobot=={lerobot_version} hf-libero=={_installed('hf-libero')} "
                f"{pins.LIBERO_ASSETS.repo_id}@{pins.LIBERO_ASSETS.revision}"
            ),
        ),
    )
    agent = LeRobotPolicy(
        policy,
        preprocessor,
        postprocessor,
        ref=ArtifactRef(
            name=pins.POLICY.repo_id,
            revision=f"{pins.POLICY.revision} {pins.VLM.repo_id}@{pins.VLM.revision}",
        ),
    )
    return env, agent, env.max_steps
