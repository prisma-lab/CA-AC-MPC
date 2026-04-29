from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib
import numpy as np
import torch as th

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "mpc.pytorch"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "diff_mpc_drones"))
sys.path.insert(0, str(ROOT / "differentialMPCPerformance"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "training_modules"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "stable-baselines3-acmpc-acmpc"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecNormalize

from envs.gate_racing_env import GateRacingEnv
from mlp_mpc_policy import MlpMpcPolicy
from mlp_mpc_policy_diffmpc import MlpMpcPolicyDiffMPC
from mlp_only_policy import MlpOnlyPolicy
from utils.evaluate_acmpc2 import evaluate_model


class ResetWithRawStateWrapper(VecEnvWrapper):
    """
    The SB3-ACMPC fork bundled with this repo expects `env.reset()` to return `(obs, state)`.
    We return the (possibly normalized) observation plus a raw/un-normalized state slice
    for the MPC block.
    """

    def __init__(self, venv, state_dim: int = 10):
        super().__init__(venv)
        self.state_dim = int(state_dim)

    def reset(self):
        obs = self.venv.reset()

        raw_obs = obs
        if hasattr(self.venv, "get_original_obs"):
            try:
                raw_obs = self.venv.get_original_obs()
            except Exception:
                raw_obs = obs

        if isinstance(raw_obs, np.ndarray):
            state = raw_obs[:, : self.state_dim]
        else:
            state = raw_obs
        return obs, state

    def step_wait(self):
        return self.venv.step_wait()


def _run_input_sanity_checks(
    *,
    train_env,
    mpc_state_dim: int,
    normalize_obs: bool,
    n_envs: int,
    n_steps: int,
    batch_size: int,
) -> None:
    rollout_batch = int(n_envs) * int(n_steps)
    if rollout_batch <= 0:
        raise ValueError("Invalid PPO rollout batch size: n_envs * n_steps must be > 0.")
    if int(batch_size) > rollout_batch:
        raise ValueError(
            f"Invalid PPO setup: batch_size ({batch_size}) > n_envs*n_steps ({rollout_batch})."
        )
    if rollout_batch % int(batch_size) != 0:
        print(
            "[Sanity] Warning: n_envs*n_steps is not divisible by batch_size. "
            f"rollout_batch={rollout_batch}, batch_size={batch_size}."
        )

    try:
        obs, state = train_env.reset()
        raw_obs = obs
        if hasattr(train_env, "get_original_obs"):
            raw_obs = train_env.get_original_obs()

        if isinstance(raw_obs, np.ndarray) and isinstance(state, np.ndarray):
            raw_state = raw_obs[:, : int(mpc_state_dim)]
            state_delta = float(np.max(np.abs(raw_state - state)))
            print(f"[Sanity] raw_state vs state_delta_max={state_delta:.3e}")
            if normalize_obs and isinstance(obs, np.ndarray):
                obs_delta_mean = float(np.mean(np.abs(obs - raw_obs)))
                print(f"[Sanity] normalized_obs_delta_mean={obs_delta_mean:.3e}")
    except Exception as exc:
        print(f"[Sanity] Warning: normalization/state checks skipped: {exc}")


def _track_tag(track: str, track_config_path: Optional[str]) -> str:
    raw = Path(track_config_path).stem if track_config_path is not None else str(track).strip().lower()
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._-").lower()
    return tag if tag else "track"


def _append_track_dir(path_raw: str, tag: str) -> str:
    p = Path(path_raw)
    if any(part == tag for part in p.parts):
        return str(p)
    return str(p / tag)


def _append_track_to_save_path(path_raw: str, tag: str) -> str:
    p = Path(path_raw)
    parent = p.parent
    if str(parent) == ".":
        stem = p.stem
        if tag in stem.lower():
            return str(p)
        return str(p.with_name(f"{stem}_{tag}{p.suffix}"))
    parent2 = Path(_append_track_dir(str(parent), tag))
    return str(parent2 / p.name)


def _build_training_signature(
    *,
    policy_type: str,
    mpc_backend: str,
    mpc_horizon: int,
    track: str,
    track_config_path: Optional[str],
) -> Dict[str, Any]:
    if track_config_path is not None:
        track_key = f"yaml:{Path(track_config_path).expanduser().resolve()}"
    else:
        track_key = f"builtin:{str(track).strip().lower()}"
    return {
        "version": 1,
        "policy_type": str(policy_type),
        "mpc_backend": str(mpc_backend),
        "mpc_horizon": int(mpc_horizon),
        "track_key": track_key,
    }


def _meta_path_for_checkpoint(ckpt_path: Union[str, Path]) -> Path:
    ckpt = Path(ckpt_path)
    return Path(str(ckpt) + ".meta.json")


def _load_checkpoint_metadata(ckpt_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    meta_path = _meta_path_for_checkpoint(ckpt_path)
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _check_signature_compatibility(
    expected: Dict[str, Any],
    observed: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if observed is None:
        return False, "missing signature metadata"
    keys = ("policy_type", "mpc_backend", "mpc_horizon", "track_key")
    for key in keys:
        if observed.get(key) != expected.get(key):
            return False, f"mismatch {key}: expected={expected.get(key)!r} observed={observed.get(key)!r}"
    return True, "ok"


def _build_learning_rate(ppo_cfg: Dict[str, Any]) -> Union[float, Callable[[float], float]]:
    """
    Accept either:
      - scalar: learning_rate: 3e-4
      - mapping:
          learning_rate:
            schedule: linear|cosine|poly|exp|constant
            initial: 3e-4
            final: 3e-5        # optional
            final_factor: 0.1  # optional alternative to final
            power: 1.0         # for poly
            decay_start: 0.0   # optional, fraction of training elapsed before decay starts
    """
    lr_raw = ppo_cfg.get("learning_rate", 3e-4)
    if isinstance(lr_raw, (int, float)):
        lr = float(lr_raw)
        print(f"[LR] schedule=constant initial={lr:.6g}")
        return lr

    if not isinstance(lr_raw, dict):
        raise ValueError("ppo.learning_rate must be a float or a dict schedule config.")

    schedule = str(lr_raw.get("schedule", lr_raw.get("type", "constant"))).strip().lower()
    initial = float(lr_raw.get("initial", lr_raw.get("value", 3e-4)))
    final_cfg = lr_raw.get("final", None)
    final_factor_cfg = lr_raw.get("final_factor", None)
    power = float(lr_raw.get("power", 1.0))
    decay_start = float(lr_raw.get("decay_start", 0.0))
    if not (0.0 <= decay_start < 1.0):
        raise ValueError("ppo.learning_rate.decay_start must be in [0, 1).")

    if final_cfg is None:
        if final_factor_cfg is None:
            final_factor_cfg = 1.0 if schedule in {"constant", "none"} else 0.1
        final = initial * float(final_factor_cfg)
    else:
        final = float(final_cfg)

    if schedule in {"constant", "none"}:
        print(f"[LR] schedule=constant initial={initial:.6g}")
        return initial

    def _effective_progress(progress_remaining: float) -> float:
        """Map SB3 progress_remaining to an adjusted value that delays LR decay."""
        p = float(np.clip(progress_remaining, 0.0, 1.0))
        if decay_start <= 0.0:
            return p
        elapsed = 1.0 - p
        if elapsed <= decay_start:
            return 1.0
        elapsed_after_start = (elapsed - decay_start) / max(1e-12, (1.0 - decay_start))
        return float(np.clip(1.0 - elapsed_after_start, 0.0, 1.0))

    if schedule == "linear":
        print(
            f"[LR] schedule=linear initial={initial:.6g} final={final:.6g} "
            f"decay_start={decay_start:.3g}"
        )

        def _lr_fn(progress_remaining: float) -> float:
            p = _effective_progress(progress_remaining)
            return float(final + (initial - final) * p)

        return _lr_fn

    if schedule in {"cos", "cosine"}:
        print(
            f"[LR] schedule=cosine initial={initial:.6g} final={final:.6g} "
            f"decay_start={decay_start:.3g}"
        )

        def _lr_fn(progress_remaining: float) -> float:
            p_eff = _effective_progress(progress_remaining)
            elapsed_eff = 1.0 - p_eff
            return float(final + 0.5 * (initial - final) * (1.0 + np.cos(np.pi * elapsed_eff)))

        return _lr_fn

    if schedule in {"poly", "polynomial"}:
        print(
            f"[LR] schedule=poly initial={initial:.6g} final={final:.6g} power={power:.3g} "
            f"decay_start={decay_start:.3g}"
        )

        def _lr_fn(progress_remaining: float) -> float:
            p = _effective_progress(progress_remaining)
            return float(final + (initial - final) * (p**power))

        return _lr_fn

    if schedule in {"exp", "exponential"}:
        if initial <= 0.0 or final <= 0.0:
            raise ValueError("Exponential LR schedule requires positive initial and final values.")
        ratio = final / initial
        print(
            f"[LR] schedule=exp initial={initial:.6g} final={final:.6g} "
            f"decay_start={decay_start:.3g}"
        )

        def _lr_fn(progress_remaining: float) -> float:
            p_eff = _effective_progress(progress_remaining)
            elapsed_eff = 1.0 - p_eff
            return float(initial * (ratio**elapsed_eff))

        return _lr_fn

    raise ValueError(
        f"Unsupported ppo.learning_rate.schedule={schedule}. "
        "Use one of: constant, linear, cosine, poly, exp."
    )


def _report_policy_init_stats(model: PPO, policy_type: str, resumed: bool) -> None:
    tag = "resume" if resumed else "fresh"
    print(f"[Init] policy_type={policy_type} mode={tag}")
    try:
        policy = model.policy
        action_net = getattr(policy, "action_net", None)
        if action_net is not None:
            print(f"[Init] action_head={action_net.__class__.__name__}")

        log_std = getattr(policy, "log_std", None)
        if isinstance(log_std, th.Tensor):
            ls = log_std.detach().cpu().float()
            print(
                "[Init] log_std "
                f"mean={float(ls.mean()):.4f} min={float(ls.min()):.4f} max={float(ls.max()):.4f}"
            )

        extractor = getattr(policy, "mlp_extractor", None)
        if extractor is None:
            return

        def _linear_stats(net_name: str, net_obj: Any) -> None:
            linears = [m for m in net_obj.modules() if isinstance(m, th.nn.Linear)]
            if not linears:
                return
            first = linears[0].weight.detach().cpu().float()
            last = linears[-1].weight.detach().cpu().float()
            print(
                f"[Init] {net_name} first_std={float(first.std()):.4e} last_std={float(last.std()):.4e}"
            )

        if hasattr(extractor, "policy_net"):
            _linear_stats("policy_net", extractor.policy_net)
        if hasattr(extractor, "value_net"):
            _linear_stats("value_net", extractor.value_net)
    except Exception as exc:
        print(f"[Init] Warning: could not collect initialization stats: {exc}")


def _find_vecnormalize(venv) -> Optional[VecNormalize]:
    cur = venv
    for _ in range(16):
        if isinstance(cur, VecNormalize):
            return cur
        if hasattr(cur, "venv"):
            cur = cur.venv
        else:
            break
    return None


def _make_gate_env_fn(track: str, env_kwargs: Dict[str, Any]):
    def _thunk():
        return GateRacingEnv(track=track, **env_kwargs)

    return _thunk


def _make_vec_env_from_cfg(
    *,
    track: str,
    env_kwargs: Dict[str, Any],
    seed: int,
    n_envs: int,
    vec_type: str,
    normalize_obs: bool,
    clip_obs: float,
    log_dir: str,
    state_dim: int,
    train_vecnorm: Optional[VecNormalize] = None,
    vecnorm_stats_path: Optional[str] = None,
):
    vec_env_cls = DummyVecEnv if vec_type == "dummy" else SubprocVecEnv
    venv = make_vec_env(
        _make_gate_env_fn(track, env_kwargs),
        n_envs=int(n_envs),
        seed=int(seed),
        vec_env_cls=vec_env_cls,
    )

    if normalize_obs:
        if train_vecnorm is not None:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=float(clip_obs))
            venv.obs_rms = train_vecnorm.obs_rms
            venv.training = False
            venv.norm_reward = False
        elif vecnorm_stats_path:
            venv = VecNormalize.load(vecnorm_stats_path, venv)
            venv.training = True
            venv.norm_reward = False
        else:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=float(clip_obs))

    venv = ResetWithRawStateWrapper(venv, state_dim=state_dim)
    return venv


def evaluate_policy_vec(
    model: PPO,
    env,
    *,
    n_episodes: int,
    state_dim: int,
    deterministic: bool,
) -> Tuple[float, float]:
    episode_rewards = []
    n_envs = env.num_envs
    obs, _state = env.reset()
    ep_rewards = np.zeros(n_envs, dtype=np.float32)

    while len(episode_rewards) < n_episodes:
        # BaseAlgorithm.predict() in this SB3 fork does not accept `drone_state`,
        # so we call the policy directly.
        if hasattr(env, "get_original_obs"):
            raw_obs = env.get_original_obs()
        else:
            raw_obs = obs

        drone_state = raw_obs[:, :state_dim] if isinstance(raw_obs, np.ndarray) else raw_obs
        actions, _ = model.policy.predict(obs, drone_state=drone_state, deterministic=deterministic)
        obs, rewards, dones, _infos = env.step(actions)

        rewards = np.array(rewards).reshape(-1)
        dones = np.array(dones).reshape(-1)
        ep_rewards += rewards

        for i in range(n_envs):
            if dones[i]:
                episode_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0.0
                if len(episode_rewards) >= n_episodes:
                    break

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def save_with_vecnorm(model: PPO, env, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    model.save(path)
    if hasattr(env, "save"):
        env.save(path + ".vecnorm.pkl")
    if metadata is not None:
        meta_path = _meta_path_for_checkpoint(path)
        try:
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:
            print(f"[Checkpoint] Warning: failed to write metadata {meta_path}: {exc}")


class CheckpointEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        checkpoint_dir: str,
        checkpoint_freq: int,
        eval_env=None,
        eval_freq: int = 0,
        eval_episodes: int = 20,
        eval_deterministic: bool = True,
        state_dim: int = 10,
        periodic_plot_enabled: bool = False,
        periodic_plot_freq: int = 0,
        periodic_plot_log_dir: str = "./runs/latest",
        periodic_plot_policy_type: str = "acmpc_mlp",
        periodic_plot_mpc_backend: str = "auto",
        periodic_plot_env_kwargs: Optional[Dict[str, Any]] = None,
        periodic_plot_seed: int = 0,
        periodic_plot_device: str = "auto",
        periodic_plot_track_config_path: Optional[str] = None,
        periodic_plot_track_name: Optional[str] = None,
        periodic_plot_max_steps_cap: Optional[int] = None,
        training_signature: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = int(checkpoint_freq)

        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.eval_deterministic = bool(eval_deterministic)
        self.state_dim = int(state_dim)
        self.periodic_plot_enabled = bool(periodic_plot_enabled)
        self.periodic_plot_freq = int(periodic_plot_freq)
        self.periodic_plot_log_dir = str(periodic_plot_log_dir)
        self.periodic_plot_policy_type = str(periodic_plot_policy_type)
        self.periodic_plot_mpc_backend = str(periodic_plot_mpc_backend)
        self.periodic_plot_env_kwargs = dict(periodic_plot_env_kwargs or {})
        self.periodic_plot_seed = int(periodic_plot_seed)
        self.periodic_plot_device = str(periodic_plot_device)
        self.periodic_plot_track_config_path = (
            None if periodic_plot_track_config_path is None else str(periodic_plot_track_config_path)
        )
        self.periodic_plot_track_name = (
            None if periodic_plot_track_name is None else str(periodic_plot_track_name)
        )
        self.periodic_plot_max_steps_cap = (
            None if periodic_plot_max_steps_cap is None else int(periodic_plot_max_steps_cap)
        )
        self.training_signature = dict(training_signature or {})

        self.best_mean_reward = -np.inf
        self.last_eval = 0
        self.last_checkpoint = 0
        self.last_plot = 0
        self.eval_history = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_checkpoint_metadata(self, *, kind: str, num_timesteps: int) -> Dict[str, Any]:
        return {
            "kind": str(kind),
            "num_timesteps": int(num_timesteps),
            "signature": dict(self.training_signature),
        }

    def _on_step(self) -> bool:
        num_timesteps = self.num_timesteps

        if self.checkpoint_freq > 0 and (num_timesteps - self.last_checkpoint) >= self.checkpoint_freq:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{num_timesteps}.zip"
            save_with_vecnorm(
                self.model,
                self.training_env,
                str(ckpt_path),
                metadata=self._build_checkpoint_metadata(kind="checkpoint", num_timesteps=num_timesteps),
            )
            self.last_checkpoint = num_timesteps

        if (
            self.eval_env is not None
            and self.eval_freq > 0
            and (num_timesteps - self.last_eval) >= self.eval_freq
        ):
            mean_reward, std_reward = evaluate_policy_vec(
                self.model,
                self.eval_env,
                n_episodes=self.eval_episodes,
                state_dim=self.state_dim,
                deterministic=self.eval_deterministic,
            )
            if self.verbose:
                print(f"[Eval] steps={num_timesteps} mean={mean_reward:.2f} std={std_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = self.checkpoint_dir / "best_model.zip"
                save_with_vecnorm(
                    self.model,
                    self.training_env,
                    str(best_path),
                    metadata=self._build_checkpoint_metadata(kind="best", num_timesteps=num_timesteps),
                )

            self.eval_history.append(
                {
                    "timesteps": int(num_timesteps),
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                }
            )
            if self.logger is not None:
                self.logger.record("eval/mean_reward", float(mean_reward))
                self.logger.record("eval/std_reward", float(std_reward))
                self.logger.record("eval/best_mean_reward", float(self.best_mean_reward))

            self.last_eval = num_timesteps

        if (
            self.periodic_plot_enabled
            and self.periodic_plot_freq > 0
            and (num_timesteps - self.last_plot) >= self.periodic_plot_freq
        ):
            try:
                periodic_model_path = self.checkpoint_dir / "periodic_plot_latest.zip"
                save_with_vecnorm(
                    self.model,
                    self.training_env,
                    str(periodic_model_path),
                    metadata=self._build_checkpoint_metadata(kind="periodic_eval", num_timesteps=num_timesteps),
                )

                periodic_log_dir = Path(self.periodic_plot_log_dir) / f"periodic_eval_{num_timesteps}"
                periodic_log_dir.mkdir(parents=True, exist_ok=True)
                eval_backend_request = resolve_eval_backend_request(
                    self.periodic_plot_policy_type,
                    self.periodic_plot_mpc_backend,
                )
                evaluate_model(
                    model_path=str(periodic_model_path),
                    log_dir=str(periodic_log_dir),
                    track_config_path=self.periodic_plot_track_config_path,
                    device=self.periodic_plot_device,
                    policy_type=self.periodic_plot_policy_type,
                    mpc_backend=eval_backend_request,
                    single_track=self.periodic_plot_track_name,
                    env_kwargs=self.periodic_plot_env_kwargs,
                    seed=self.periodic_plot_seed,
                    max_steps_cap=self.periodic_plot_max_steps_cap,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[PeriodicPlot] failed at step={num_timesteps}: {exc}")
            self.last_plot = num_timesteps

        return True


class EpisodeCountCallback(BaseCallback):
    """
    Count completed episodes during training (across all parallel envs).
    Useful to sanity-check how many episodes the agent actually experienced.
    """

    def __init__(self, *, print_every_episodes: int = 0, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.print_every_episodes = int(print_every_episodes)
        self.episode_count = 0
        self._next_print_at = int(print_every_episodes) if int(print_every_episodes) > 0 else 0
        self.gates_passed_total = 0.0
        self.gates_completion_total = 0.0
        self.gate_stats_episode_count = 0
        self._gates_passed_window = deque(maxlen=100)
        self._gates_completion_window = deque(maxlen=100)

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        if dones is None:
            return True
        try:
            n_done = int(np.sum(dones))
        except Exception:
            n_done = int(bool(dones))

        if n_done > 0:
            self.episode_count += n_done

            infos = self.locals.get("infos", None)
            if infos is not None:
                try:
                    dones_arr = np.array(dones).reshape(-1).astype(bool)
                except Exception:
                    dones_arr = np.array([bool(dones)], dtype=bool)

                gate_sum = 0.0
                completion_sum = 0.0
                gate_count = 0
                for i, done_i in enumerate(dones_arr):
                    if not bool(done_i):
                        continue
                    info_i = infos[i] if isinstance(infos, (list, tuple)) and i < len(infos) else None
                    if not isinstance(info_i, dict):
                        continue
                    gates_passed = self._as_float(info_i.get("target_gate_idx", None))
                    if gates_passed is None:
                        continue
                    num_gates = self._as_float(info_i.get("num_gates", None))
                    gate_sum += max(gates_passed, 0.0)
                    gate_count += 1
                    if num_gates is not None and num_gates > 0.0:
                        completion = float(np.clip(gates_passed / num_gates, 0.0, 1.0))
                        completion_sum += completion
                        self._gates_completion_window.append(completion)
                    self._gates_passed_window.append(max(gates_passed, 0.0))

                if gate_count > 0:
                    self.gates_passed_total += gate_sum
                    self.gates_completion_total += completion_sum
                    self.gate_stats_episode_count += gate_count

            if self.logger is not None:
                self.logger.record("rollout/episodes", float(self.episode_count))
                if self.gate_stats_episode_count > 0:
                    self.logger.record(
                        "rollout/gates_passed_per_episode_mean",
                        float(self.gates_passed_total / self.gate_stats_episode_count),
                    )
                    if len(self._gates_passed_window) > 0:
                        self.logger.record(
                            "rollout/gates_passed_per_episode_mean_100",
                            float(np.mean(self._gates_passed_window)),
                        )
                    if len(self._gates_completion_window) > 0:
                        self.logger.record(
                            "rollout/gates_completion_ratio_mean_100",
                            float(np.mean(self._gates_completion_window)),
                        )

            if self.print_every_episodes > 0 and self.verbose:
                while self._next_print_at > 0 and self.episode_count >= self._next_print_at:
                    msg = f"[Train] episodes_completed={self.episode_count}"
                    if self.gate_stats_episode_count > 0:
                        msg += (
                            f" gates/ep_mean={self.gates_passed_total / self.gate_stats_episode_count:.2f}"
                        )
                    print(msg)
                    self._next_print_at += self.print_every_episodes

        return True


class TrainingDebugReportCallback(BaseCallback):
    """
    Collect rollout-level metrics and produce a compact post-training report:
    - CSV with per-rollout metrics
    - CSV with per-episode metrics
    - JSON summary with key aggregates
    - PNG diagnostics plot with RL learning/performance/system metrics
    """

    TRAIN_LOG_KEYS = (
        "train/learning_rate",
        "train/loss",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "train/grad_norm",
        "train/grad_norm_before_clip",
        "train/grad_var",
        "train/grad_naninf_frac",
        "train/std",
        "train/n_updates",
    )
    DEBUG_INFO_NUMERIC_KEYS = (
        "dist_to_gate",
        "progress",
        "plane_dist",
        "omega_norm",
        "thrust_mass_norm",
        "z",
    )
    TERMINAL_EVENTS = frozenset({"ground", "miss", "race_finished", "gate_passed_at_timeout", "timeout"})

    def __init__(
        self,
        *,
        log_dir: str,
        episode_callback: Optional[EpisodeCountCallback] = None,
        eval_callback: Optional[CheckpointEvalCallback] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.log_dir = Path(log_dir)
        self.episode_callback = episode_callback
        self.eval_callback = eval_callback
        self.history = []
        self._event_totals = defaultdict(int)
        self._event_rollout = defaultdict(int)
        self._terminal_event_totals = defaultdict(int)
        self._terminal_event_rollout = defaultdict(int)
        self._info_sums = defaultdict(float)
        self._info_counts = defaultdict(int)
        self._rollout_idx = 0
        self._start_time = 0.0
        self._last_time = 0.0
        self._start_timesteps = 0
        self._last_timesteps = 0
        self._start_episodes = 0
        self._last_episodes = 0
        self._env_dt_s = 0.02
        self.episode_history = []
        self._last_episode_elapsed_wall_s_by_env = {}

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _reset_rollout_accumulators(self) -> None:
        self._event_rollout = defaultdict(int)
        self._terminal_event_rollout = defaultdict(int)
        self._info_sums = defaultdict(float)
        self._info_counts = defaultdict(int)

    def _extract_policy_mpc_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            policy = getattr(self.model, "policy", None)
            extractor = getattr(policy, "mlp_extractor", None)
            metrics = getattr(extractor, "last_mpc_metrics", None)
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    fv = self._as_float(v)
                    if fv is not None:
                        out[f"mpc/{k}"] = fv
            grad_mode = getattr(extractor, "mpc_grad_mode", None)
            if isinstance(grad_mode, str):
                out["mpc/grad_mode_diff"] = float(1.0 if grad_mode == "diff" else 0.0)
                out["mpc/grad_mode_stop"] = float(1.0 if grad_mode == "stop" else 0.0)
                out["mpc/grad_mode_unroll"] = float(1.0 if grad_mode == "unroll" else 0.0)
        except Exception:
            pass
        return out

    def _extract_actor_grad_stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            policy = getattr(self.model, "policy", None)
            action_net = getattr(policy, "action_net", None)
            if action_net is None:
                return out
            grads = []
            for p in action_net.parameters():
                if p.grad is not None:
                    g = p.grad.detach().flatten()
                    if g.numel() > 0:
                        grads.append(g)
            if not grads:
                return out
            g_all = th.cat(grads, dim=0)
            out["debug/actor_grad_norm"] = float(th.norm(g_all).cpu().item())
            out["debug/actor_grad_var"] = float(th.var(g_all, unbiased=False).cpu().item())
            out["debug/actor_grad_naninf_frac"] = float((~th.isfinite(g_all)).float().mean().cpu().item())
        except Exception:
            pass
        return out

    def _infer_env_dt(self) -> float:
        cur = getattr(self, "training_env", None)
        seen = set()
        for _ in range(48):
            if cur is None:
                break
            cur_id = id(cur)
            if cur_id in seen:
                break
            seen.add(cur_id)
            value = getattr(cur, "dt", None)
            value = self._as_float(value)
            if value is not None and value > 0.0:
                return float(value)
            if hasattr(cur, "envs"):
                envs = getattr(cur, "envs", [])
                cur = envs[0] if envs else None
                continue
            if hasattr(cur, "env"):
                cur = getattr(cur, "env")
                continue
            if hasattr(cur, "venv"):
                cur = getattr(cur, "venv")
                continue
            break
        return 0.02

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0:
            return values
        window = max(1, int(window))
        out = np.full(values.shape, np.nan, dtype=np.float64)
        for i in range(values.size):
            lo = max(0, i - window + 1)
            chunk = values[lo : i + 1]
            finite = chunk[np.isfinite(chunk)]
            if finite.size > 0:
                out[i] = float(np.mean(finite))
        return out

    def _on_training_start(self) -> None:
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._start_timesteps = int(self.model.num_timesteps)
        self._last_timesteps = self._start_timesteps
        self._start_episodes = int(self.episode_callback.episode_count) if self.episode_callback is not None else 0
        self._last_episodes = self._start_episodes
        self._env_dt_s = self._infer_env_dt()
        self._rollout_idx = 0
        self._last_episode_elapsed_wall_s_by_env = {}
        self._reset_rollout_accumulators()

    def _on_rollout_start(self) -> None:
        self._reset_rollout_accumulators()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        step_timesteps = int(self.num_timesteps)
        infos_iter = infos if isinstance(infos, (list, tuple)) else [infos]
        for env_idx, info in enumerate(infos_iter):
            if not isinstance(info, dict):
                continue
            event = info.get("event", None)
            if isinstance(event, str):
                self._event_rollout[event] += 1
                self._event_totals[event] += 1
                if event in self.TERMINAL_EVENTS:
                    self._terminal_event_rollout[event] += 1
                    self._terminal_event_totals[event] += 1

            episode_info = info.get("episode", None)
            if isinstance(episode_info, dict):
                episode_row = {
                    "time/timesteps": float(step_timesteps),
                }
                ep_reward = self._as_float(episode_info.get("r", None))
                ep_len_steps = self._as_float(episode_info.get("l", None))
                ep_elapsed_wall_s = self._as_float(episode_info.get("t", None))
                if ep_reward is not None:
                    episode_row["episode/reward"] = ep_reward
                if ep_len_steps is not None:
                    episode_row["episode/len_steps"] = ep_len_steps
                    episode_row["episode/len_sim_s"] = ep_len_steps * float(self._env_dt_s)
                if ep_elapsed_wall_s is not None:
                    episode_row["episode/elapsed_wall_s"] = ep_elapsed_wall_s
                    prev_elapsed = self._last_episode_elapsed_wall_s_by_env.get(env_idx, None)
                    if prev_elapsed is not None and ep_elapsed_wall_s >= prev_elapsed:
                        episode_row["episode/len_wall_s"] = ep_elapsed_wall_s - prev_elapsed
                    self._last_episode_elapsed_wall_s_by_env[env_idx] = ep_elapsed_wall_s
                if isinstance(event, str):
                    episode_row["episode/event"] = event
                self.episode_history.append(episode_row)

            for key in self.DEBUG_INFO_NUMERIC_KEYS:
                value = self._as_float(info.get(key, None))
                if value is None:
                    continue
                self._info_sums[key] += value
                self._info_counts[key] += 1
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_idx += 1
        now = time.perf_counter()
        elapsed = max(now - self._start_time, 1e-9)
        delta_t = max(now - self._last_time, 1e-9)
        timesteps = int(self.num_timesteps)
        train_timesteps = max(timesteps - self._start_timesteps, 0)
        delta_steps = max(timesteps - self._last_timesteps, 0)

        snapshot = {
            "time/rollout_iteration": float(self._rollout_idx),
            "time/elapsed_s": float(elapsed),
            "time/timesteps": float(timesteps),
            "time/train_timesteps": float(train_timesteps),
            "time/timesteps_per_s_avg": float(train_timesteps / elapsed),
            "time/timesteps_per_s_inst": float(delta_steps / delta_t),
            "time/iterations_per_s_avg": float(self._rollout_idx / elapsed),
        }

        if self.episode_callback is not None:
            episodes_total = int(self.episode_callback.episode_count)
            snapshot["rollout/episodes_total"] = float(episodes_total)
            snapshot["rollout/episodes_per_s_avg"] = float(max(episodes_total - self._start_episodes, 0) / elapsed)
            snapshot["rollout/episodes_per_s_inst"] = float(max(episodes_total - self._last_episodes, 0) / delta_t)
            if getattr(self.episode_callback, "gate_stats_episode_count", 0) > 0:
                snapshot["rollout/gates_passed_per_episode_mean"] = float(
                    self.episode_callback.gates_passed_total / self.episode_callback.gate_stats_episode_count
                )
                if len(self.episode_callback._gates_passed_window) > 0:
                    snapshot["rollout/gates_passed_per_episode_mean_100"] = float(
                        np.mean(self.episode_callback._gates_passed_window)
                    )
                if len(self.episode_callback._gates_completion_window) > 0:
                    snapshot["rollout/gates_completion_ratio_mean_100"] = float(
                        np.mean(self.episode_callback._gates_completion_window)
                    )
            self._last_episodes = episodes_total

        ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
        if ep_info_buffer is not None and len(ep_info_buffer) > 0:
            rew_vals = [ep.get("r") for ep in ep_info_buffer if isinstance(ep, dict) and "r" in ep]
            len_vals = [ep.get("l") for ep in ep_info_buffer if isinstance(ep, dict) and "l" in ep]
            if rew_vals:
                snapshot["rollout/ep_rew_mean_100"] = float(np.mean(rew_vals))
            if len_vals:
                snapshot["rollout/ep_len_mean_100"] = float(np.mean(len_vals))
                snapshot["rollout/ep_len_sim_s_mean_100"] = float(np.mean(len_vals) * float(self._env_dt_s))
            if rew_vals and len_vals:
                mean_len = max(float(np.mean(len_vals)), 1e-9)
                snapshot["rollout/ep_rew_per_step_mean_100"] = float(np.mean(rew_vals) / mean_len)

        model_logger = getattr(self.model, "logger", None)
        logger_values = dict(getattr(model_logger, "name_to_value", {}))
        for key in self.TRAIN_LOG_KEYS:
            value = self._as_float(logger_values.get(key, None))
            if value is not None:
                snapshot[key] = value
        snapshot.update(self._extract_policy_mpc_metrics())
        snapshot.update(self._extract_actor_grad_stats())

        for key, count in self._info_counts.items():
            if count > 0:
                snapshot[f"debug/{key}_mean"] = float(self._info_sums[key] / count)

        for event, count in self._event_rollout.items():
            snapshot[f"events/{event}"] = float(count)
        for event, count in self._event_totals.items():
            snapshot[f"events_total/{event}"] = float(count)
        for event, count in self._terminal_event_rollout.items():
            snapshot[f"events_terminal/{event}"] = float(count)
        for event, count in self._terminal_event_totals.items():
            snapshot[f"events_terminal_total/{event}"] = float(count)

        if self.eval_callback is not None and self.eval_callback.eval_history:
            last_eval = self.eval_callback.eval_history[-1]
            snapshot["eval/last_mean_reward"] = float(last_eval["mean_reward"])
            snapshot["eval/last_std_reward"] = float(last_eval["std_reward"])
            snapshot["eval/best_mean_reward"] = float(self.eval_callback.best_mean_reward)

        if th.cuda.is_available() and str(getattr(self.model, "device", "cpu")).startswith("cuda"):
            try:
                device_index = th.cuda.current_device()
                snapshot["system/cuda_mem_alloc_mb"] = float(th.cuda.memory_allocated(device_index) / (1024.0**2))
                snapshot["system/cuda_mem_reserved_mb"] = float(th.cuda.memory_reserved(device_index) / (1024.0**2))
            except Exception:
                pass

        self.history.append(snapshot)
        self._last_time = now
        self._last_timesteps = timesteps

    def _build_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        final = self.history[-1]
        wall_time_s = float(final.get("time/elapsed_s", 0.0))
        train_timesteps = int(final.get("time/train_timesteps", 0.0))
        rollouts = int(final.get("time/rollout_iteration", 0.0))
        summary: Dict[str, Any] = {
            "wall_time_s": wall_time_s,
            "train_timesteps": train_timesteps,
            "rollout_iterations": rollouts,
            "timesteps_per_s_avg": float(train_timesteps / wall_time_s) if wall_time_s > 0 else 0.0,
            "iterations_per_s_avg": float(rollouts / wall_time_s) if wall_time_s > 0 else 0.0,
            "timesteps_per_s_peak": float(
                max((row.get("time/timesteps_per_s_inst", 0.0) for row in self.history), default=0.0)
            ),
            "episodes_completed": int(self.episode_callback.episode_count) if self.episode_callback else 0,
            "events_total": {key: int(value) for key, value in sorted(self._event_totals.items())},
            "terminal_events_total": {key: int(value) for key, value in sorted(self._terminal_event_totals.items())},
            "env_dt_s": float(self._env_dt_s),
        }
        if "rollout/ep_rew_mean_100" in final:
            summary["final_ep_rew_mean_100"] = float(final["rollout/ep_rew_mean_100"])
        if "rollout/ep_len_mean_100" in final:
            summary["final_ep_len_mean_100"] = float(final["rollout/ep_len_mean_100"])
        if "rollout/ep_len_sim_s_mean_100" in final:
            summary["final_ep_len_sim_s_mean_100"] = float(final["rollout/ep_len_sim_s_mean_100"])
        if "rollout/ep_rew_per_step_mean_100" in final:
            summary["final_ep_rew_per_step_mean_100"] = float(final["rollout/ep_rew_per_step_mean_100"])
        if "rollout/episodes_per_s_avg" in final:
            summary["episodes_per_s_avg"] = float(final["rollout/episodes_per_s_avg"])
        if "rollout/gates_passed_per_episode_mean" in final:
            summary["gates_passed_per_episode_mean"] = float(final["rollout/gates_passed_per_episode_mean"])
        if "rollout/gates_passed_per_episode_mean_100" in final:
            summary["gates_passed_per_episode_mean_100"] = float(final["rollout/gates_passed_per_episode_mean_100"])
        if "rollout/gates_completion_ratio_mean_100" in final:
            summary["gates_completion_ratio_mean_100"] = float(final["rollout/gates_completion_ratio_mean_100"])
        if self.eval_callback is not None and self.eval_callback.eval_history:
            best = max(self.eval_callback.eval_history, key=lambda item: item["mean_reward"])
            summary["eval_best_mean_reward"] = float(best["mean_reward"])
            summary["eval_best_step"] = int(best["timesteps"])
        if self.episode_history:
            ep_rewards = np.array([self._as_float(row.get("episode/reward", np.nan)) for row in self.episode_history], dtype=np.float64)
            ep_len_steps = np.array(
                [self._as_float(row.get("episode/len_steps", np.nan)) for row in self.episode_history], dtype=np.float64
            )
            ep_len_wall_s = np.array(
                [self._as_float(row.get("episode/len_wall_s", np.nan)) for row in self.episode_history], dtype=np.float64
            )
            if np.isfinite(ep_rewards).any():
                summary["episode_reward_mean"] = float(np.nanmean(ep_rewards))
                summary["episode_reward_median"] = float(np.nanmedian(ep_rewards))
                summary["episode_reward_p10"] = float(np.nanpercentile(ep_rewards, 10))
                summary["episode_reward_p90"] = float(np.nanpercentile(ep_rewards, 90))
            if np.isfinite(ep_len_steps).any():
                summary["episode_len_steps_mean"] = float(np.nanmean(ep_len_steps))
                summary["episode_len_steps_median"] = float(np.nanmedian(ep_len_steps))
                summary["episode_len_sim_s_mean"] = float(np.nanmean(ep_len_steps) * float(self._env_dt_s))
            if np.isfinite(ep_len_wall_s).any():
                summary["episode_len_wall_s_mean"] = float(np.nanmean(ep_len_wall_s))
                summary["episode_len_wall_s_p90"] = float(np.nanpercentile(ep_len_wall_s, 90))

        alerts = []
        approx_kl = self._extract_series("train/approx_kl")
        clip_frac = self._extract_series("train/clip_fraction")
        explained_var = self._extract_series("train/explained_variance")
        if approx_kl.size > 0 and np.isfinite(approx_kl).any() and np.nanmax(approx_kl) > 0.15:
            alerts.append("High KL spike (>0.15): policy updates may be unstable.")
        if approx_kl.size > 0 and np.isfinite(approx_kl).any() and np.nanmax(approx_kl) > 2.0:
            alerts.append("Very high KL spike (>2.0): likely catastrophic PPO update(s).")
        if clip_frac.size > 0 and np.isfinite(clip_frac).any() and np.nanmean(clip_frac[-10:]) > 0.35:
            alerts.append("High clip fraction in final rollouts (>0.35): policy updates may be too aggressive.")
        if explained_var.size > 0 and np.isfinite(explained_var).any() and np.nanmean(explained_var[-10:]) < 0.0:
            alerts.append("Negative explained variance near end: value function fit appears poor.")
        if alerts:
            summary["alerts"] = alerts
        model_logger = getattr(self.model, "logger", None)
        logger_values = dict(getattr(model_logger, "name_to_value", {}))
        for key in self.TRAIN_LOG_KEYS:
            value = self._as_float(logger_values.get(key, None))
            if value is not None:
                summary[f"final_{key.replace('/', '_')}"] = value
        return summary

    def _write_history_csv(self, csv_path: Path) -> None:
        keys = []
        seen = set()
        for row in self.history:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def _write_episode_csv(self, csv_path: Path) -> None:
        if not self.episode_history:
            return
        keys = []
        seen = set()
        for row in self.episode_history:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.episode_history:
                writer.writerow(row)

    def _plot_series(self, ax, x: np.ndarray, y: np.ndarray, label: str) -> bool:
        if y.size == 0 or not np.isfinite(y).any():
            return False
        ax.plot(x, y, label=label, linewidth=1.8)
        return True

    def _extract_series(self, key: str) -> np.ndarray:
        return np.array([self._as_float(row.get(key, np.nan)) for row in self.history], dtype=np.float64)

    def _extract_episode_series(self, key: str) -> np.ndarray:
        return np.array([self._as_float(row.get(key, np.nan)) for row in self.episode_history], dtype=np.float64)

    def _save_plots(self, out_path: Path) -> None:
        x_time = self._extract_series("time/elapsed_s")
        x_steps = self._extract_series("time/train_timesteps")

        fig, axs = plt.subplots(4, 2, figsize=(16, 18), sharex=False)

        # Per-episode reward trajectory (raw + rolling mean)
        ep_steps = self._extract_episode_series("time/timesteps")
        ep_reward = self._extract_episode_series("episode/reward")
        ep_reward_ax = axs[0, 0]
        has_ep_reward = ep_steps.size > 0 and ep_reward.size > 0 and np.isfinite(ep_reward).any()
        if has_ep_reward:
            ep_reward_ax.scatter(ep_steps, ep_reward, s=8, alpha=0.25, label="episode reward (raw)")
            reward_ma = self._rolling_mean(ep_reward, window=100)
            ep_reward_ax.plot(ep_steps, reward_ma, linewidth=2.0, color="tab:blue", label="rolling mean (100)")
            ep_reward_ax.set_title("Episode Reward vs Train Steps")
            ep_reward_ax.set_xlabel("Train timesteps")
            ep_reward_ax.set_ylabel("Return")
            ep_reward_ax.grid(True, alpha=0.3)
            ep_reward_ax.legend(loc="best")
        else:
            ep_reward_ax.text(0.5, 0.5, "No per-episode reward data", ha="center", va="center")

        # Episode closure time: simulation duration and wall-clock duration
        ep_len_ax = axs[0, 1]
        ep_len_steps = self._extract_episode_series("episode/len_steps")
        ep_len_sim_s = self._extract_episode_series("episode/len_sim_s")
        ep_len_wall_s = self._extract_episode_series("episode/len_wall_s")
        has_len = False
        if ep_steps.size > 0 and np.isfinite(ep_len_steps).any():
            has_len = True
            ep_len_ax.plot(ep_steps, self._rolling_mean(ep_len_steps, 100), linewidth=2.0, label="ep length [steps]")
        if ep_steps.size > 0 and np.isfinite(ep_len_sim_s).any():
            has_len = True
            ep_len_ax.plot(ep_steps, self._rolling_mean(ep_len_sim_s, 100), linewidth=2.0, label="ep length [sim s]")
        if ep_steps.size > 0 and np.isfinite(ep_len_wall_s).any():
            has_len = True
            ep_len_ax.plot(ep_steps, self._rolling_mean(ep_len_wall_s, 100), linewidth=2.0, label="ep duration [wall s]")
        if has_len:
            ep_len_ax.set_title("Episode Closure Time")
            ep_len_ax.set_xlabel("Train timesteps")
            ep_len_ax.grid(True, alpha=0.3)
            ep_len_ax.legend(loc="best")
        else:
            ep_len_ax.text(0.5, 0.5, "No per-episode length/time data", ha="center", va="center")

        # PPO optimization losses
        loss_ax = axs[1, 0]
        has_loss = False
        has_loss |= self._plot_series(loss_ax, x_steps, self._extract_series("train/loss"), "total loss")
        has_loss |= self._plot_series(loss_ax, x_steps, self._extract_series("train/value_loss"), "value loss")
        has_loss |= self._plot_series(loss_ax, x_steps, self._extract_series("train/policy_gradient_loss"), "policy grad loss")
        has_loss |= self._plot_series(loss_ax, x_steps, self._extract_series("train/entropy_loss"), "entropy loss")
        if has_loss:
            loss_ax.set_title("PPO Objective Terms")
            loss_ax.set_xlabel("Train timesteps")
            loss_ax.grid(True, alpha=0.3)
            loss_ax.legend(loc="best")
        else:
            loss_ax.text(0.5, 0.5, "No PPO loss data", ha="center", va="center")

        # Trust-region and policy noise diagnostics
        stab_ax = axs[1, 1]
        has_stab = False
        has_stab |= self._plot_series(stab_ax, x_steps, self._extract_series("train/approx_kl"), "approx_kl")
        has_stab |= self._plot_series(stab_ax, x_steps, self._extract_series("train/clip_fraction"), "clip_fraction")
        has_stab |= self._plot_series(stab_ax, x_steps, self._extract_series("train/explained_variance"), "explained_variance")
        has_stab |= self._plot_series(stab_ax, x_steps, self._extract_series("train/std"), "policy std")
        if has_stab:
            stab_ax.set_title("Policy Stability & Critic Fit")
            stab_ax.set_xlabel("Train timesteps")
            stab_ax.grid(True, alpha=0.3)
            stab_ax.legend(loc="best")
        else:
            stab_ax.text(0.5, 0.5, "No stability data", ha="center", va="center")

        # Throughput diagnostics (step/s and episodes/s)
        speed_ax = axs[2, 0]
        has_speed = False
        has_speed |= self._plot_series(speed_ax, x_time, self._extract_series("time/timesteps_per_s_avg"), "steps/s avg")
        has_speed |= self._plot_series(speed_ax, x_time, self._extract_series("time/timesteps_per_s_inst"), "steps/s inst")
        has_speed |= self._plot_series(speed_ax, x_time, self._extract_series("rollout/episodes_per_s_avg"), "episodes/s avg")
        has_speed |= self._plot_series(speed_ax, x_time, self._extract_series("rollout/episodes_per_s_inst"), "episodes/s inst")
        if has_speed:
            speed_ax.set_title("Training Throughput")
            speed_ax.set_xlabel("Wall time [s]")
            speed_ax.grid(True, alpha=0.3)
            speed_ax.legend(loc="best")
        else:
            speed_ax.text(0.5, 0.5, "No throughput data", ha="center", va="center")

        # Rollout aggregate quality (100-episode averages)
        quality_ax = axs[2, 1]
        has_quality = False
        has_quality |= self._plot_series(quality_ax, x_steps, self._extract_series("rollout/ep_rew_mean_100"), "ep_rew_mean_100")
        has_quality |= self._plot_series(quality_ax, x_steps, self._extract_series("rollout/ep_len_mean_100"), "ep_len_mean_100")
        has_quality |= self._plot_series(
            quality_ax, x_steps, self._extract_series("rollout/ep_rew_per_step_mean_100"), "ep_rew_per_step_100"
        )
        if has_quality:
            quality_ax.set_title("Rollout Quality (100-Episode Means)")
            quality_ax.set_xlabel("Train timesteps")
            quality_ax.grid(True, alpha=0.3)
            quality_ax.legend(loc="best")
        else:
            quality_ax.text(0.5, 0.5, "No rollout quality data", ha="center", va="center")

        # Evaluation learning curve
        eval_ax = axs[3, 0]
        if self.eval_callback is not None and self.eval_callback.eval_history:
            eval_steps = np.array([item["timesteps"] for item in self.eval_callback.eval_history], dtype=np.float64)
            eval_mean = np.array([item["mean_reward"] for item in self.eval_callback.eval_history], dtype=np.float64)
            eval_std = np.array([item["std_reward"] for item in self.eval_callback.eval_history], dtype=np.float64)
            eval_ax.plot(eval_steps, eval_mean, label="eval mean reward", linewidth=2.0)
            eval_ax.fill_between(eval_steps, eval_mean - eval_std, eval_mean + eval_std, alpha=0.2, label="mean +/- std")
            eval_ax.set_title("Evaluation Performance")
            eval_ax.set_xlabel("Train timesteps")
            eval_ax.grid(True, alpha=0.3)
            eval_ax.legend(loc="best")
        else:
            eval_ax.text(0.5, 0.5, "No evaluation history", ha="center", va="center")

        # Event outcomes
        events_ax = axs[3, 1]
        event_keys = sorted([key for key in self.history[-1].keys() if key.startswith("events_terminal_total/")])
        event_title = "Terminal Event Totals"
        if not event_keys:
            event_keys = sorted([key for key in self.history[-1].keys() if key.startswith("events_total/")])
            event_title = "Event Totals"
        if event_keys:
            event_names = [key.split("/", 1)[1] for key in event_keys]
            counts = [float(self.history[-1].get(key, 0.0)) for key in event_keys]
            events_ax.bar(event_names, counts, color="tab:blue", alpha=0.85)
            events_ax.set_title(event_title)
            events_ax.set_ylabel("Count")
            events_ax.grid(True, axis="y", alpha=0.3)
            for tick in events_ax.get_xticklabels():
                tick.set_rotation(20)
        else:
            events_ax.text(0.5, 0.5, "No event totals", ha="center", va="center")

        fig.suptitle("Training Diagnostics (RL + System)", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    def _on_training_end(self) -> None:
        if not self.history:
            if self.verbose:
                print("[TrainReport] No rollout history collected.")
            return

        out_dir = self.log_dir / "training_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "rollout_metrics.csv"
        episode_csv_path = out_dir / "episode_metrics.csv"
        summary_path = out_dir / "summary.json"
        plot_path = out_dir / "training_summary.png"

        self._write_history_csv(csv_path)
        self._write_episode_csv(episode_csv_path)
        summary = self._build_summary()
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        self._save_plots(plot_path)

        if self.verbose:
            print(f"[TrainReport] CSV: {csv_path}")
            if self.episode_history:
                print(f"[TrainReport] Episode CSV: {episode_csv_path}")
            print(f"[TrainReport] Plot: {plot_path}")
            print(f"[TrainReport] Summary: {summary_path}")
            print(
                "[TrainReport] "
                f"wall_time={summary.get('wall_time_s', 0.0):.1f}s "
                f"timesteps/s={summary.get('timesteps_per_s_avg', 0.0):.1f} "
                f"iter/s={summary.get('iterations_per_s_avg', 0.0):.3f}"
            )


def resolve_policy_class(policy_type: str, mpc_backend: Optional[str] = None):
    policy_type = str(policy_type)
    mpc_backend = None if mpc_backend is None else str(mpc_backend).lower()
    if policy_type == "acmpc_mlp":
        return MlpMpcPolicy
    if policy_type == "acmpc_diffmpc":
        if mpc_backend == "pytorch":
            return MlpMpcPolicy
        return MlpMpcPolicyDiffMPC
    if policy_type == "mlp_only":
        return MlpOnlyPolicy
    raise ValueError(f"Unknown policy_type: {policy_type}")


def resolve_eval_backend_request(policy_type: str, mpc_backend: str) -> str:
    """
    For acmpc_diffmpc, evaluate both DifferentialMPC and FastMPC(CUDA).
    """
    if str(policy_type) == "acmpc_diffmpc":
        return "both"
    return str(mpc_backend).lower()


def _resolve_resume_paths(resume_path: str, load_vecnormalize: bool) -> Tuple[Path, Optional[Path]]:
    ckpt_path = Path(resume_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume.path not found: {ckpt_path}")
    vecnorm_path = ckpt_path.with_name(ckpt_path.name + ".vecnorm.pkl") if load_vecnormalize else None
    if vecnorm_path is not None and not vecnorm_path.exists():
        print(f"[Resume] Warning: VecNormalize stats not found: {vecnorm_path}. Continuing without loading stats.")
        vecnorm_path = None
    return ckpt_path, vecnorm_path
