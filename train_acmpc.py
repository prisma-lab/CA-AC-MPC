#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch as th

# Setup paths for vendored SB3, MPC, and policy modules.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "mpc.pytorch"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "diff_mpc_drones"))
sys.path.insert(0, str(ROOT / "differentialMPCPerformance"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "training_modules"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "stable-baselines3-acmpc-acmpc"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from utils.acmpc_obs_extractor import AcmpcPolicyObsExtractor
from utils.evaluate_acmpc2 import evaluate_model
from utils.train_config import apply_overrides, deep_update, load_yaml
from utils.train_support import (
    CheckpointEvalCallback,
    EpisodeCountCallback,
    TrainingDebugReportCallback,
    _append_track_dir,
    _append_track_to_save_path,
    _build_learning_rate,
    _build_training_signature,
    _check_signature_compatibility,
    _find_vecnormalize,
    _load_checkpoint_metadata,
    _make_vec_env_from_cfg,
    _report_policy_init_stats,
    _resolve_resume_paths,
    _run_input_sanity_checks,
    _track_tag,
    resolve_eval_backend_request,
    resolve_policy_class,
    save_with_vecnorm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACMPC using a YAML config from configs/."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file path, for example configs/train_acmpc_fixed_map.yaml.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config keys, e.g. --override ppo.total_timesteps=50000",
    )
    parser.add_argument(
        "--track-config",
        type=str,
        default=None,
        help="Optional YAML track config path. If set, training/eval use this fixed map only.",
    )
    return parser.parse_args()


def _resolve_track_config_path(path_raw: Optional[str]) -> Optional[str]:
    if path_raw is None:
        return None
    path_raw = str(path_raw).strip()
    if path_raw == "":
        return None
    p = Path(path_raw).expanduser()
    if not p.is_absolute():
        if p.exists():
            p = p.resolve()
        else:
            candidate = (ROOT / p).resolve()
            if candidate.exists():
                p = candidate
    if not p.exists():
        raise FileNotFoundError(f"track_config_path not found: {path_raw}")
    return str(p)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)

    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "auto"))
    policy_type = str(cfg.get("policy_type", "mlp_only"))
    mpc_backend = str(cfg.get("mpc_backend", "auto")).lower()
    mpc_horizon = int(cfg.get("mpc_horizon", 2))
    mpc_max_iter = int(cfg.get("mpc_max_iter", 10))
    if mpc_max_iter <= 0:
        raise ValueError("mpc_max_iter must be > 0.")
    log_std_init = float(cfg.get("log_std_init", -2.0))

    # Leading dims of the observation that represent the raw MPC state [p,q,v].
    mpc_state_dim = int(cfg.get("mpc_state_dim", 10))

    # Policy needs this environment variable (used by MPC-backed policy modules).
    os.environ["ACMPC_T"] = str(mpc_horizon)
    os.environ["ACMPC_MPC_MAX_ITER"] = str(mpc_max_iter)
    print(f"[MPC] horizon={mpc_horizon} max_iter={mpc_max_iter}")
    if mpc_backend == "auto":
        if policy_type == "acmpc_mlp":
            mpc_backend = "pytorch"
        elif policy_type == "acmpc_diffmpc":
            mpc_backend = "fast" if th.cuda.is_available() else "diffmpc"
        else:
            mpc_backend = "diffmpc"
    if mpc_backend not in {"pytorch", "diffmpc", "fast"}:
        raise ValueError(f"Unsupported mpc_backend={mpc_backend}. Use one of: pytorch, diffmpc, fast.")
    os.environ["ACMPC_MPC_BACKEND"] = mpc_backend

    if device.startswith("cuda") and not th.cuda.is_available():
        print("[Warning] device=cuda requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = "cpu"

    env_cfg = cfg.get("env", {}) or {}
    track = str(env_cfg.get("track", "all"))
    env_kwargs = dict(env_cfg.get("kwargs", {}) or {})
    track_config_path = _resolve_track_config_path(
        args.track_config if args.track_config is not None else env_cfg.get("track_config_path", None)
    )
    if track_config_path is not None:
        env_kwargs["track_config_path"] = track_config_path
        print(f"[Track] Using fixed map config: {track_config_path}")

    vec_cfg = cfg.get("vec_env", {}) or {}
    n_envs = int(vec_cfg.get("n_envs", 32))
    vec_type = str(vec_cfg.get("type", "dummy"))
    normalize_obs = bool(vec_cfg.get("normalize_obs", True))
    clip_obs = float(vec_cfg.get("clip_obs", 10.0))

    ppo_cfg = cfg.get("ppo", {}) or {}
    total_timesteps = int(ppo_cfg.get("total_timesteps", 2_000_000))
    n_steps = int(ppo_cfg.get("n_steps", 128))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    n_epochs = int(ppo_cfg.get("n_epochs", 10))
    learning_rate = _build_learning_rate(ppo_cfg)
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    clip_range = float(ppo_cfg.get("clip_range", 0.2))
    target_kl_raw = ppo_cfg.get("target_kl", None)
    target_kl = None if target_kl_raw is None else float(target_kl_raw)
    ent_coef = float(ppo_cfg.get("ent_coef", 0.0))
    vf_coef = float(ppo_cfg.get("vf_coef", 0.5))
    max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))

    log_cfg = cfg.get("logging", {}) or {}
    log_dir = str(log_cfg.get("log_dir", "./runs/latest"))
    save_path = str(log_cfg.get("save_path", "acmpc_model.zip"))
    episode_print_freq = int(log_cfg.get("episode_print_freq", 0))
    progress_bar = bool(log_cfg.get("progress_bar", True))
    auto_track_naming = bool(log_cfg.get("auto_track_naming", True))

    ckpt_cfg = cfg.get("checkpoint", {}) or {}
    checkpoint_dir = str(ckpt_cfg.get("dir", "checkpoints"))
    checkpoint_freq = int(ckpt_cfg.get("freq", 0))

    if auto_track_naming:
        tag = _track_tag(track=track, track_config_path=track_config_path)
        log_dir = _append_track_dir(log_dir, tag)
        checkpoint_dir = _append_track_dir(checkpoint_dir, tag)
        if save_path:
            save_path = _append_track_to_save_path(save_path, tag)
        print(f"[Paths] auto_track_naming=true tag={tag}")
        print(f"[Paths] log_dir={log_dir}")
        print(f"[Paths] checkpoint_dir={checkpoint_dir}")
        if save_path:
            print(f"[Paths] save_path={save_path}")

    training_signature = _build_training_signature(
        policy_type=policy_type,
        mpc_backend=mpc_backend,
        mpc_horizon=mpc_horizon,
        track=track,
        track_config_path=track_config_path,
    )
    print(
        "[Signature] "
        f"policy_type={training_signature['policy_type']} "
        f"mpc_backend={training_signature['mpc_backend']} "
        f"mpc_horizon={training_signature['mpc_horizon']} "
        f"track={training_signature['track_key']}"
    )

    resume_cfg = cfg.get("resume", {}) or {}
    resume_enabled = bool(resume_cfg.get("enabled", False))
    resume_path_raw = str(resume_cfg.get("path", "")).strip()
    resume_reset_num_timesteps = bool(resume_cfg.get("reset_num_timesteps", False))
    resume_load_vecnormalize = bool(resume_cfg.get("load_vecnormalize", True))
    resume_auto_best_if_compatible = bool(resume_cfg.get("auto_best_if_compatible", True))
    if "auto_best_path" in resume_cfg:
        resume_auto_best_path_raw = str(resume_cfg.get("auto_best_path", "")).strip()
    else:
        resume_auto_best_path_raw = str(Path(checkpoint_dir) / "best_model.zip")
    resume_ckpt_path: Optional[Path] = None
    resume_vecnorm_path: Optional[Path] = None

    if resume_enabled:
        if not resume_path_raw:
            raise ValueError("resume.enabled=true but resume.path is empty.")
        resume_ckpt_path, resume_vecnorm_path = _resolve_resume_paths(
            resume_path_raw, load_vecnormalize=resume_load_vecnormalize
        )
        meta = _load_checkpoint_metadata(resume_ckpt_path)
        observed_sig = meta.get("signature") if isinstance(meta, dict) else None
        ok, reason = _check_signature_compatibility(training_signature, observed_sig)
        if not ok:
            print(f"[Resume] Warning: manual checkpoint compatibility check failed: {reason}")
    elif resume_auto_best_if_compatible:
        auto_path = Path(resume_auto_best_path_raw).expanduser()
        if auto_path.exists():
            auto_meta = _load_checkpoint_metadata(auto_path)
            observed_sig = auto_meta.get("signature") if isinstance(auto_meta, dict) else None
            ok, reason = _check_signature_compatibility(training_signature, observed_sig)
            if ok:
                resume_ckpt_path, resume_vecnorm_path = _resolve_resume_paths(
                    str(auto_path), load_vecnormalize=resume_load_vecnormalize
                )
                resume_enabled = True
                print(f"[Resume] Auto-enabled from best checkpoint: {resume_ckpt_path}")
            else:
                print(f"[Resume] Auto-skip best checkpoint: {reason}")
        else:
            print(f"[Resume] Auto-skip best checkpoint: not found at {auto_path}")

    eval_cfg = cfg.get("eval", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", False))
    eval_freq = int(eval_cfg.get("freq", 0)) if eval_enabled else 0
    eval_episodes = int(eval_cfg.get("episodes", 20))
    eval_n_envs = int(eval_cfg.get("n_envs", 1))
    eval_track = str(eval_cfg.get("track", track))
    eval_deterministic = bool(eval_cfg.get("deterministic", True))
    eval_env_kwargs = dict(eval_cfg.get("env_kwargs", {}) or {})
    eval_plot_enabled = bool(eval_cfg.get("plot_enabled", False))
    eval_plot_freq = int(eval_cfg.get("plot_freq", eval_freq))

    post_cfg = cfg.get("post_eval", {}) or {}
    post_eval_enabled = bool(post_cfg.get("enabled", False))
    post_eval_max_steps_cap = post_cfg.get("max_steps_cap", None)
    if post_eval_max_steps_cap is not None:
        post_eval_max_steps_cap = int(post_eval_max_steps_cap)

    os.makedirs(log_dir, exist_ok=True)

    train_env = _make_vec_env_from_cfg(
        track=track,
        env_kwargs=env_kwargs,
        seed=seed,
        n_envs=n_envs,
        vec_type=vec_type,
        normalize_obs=normalize_obs,
        clip_obs=clip_obs,
        log_dir=log_dir,
        state_dim=mpc_state_dim,
        vecnorm_stats_path=str(resume_vecnorm_path) if resume_vecnorm_path is not None else None,
    )

    train_vecnorm = _find_vecnormalize(train_env)
    _run_input_sanity_checks(
        train_env=train_env,
        mpc_state_dim=mpc_state_dim,
        normalize_obs=normalize_obs,
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    merged_eval_kwargs = deep_update(env_kwargs, eval_env_kwargs)
    eval_env = None
    if eval_freq > 0:
        eval_env = _make_vec_env_from_cfg(
            track=eval_track,
            env_kwargs=merged_eval_kwargs,
            seed=seed,
            n_envs=eval_n_envs,
            vec_type="dummy",
            normalize_obs=normalize_obs,
            clip_obs=clip_obs,
            log_dir=log_dir,
            state_dim=mpc_state_dim,
            train_vecnorm=train_vecnorm,
        )

    episode_cb = EpisodeCountCallback(print_every_episodes=episode_print_freq, verbose=1)
    ckpt_eval_cb = CheckpointEvalCallback(
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=checkpoint_freq,
        eval_env=eval_env,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        eval_deterministic=eval_deterministic,
        state_dim=mpc_state_dim,
        periodic_plot_enabled=eval_plot_enabled,
        periodic_plot_freq=eval_plot_freq,
        periodic_plot_log_dir=log_dir,
        periodic_plot_policy_type=policy_type,
        periodic_plot_mpc_backend=mpc_backend,
        periodic_plot_env_kwargs=merged_eval_kwargs,
        periodic_plot_seed=seed,
        periodic_plot_device=device,
        periodic_plot_track_config_path=track_config_path,
        periodic_plot_track_name=None if track_config_path is not None else track,
        periodic_plot_max_steps_cap=post_eval_max_steps_cap,
        training_signature=training_signature,
        verbose=1,
    )
    report_cb = TrainingDebugReportCallback(
        log_dir=log_dir,
        episode_callback=episode_cb,
        eval_callback=ckpt_eval_cb,
        verbose=1,
    )
    callback = CallbackList([episode_cb, ckpt_eval_cb, report_cb])

    policy_class = resolve_policy_class(policy_type, mpc_backend=mpc_backend)

    policy_kwargs = {
        # The cost network should only see the "paper observation" tail [v, R, o_track].
        "features_extractor_class": AcmpcPolicyObsExtractor,
        "features_extractor_kwargs": {"mpc_state_dim": mpc_state_dim},
        "log_std_init": log_std_init,
    }

    if resume_enabled:
        print(f"[Resume] Loading model from: {resume_ckpt_path}")
        if resume_vecnorm_path is not None:
            print(f"[Resume] Loaded VecNormalize stats from: {resume_vecnorm_path}")
        model = PPO.load(
            str(resume_ckpt_path),
            env=train_env,
            device=device,
            custom_objects={"policy_class": policy_class},
        )
        # Keep resumed runs writing logs under the current config log_dir.
        model.tensorboard_log = log_dir
        _report_policy_init_stats(model, policy_type=policy_type, resumed=True)
    else:
        model = PPO(
            policy_class,
            train_env,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            target_kl=target_kl,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            seed=seed,
            device=device,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
        )
        _report_policy_init_stats(model, policy_type=policy_type, resumed=False)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=progress_bar,
        callback=callback,
        reset_num_timesteps=resume_reset_num_timesteps if resume_enabled else True,
    )

    if save_path:
        save_with_vecnorm(
            model,
            train_env,
            save_path,
            metadata={
                "kind": "final",
                "num_timesteps": int(model.num_timesteps),
                "signature": dict(training_signature),
            },
        )

    if post_eval_enabled:
        print("\n--- Starting Post-Training Evaluation ---")
        best_ckpt_path = Path(checkpoint_dir) / "best_model.zip"
        selected_eval_model_path: Optional[str] = None

        if best_ckpt_path.exists():
            selected_eval_model_path = str(best_ckpt_path)
            print(f"[PostEval] Using best checkpoint: {selected_eval_model_path}")
        elif save_path and Path(save_path).exists():
            selected_eval_model_path = save_path
            print(f"[PostEval] best_model.zip not found, using final checkpoint: {selected_eval_model_path}")
        else:
            raise FileNotFoundError(
                "Post-eval enabled but no checkpoint available. "
                f"Checked best={best_ckpt_path} and final={save_path!r}."
            )

        eval_backend_request = resolve_eval_backend_request(policy_type, mpc_backend)
        evaluate_model(
            selected_eval_model_path,
            log_dir,
            track_config_path=track_config_path,
            device=device,
            policy_type=policy_type,
            mpc_backend=eval_backend_request,
            single_track=None if track_config_path is not None else track,
            env_kwargs=env_kwargs,
            seed=seed,
            max_steps_cap=post_eval_max_steps_cap,
        )

    print(
        f"[Train] done: timesteps={total_timesteps} n_envs={n_envs} episodes_completed={episode_cb.episode_count}"
    )

    if eval_env is not None:
        eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
