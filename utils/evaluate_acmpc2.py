import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

def _parse_env_kwargs(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw == "":
        return {}
    import json

    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--env-kwargs must be a JSON dict.")
    return data


def _extract_cost_params_t0(raw_costs: Optional[torch.Tensor | np.ndarray]) -> np.ndarray:
    """
    Convert policy `last_cost_vectors` into a 28-dim vector for plotting.
    """
    if raw_costs is None:
        return np.zeros(28, dtype=np.float32)

    if isinstance(raw_costs, torch.Tensor):
        arr = raw_costs.detach().cpu().numpy()
    else:
        arr = np.asarray(raw_costs)

    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.reshape(-1)

    if arr.size < 28:
        out = np.zeros(28, dtype=np.float32)
        out[: arr.size] = arr.astype(np.float32)
        return out

    if arr.size % 28 == 0:
        horizon = arr.size // 28
        q_all = arr[: 14 * horizon]
        p_all = arr[14 * horizon : 28 * horizon]
        return np.concatenate([q_all[:14], p_all[:14]]).astype(np.float32)

    return arr[:28].astype(np.float32)


def _denormalize_action(action_norm: np.ndarray, env) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Convert normalized action [-1, 1] to physical controls.
    """
    normalization_max = 8.5
    mass = 0.752
    omega_max = np.array([10.0, 10.0, 4.0], dtype=np.float32)
    if hasattr(env, "dx"):
        normalization_max = float(getattr(env.dx, "thrust_max", normalization_max))
        mass = float(getattr(env.dx, "mass", mass))
        try:
            omega_max = env.dx.omega_max.detach().cpu().numpy().astype(np.float32)
        except Exception:
            omega_max = np.array(env.dx.omega_max, dtype=np.float32)

    max_acc = (normalization_max * 4) / mass
    force_mean = 9.8066
    force_std = max(1e-6, max_acc - force_mean)

    thrust_mass_norm = float(action_norm[0]) * force_std + force_mean
    thrust_mass_norm = float(np.clip(thrust_mass_norm, 0.0, max_acc))
    thrust_phys = thrust_mass_norm * mass
    rates_phys = action_norm[1:] * omega_max
    physical_u = np.concatenate([[thrust_phys], rates_phys.astype(np.float32)], axis=0).astype(np.float32)
    return physical_u, thrust_mass_norm, omega_max

# Add paths (this repo vendors a fork of SB3 and MPC code)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "mpc.pytorch"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "diff_mpc_drones"))
sys.path.insert(0, str(ROOT / "differentialMPCPerformance"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "training_modules"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "stable-baselines3-acmpc-acmpc"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.gate_racing_env import GateRacingEnv
from utils.plotting import (
    plot_trajectory_3d,
    plot_states,
    plot_controls,
    plot_trajectory_top_down,
    plot_metrics,
    save_speed_direction_video_topdown,
)
from mlp_mpc_policy import MlpMpcPolicy
from mlp_mpc_policy_diffmpc import MlpMpcPolicyDiffMPC
from mlp_only_policy import MlpOnlyPolicy


def resolve_eval_policy_class(policy_type: str, mpc_backend: str = "auto"):
    policy_type = str(policy_type)
    mpc_backend = str(mpc_backend).lower()
    if policy_type == "acmpc_mlp":
        return MlpMpcPolicy
    if policy_type == "acmpc_diffmpc":
        if mpc_backend == "pytorch":
            return MlpMpcPolicy
        return MlpMpcPolicyDiffMPC
    if policy_type == "mlp_only":
        return MlpOnlyPolicy
    raise ValueError(f"Unknown policy type: {policy_type}")


def _plot_spawn_batch(
    env: GateRacingEnv,
    plot_dir: Path,
    name: str,
    *,
    sample_count: int = 64,
    seed: int = 0,
) -> None:
    sample_count = max(1, int(sample_count))
    seed = int(seed)

    spawn_world = []
    gate_world = []
    local_offsets = []
    gate_indices = []

    for i in range(sample_count):
        try:
            env.reset(seed=seed + i)
        except TypeError:
            env.reset()

        if len(env.gates) == 0:
            continue

        gate_idx = int(getattr(env, "target_gate_idx", 0))
        gate_idx = max(0, min(gate_idx, len(env.gates) - 1))
        gate = env.gates[gate_idx]

        spawn_pos = env.state[0:3].detach().cpu().numpy().astype(np.float32)
        rel = spawn_pos - gate.center
        right_axis = gate.R[:, 0]
        up_axis = gate.R[:, 1]
        normal_axis = gate.normal

        d_forward = float(np.dot(rel, normal_axis))
        d_right = float(np.dot(rel, right_axis))
        d_up = float(np.dot(rel, up_axis))

        spawn_world.append(spawn_pos)
        gate_world.append(gate.center.astype(np.float32))
        local_offsets.append([d_forward, d_right, d_up])
        gate_indices.append(gate_idx)

    if len(spawn_world) == 0:
        print(f"[SpawnBatch] skipped for {name}: no gates available.")
        return

    spawn_world_np = np.asarray(spawn_world, dtype=np.float32)
    gate_world_np = np.asarray(gate_world, dtype=np.float32)
    local_offsets_np = np.asarray(local_offsets, dtype=np.float32)
    gate_indices_np = np.asarray(gate_indices, dtype=np.int32)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))

    ax0 = axs[0]
    wind_zone_xy = None
    if hasattr(env, "get_wind_zone_xy_polygon"):
        try:
            wind_zone_xy = env.get_wind_zone_xy_polygon()
        except Exception:
            wind_zone_xy = None
    if wind_zone_xy is not None:
        zone = np.asarray(wind_zone_xy, dtype=np.float32)
        if zone.ndim == 2 and zone.shape[1] == 2 and zone.shape[0] >= 3:
            zone_closed = np.vstack([zone, zone[0]])
            ax0.fill(zone_closed[:, 0], zone_closed[:, 1], color="deepskyblue", alpha=0.12, label="wind zone")
            ax0.plot(zone_closed[:, 0], zone_closed[:, 1], color="deepskyblue", linewidth=1.4, alpha=0.8)
            if hasattr(env, "get_wind_vector_world"):
                try:
                    wind_vec = np.asarray(env.get_wind_vector_world(), dtype=np.float32).reshape(-1)
                    wind_xy = wind_vec[:2] if wind_vec.size >= 2 else np.zeros(2, dtype=np.float32)
                    wnorm = float(np.linalg.norm(wind_xy))
                    if wnorm > 1e-6:
                        center = zone.mean(axis=0)
                        arrow_len = max(0.8, 0.22 * float(np.linalg.norm(zone.max(axis=0) - zone.min(axis=0))))
                        u = (wind_xy / wnorm) * arrow_len
                        ax0.arrow(
                            float(center[0]),
                            float(center[1]),
                            float(u[0]),
                            float(u[1]),
                            head_width=0.16,
                            head_length=0.22,
                            color="deepskyblue",
                            alpha=0.9,
                            length_includes_head=True,
                        )
                        force_n = float(getattr(env, "get_wind_strength_n", lambda: np.nan)())
                        ax0.text(
                            float(center[0]),
                            float(center[1]),
                            f"|a|={float(np.linalg.norm(wind_vec)):.2f} m/s^2 |F|={force_n:.2f} N",
                            color="deepskyblue",
                            fontsize=9,
                            ha="center",
                            va="bottom",
                        )
                except Exception:
                    pass
    ax0.scatter(gate_world_np[:, 0], gate_world_np[:, 1], c=gate_indices_np, cmap="tab20", marker="x", s=60, label="gate")
    ax0.scatter(
        spawn_world_np[:, 0],
        spawn_world_np[:, 1],
        c=gate_indices_np,
        cmap="tab20",
        marker="o",
        s=38,
        alpha=0.85,
        label="spawn",
    )
    for i in range(spawn_world_np.shape[0]):
        ax0.plot(
            [gate_world_np[i, 0], spawn_world_np[i, 0]],
            [gate_world_np[i, 1], spawn_world_np[i, 1]],
            color="gray",
            alpha=0.18,
            linewidth=0.8,
        )
    ax0.set_title("World XY: Gate Center vs Spawn")
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")
    try:
        ax0.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    ax1 = axs[1]
    sc = ax1.scatter(
        local_offsets_np[:, 1],
        local_offsets_np[:, 2],
        c=local_offsets_np[:, 0],
        cmap="coolwarm",
        s=45,
        alpha=0.9,
    )
    ax1.axvline(0.0, color="k", linewidth=1.0, alpha=0.35)
    ax1.axhline(0.0, color="k", linewidth=1.0, alpha=0.35)
    ax1.set_title("Gate Frame: lateral/vertical")
    ax1.set_xlabel("right offset [m]")
    ax1.set_ylabel("up offset [m]")
    ax1.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("forward offset [m]")

    ax2 = axs[2]
    ax2.hist(local_offsets_np[:, 0], bins=min(20, max(8, sample_count // 3)), color="tab:blue", alpha=0.8)
    spawn_margin = float(getattr(env, "spawn_plane_margin", 0.0))
    spawn_dist = float(getattr(env, "spawn_distance", 0.0))
    ax2.axvline(-spawn_margin, color="tab:red", linestyle="--", linewidth=1.8, label="-spawn_plane_margin")
    ax2.axvline(-spawn_dist, color="tab:green", linestyle="--", linewidth=1.8, label="-spawn_distance")
    ax2.set_title("Forward Offset Distribution")
    ax2.set_xlabel("dot(spawn-gate_center, gate_normal) [m]")
    ax2.set_ylabel("count")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    n = int(spawn_world_np.shape[0])
    fig.suptitle(
        f"Spawn Batch ({n} samples): {name}\n"
        f"forward mean={local_offsets_np[:, 0].mean():.3f}m, std={local_offsets_np[:, 0].std():.3f}m",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = plot_dir / f"{name}_spawn_batch_{n}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[SpawnBatch] saved: {out_path}")


def evaluate_model(
    model_path,
    log_dir,
    track_config_path: str | None = None,
    device="cuda",
    policy_type="acmpc_mlp",
    mpc_backend="auto",
    single_track: str | None = None,
    env_kwargs=None,
    seed: int = 0,
    max_steps_cap: int | None = None,
):
    print(f"Loading model from {model_path}...")
    track_config_path = None if track_config_path is None else str(track_config_path).strip()
    single_track = None if single_track is None else str(single_track).strip().lower()
    if track_config_path:
        print(f"Target track config: {track_config_path}")
    elif single_track:
        print(f"Target track: {single_track}")
    else:
        print("Target tracks: splits + circle + straight_wind")

    # Match train behavior
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    env_kwargs = dict(env_kwargs or {})

    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Clean up kwargs that are specific to random generation logic, just in case
    env_kwargs.pop("straight_gate_y_offsets_eval", None)
    env_kwargs.pop("straight_gate_y_eval_offset", None)
    env_kwargs.pop("straight_gate_y_offset_std", None)
    env_kwargs.pop("straight_gate_y_offset_seed", None)
    video_export = bool(env_kwargs.pop("video_export", True))
    video_laps = int(env_kwargs.pop("video_laps", 2))
    video_fps = int(env_kwargs.pop("video_fps", 30))
    if video_laps <= 0:
        raise ValueError("video_laps must be > 0.")
    if video_fps <= 0:
        raise ValueError("video_fps must be > 0.")
    spawn_batch_size = int(env_kwargs.pop("spawn_batch_size", 64))
    if spawn_batch_size <= 0:
        raise ValueError("spawn_batch_size must be > 0.")
    spawn_batch_plot = bool(env_kwargs.pop("spawn_batch_plot", True))

    backend_request = str(mpc_backend).lower().strip()
    if backend_request == "auto":
        if policy_type == "acmpc_mlp":
            requested_backends = ["pytorch"]
        elif policy_type == "acmpc_diffmpc":
            requested_backends = ["diffmpc", "fast"]
        else:
            requested_backends = ["diffmpc"]
    elif backend_request in {"both", "dual", "all", "diffmpc+fast", "fast+diffmpc"}:
        requested_backends = ["diffmpc", "fast"]
    else:
        requested_backends = [backend_request]

    eval_backends = []
    for backend in requested_backends:
        if backend not in {"pytorch", "diffmpc", "fast"}:
            raise ValueError(f"Unsupported mpc_backend={backend}.")
        if backend == "fast" and not torch.cuda.is_available():
            print("[Eval] Skipping FastMPC backend: CUDA is not available.")
            continue
        if backend not in eval_backends:
            eval_backends.append(backend)

    if len(eval_backends) == 0:
        raise RuntimeError("No valid evaluation backend selected.")

    if max_steps_cap is not None:
        max_steps_cap = int(max_steps_cap)
        if max_steps_cap <= 0:
            raise ValueError("--max-steps-cap must be > 0.")

    for backend in eval_backends:
        print(f"\n=== Evaluating backend: {backend} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["ACMPC_MPC_BACKEND"] = backend

        policy_class = resolve_eval_policy_class(policy_type, mpc_backend=backend)
        custom_objects = {"policy_class": policy_class}

        if policy_type in {"acmpc_mlp", "acmpc_diffmpc"} and "ACMPC_T" not in os.environ:
            os.environ["ACMPC_T"] = "2"

        model = PPO.load(model_path, device=device, custom_objects=custom_objects)

        vecnorm_path = model_path + ".vecnorm.pkl"
        if not os.path.exists(vecnorm_path):
            vecnorm_path = None

        plot_dir = Path(log_dir) / f"eval_plots_{backend}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if track_config_path:
            steps = 3000
            kwargs = dict(env_kwargs)
            kwargs["track_config_path"] = track_config_path
            kwargs["max_steps"] = min(steps, max_steps_cap) if max_steps_cap is not None else steps
            if video_export:
                kwargs["loop_track_on_finish"] = True
                kwargs["max_laps_per_episode"] = video_laps
            kwargs.setdefault("debug_info", True)
            env = GateRacingEnv(**kwargs)
            env.seed(seed)

            track_name = Path(track_config_path).stem
            run_name = f"eval_{track_name}"
            if spawn_batch_plot:
                _plot_spawn_batch(env, plot_dir, run_name, sample_count=spawn_batch_size, seed=seed)
            run_episode(
                env,
                model,
                plot_dir,
                run_name,
                vecnorm_path=vecnorm_path,
                save_video=video_export,
                video_fps=video_fps,
                video_title=f"{run_name} [{backend}]",
            )
        elif single_track:
            steps = 3000
            kwargs = dict(env_kwargs)
            kwargs["max_steps"] = min(steps, max_steps_cap) if max_steps_cap is not None else steps
            if video_export:
                kwargs["loop_track_on_finish"] = True
                kwargs["max_laps_per_episode"] = video_laps
            kwargs.setdefault("debug_info", True)
            env = GateRacingEnv(track=single_track, **kwargs)
            env.seed(seed)

            run_name = f"eval_{single_track}"
            if spawn_batch_plot:
                _plot_spawn_batch(env, plot_dir, run_name, sample_count=spawn_batch_size, seed=seed)
            run_episode(
                env,
                model,
                plot_dir,
                run_name,
                vecnorm_path=vecnorm_path,
                save_video=video_export,
                video_fps=video_fps,
                video_title=f"{run_name} [{backend}]",
            )
        else:
            tracks = ["splits", "circle", "straight_wind"]
            for track in tracks:
                print(f"Evaluating on {track}...")
                if track == "splits":
                    steps = 20000
                elif track == "straight_wind":
                    steps = 3000
                else:
                    steps = 2000
                kwargs = dict(env_kwargs)
                kwargs["max_steps"] = min(steps, max_steps_cap) if max_steps_cap is not None else steps
                if video_export:
                    kwargs["loop_track_on_finish"] = True
                    kwargs["max_laps_per_episode"] = video_laps
                kwargs.setdefault("debug_info", True)
                env = GateRacingEnv(track=track, **kwargs)
                env.seed(seed)

                run_name = f"eval_{track}"
                if spawn_batch_plot:
                    _plot_spawn_batch(env, plot_dir, run_name, sample_count=spawn_batch_size, seed=seed)
                run_episode(
                    env,
                    model,
                    plot_dir,
                    run_name,
                    vecnorm_path=vecnorm_path,
                    save_video=video_export,
                    video_fps=video_fps,
                    video_title=f"{run_name} [{backend}]",
                )
        
        print(f"Evaluation complete for backend={backend}. Plots saved to {plot_dir}")


def run_episode(
    env,
    model,
    plot_dir,
    name,
    vecnorm_path=None,
    *,
    save_video: bool = True,
    video_fps: int = 30,
    video_title: str | None = None,
):
    # Always evaluate through a (possibly normalized) VecEnv to match training-time input scaling.
    venv = DummyVecEnv([lambda: env])
    if vecnorm_path is not None:
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    obs = venv.reset()
    done = False

    states = []
    actions = []

    # Metrics storage
    cost_params_history = []
    distances_history = []
    inference_ms_history = []
    wind_accel_norm_history = []
    wind_force_history = []
    wind_active_history = []

    policy_device = torch.device("cpu")
    try:
        policy_device = next(model.policy.parameters()).device
    except Exception:
        policy_device = torch.device("cpu")
    use_cuda_timing = policy_device.type == "cuda" and torch.cuda.is_available()

    states.append(env.state.numpy().copy())
    gates_passed = int(getattr(env, "target_gate_idx", 0))
    terminal_info = {}
    terminal_event = None

    while not done:
        current_drone_state = env.state.numpy()
        
        if use_cuda_timing:
            torch.cuda.synchronize(policy_device)
        t0 = time.perf_counter()
        
        action, _ = model.policy.predict(obs, drone_state=current_drone_state[None, :], deterministic=True)
        
        if use_cuda_timing:
            torch.cuda.synchronize(policy_device)
        inference_ms_history.append((time.perf_counter() - t0) * 1000.0)

        # --- Collect Internal Metrics ---
        if hasattr(model.policy, 'mlp_extractor') and hasattr(model.policy.mlp_extractor, 'last_cost_vectors'):
            raw_costs = model.policy.mlp_extractor.last_cost_vectors
            cost_params_history.append(_extract_cost_params_t0(raw_costs))
        else:
             cost_params_history.append(np.zeros(28, dtype=np.float32))

        current_gate_idx = env.target_gate_idx
        if current_gate_idx < len(env.gates):
            target_gate = env.gates[current_gate_idx]
            dist = np.linalg.norm(env.state[0:3].numpy() - target_gate.center)
            distances_history.append(dist)
        else:
            distances_history.append(0.0)

        pre_step_state = env.state.numpy().copy()
        pre_step_gate_idx = int(getattr(env, "target_gate_idx", 0))

        action_norm = action[0].astype(np.float32).copy()
        physical_u, _thrust_mass_norm, _omega_max = _denormalize_action(action_norm, env)

        obs, reward, done, info = venv.step(action)
        done = bool(done[0])
        step_info = info[0] if isinstance(info, (list, tuple)) else info
        if step_info is None:
            step_info = {}
        actions.append(action_norm)
        wind_active = bool(step_info.get("wind_active", False))
        wind_vec = step_info.get("wind_accel_world", None)
        if wind_vec is None and hasattr(env, "get_wind_vector_world"):
            try:
                wind_vec = env.get_wind_vector_world().tolist()
            except Exception:
                wind_vec = [0.0, 0.0, 0.0]
        wind_vec_np = np.asarray(wind_vec if wind_vec is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
        if wind_vec_np.size < 3:
            w = np.zeros(3, dtype=np.float32)
            w[: wind_vec_np.size] = wind_vec_np
            wind_vec_np = w
        wind_accel_norm = float(step_info.get("wind_accel_norm", float(np.linalg.norm(wind_vec_np))))
        wind_force_n = float(step_info.get("wind_force_n", wind_accel_norm * float(getattr(env.dx, "mass", 1.0))))
        wind_active_history.append(wind_active)
        wind_accel_norm_history.append(wind_accel_norm)
        wind_force_history.append(wind_force_n)

        if done:
            terminal_info = dict(step_info)
            terminal_event = terminal_info.get("event")
            if "target_gate_idx" in terminal_info:
                gates_passed = int(terminal_info["target_gate_idx"])
            elif terminal_event == "race_finished":
                gates_passed = len(env.gates)
            elif terminal_event == "gate_passed":
                gates_passed = min(len(env.gates), pre_step_gate_idx + 1)
            else:
                gates_passed = pre_step_gate_idx
            
            # Reconstruct terminal state
            try:
                x_prev = torch.from_numpy(pre_step_state.astype(np.float32)).unsqueeze(0)
                u_prev = torch.from_numpy(physical_u.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    x_term = env.dx.forward(x_prev, u_prev).squeeze(0).numpy().astype(np.float32)
                states.append(x_term)
            except Exception:
                states.append(pre_step_state.astype(np.float32))
        else:
            gates_passed = int(getattr(env, "target_gate_idx", gates_passed))
            states.append(env.state.numpy().copy())
    
    states = np.array(states)
    actions = np.array(actions)
    cost_params_history = np.array(cost_params_history)
    distances_history = np.array(distances_history)
    inference_ms_history = np.array(inference_ms_history, dtype=np.float32)
    wind_accel_norm_history = np.array(wind_accel_norm_history, dtype=np.float32)
    wind_force_history = np.array(wind_force_history, dtype=np.float32)
    wind_active_history = np.array(wind_active_history, dtype=bool)
    
    # --- Control Statistics Analysis ---
    print(f"\n--- Control Stats for {name} ---")
    u_norm = actions
    denorm = [_denormalize_action(a, env) for a in u_norm]
    if len(denorm) > 0:
        thrust_phys = np.array([item[0][0] for item in denorm], dtype=np.float32)
        rates_phys = np.stack([item[0][1:] for item in denorm], axis=0)
        omega_max = denorm[0][2]
    else:
        thrust_phys = np.zeros((0,), dtype=np.float32)
        rates_phys = np.zeros((0, 3), dtype=np.float32)
        omega_max = np.array([10.0, 10.0, 4.0], dtype=np.float32)
    u_phys = np.concatenate([thrust_phys[:, None], rates_phys], axis=1).astype(np.float32)
    
    print("Physical Actions (Reconstructed):")
    if len(thrust_phys) > 0:
        print(f"  Thrust (N)   : Min={thrust_phys.min():.3f}, Max={thrust_phys.max():.3f}")
        print(f"  Rates (rad/s): Min={rates_phys.min():.3f}, Max={rates_phys.max():.3f}")
    
    sat_thrust = np.mean(np.abs(u_norm[:,0]) > 0.99) * 100
    print(f"  Saturation (>99%): Thrust={sat_thrust:.1f}%")
    
    total_gates = len(env.gates)
    status = "UNKNOWN"
    if gates_passed >= total_gates:
        status = "SUCCESS (All gates passed)"
    elif terminal_event == "ground":
        status = "CRASH (Ground contact)"
    elif terminal_event == "miss":
        status = "MISS (Gate miss)"
    elif len(actions) >= env.max_steps:
        status = "TIMEOUT (Max steps reached)"
    elif done:
        status = f"TERMINATED EARLY ({terminal_event or 'unknown'})"
        
    print(f"  Result: {status}")
    print(f"  Gates Passed: {gates_passed} / {total_gates}")
    print(f"  Terminal Event: {terminal_event or 'none'}")
    print(f"  Steps Flown: {len(actions)} / {env.max_steps}")
    if wind_accel_norm_history.size > 0 and np.any(wind_accel_norm_history > 1e-6):
        wind_active_steps = int(np.sum(wind_active_history))
        wind_active_pct = 100.0 * float(wind_active_steps / max(1, wind_active_history.size))
        print(
            "  Wind Disturbance: "
            f"active_steps={wind_active_steps}/{wind_active_history.size} ({wind_active_pct:.1f}%), "
            f"|a| mean/max={float(np.mean(wind_accel_norm_history)):.2f}/{float(np.max(wind_accel_norm_history)):.2f} m/s^2, "
            f"|F| mean/max={float(np.mean(wind_force_history)):.2f}/{float(np.max(wind_force_history)):.2f} N"
        )
    print("--------------------------------\n")

    wind_zone_xy = None
    if hasattr(env, "get_wind_zone_xy_polygon"):
        try:
            wind_zone_xy = env.get_wind_zone_xy_polygon()
        except Exception:
            wind_zone_xy = None
    wind_vector_world = None
    wind_force_nominal_n = None
    if hasattr(env, "get_wind_vector_world"):
        try:
            wind_vector_world = env.get_wind_vector_world()
        except Exception:
            wind_vector_world = None
    if hasattr(env, "get_wind_strength_n"):
        try:
            wind_force_nominal_n = float(env.get_wind_strength_n())
        except Exception:
            wind_force_nominal_n = None
    wind_accel_nominal = None
    if wind_vector_world is not None:
        try:
            wind_accel_nominal = float(np.linalg.norm(np.asarray(wind_vector_world, dtype=np.float32)))
        except Exception:
            wind_accel_nominal = None

    # Generate Plots
    fig1 = plot_trajectory_3d(states, env.gates, title=f"Trajectory: {name}", color_by_speed=True)
    fig1.savefig(plot_dir / f"{name}_traj.png")
    plt.close(fig1)
    
    fig2 = plot_states(states, title=f"States: {name}")
    fig2.savefig(plot_dir / f"{name}_states.png")
    plt.close(fig2)
    
    fig3 = plot_controls(u_phys, title=f"Controls (Physical): {name}", normalized=False, omega_max=omega_max)
    fig3.savefig(plot_dir / f"{name}_controls.png")
    plt.close(fig3)

    fig4 = plot_trajectory_top_down(
        states,
        env.gates,
        title=f"Top-Down: {name}",
        color_by_speed=True,
        wind_zone_xy=wind_zone_xy,
        wind_accel_world=wind_vector_world,
        wind_force_n=wind_force_nominal_n,
        wind_accel_norm=wind_accel_nominal,
        wind_active_mask=wind_active_history,
    )
    fig4.savefig(plot_dir / f"{name}_top_down.png")
    plt.close(fig4)
    
    if len(cost_params_history) > 0:
        fig5 = plot_metrics(
            distances_history,
            cost_params_history,
            inference_ms=inference_ms_history,
            wind_accel_norm=wind_accel_norm_history,
            wind_force_n=wind_force_history,
            wind_active_mask=wind_active_history,
            title=f"Metrics: {name}",
        )
        fig5.savefig(plot_dir / f"{name}_metrics.png")
        plt.close(fig5)

    if save_video and states.shape[0] > 1:
        dt = float(getattr(getattr(env, "dx", None), "dt", 0.02))
        video_path = plot_dir / f"{name}_speed_direction.mp4"
        written = save_speed_direction_video_topdown(
            states=states,
            gates=env.gates,
            out_path=video_path,
            dt=dt,
            fps=int(video_fps),
            title=(video_title or name),
        )
        print(f"[Video] saved: {written}")

    venv.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--track-config", 
        type=str, 
        default=None,
        help="Optional YAML track config (e.g. tracks/my_track.yaml). If omitted, evaluates on splits and circle."
    )
    parser.add_argument("--log-dir", type=str, default="./runs/eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default="{}",
        help=(
            "JSON dict forwarded to GateRacingEnv (e.g. '{\"gate_pass_radius\":0.6}'). "
            "Eval-only keys: spawn_batch_plot (bool), spawn_batch_size (int, default 64). "
            "Wind keys: wind_accel_world [ax,ay,az], wind_active_steps, wind_zone_depth, wind_zone_width."
        ),
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Optional built-in track name (e.g. splits, circle, straight, straight_wind). If set, evaluates only this track.",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="acmpc_mlp",
        choices=["acmpc_mlp", "acmpc_diffmpc", "mlp_only"],
    )
    parser.add_argument(
        "--mpc-backend",
        type=str,
        default="auto",
        choices=["auto", "pytorch", "diffmpc", "fast", "both"],
    )
    parser.add_argument(
        "--max-steps-cap",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    
    evaluate_model(
        args.model_path,
        args.log_dir,
        track_config_path=args.track_config,
        device=args.device,
        policy_type=args.policy_type,
        mpc_backend=args.mpc_backend,
        single_track=args.track,
        env_kwargs=_parse_env_kwargs(args.env_kwargs),
        seed=args.seed,
        max_steps_cap=args.max_steps_cap,
    )
