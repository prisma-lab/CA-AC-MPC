from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _as_states_array(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states)
    if arr.ndim != 2:
        return np.zeros((0, 10), dtype=np.float32)
    return arr


def _gate_corners_array(gates) -> np.ndarray:
    corners = [corner for g in gates for corner in g.corners()]
    if len(corners) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(corners, dtype=np.float32)


def _speed_from_states(states: np.ndarray) -> np.ndarray:
    if states.shape[1] < 10 or states.shape[0] == 0:
        return np.zeros((states.shape[0],), dtype=np.float32)
    vel = states[:, 7:10]
    return np.linalg.norm(vel, axis=1).astype(np.float32)


def _set_equal_3d_limits(ax, all_coords: np.ndarray) -> None:
    if all_coords.size == 0:
        return
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()

    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    max_range = max(max_range, 1e-3)
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _plot_gates_3d(ax, gates) -> None:
    for i, g in enumerate(gates):
        corners = g.corners()
        corners_plot = np.vstack([corners, corners[0]])
        ax.plot(corners_plot[:, 0], corners_plot[:, 1], corners_plot[:, 2], color="orange", linewidth=2, alpha=0.7)
        ax.scatter(g.center[0], g.center[1], g.center[2], color="orange", s=10)
        ax.text(g.center[0], g.center[1], g.center[2] + 0.5, f"G{i}", color="black")


def _plot_gates_topdown(ax, gates) -> None:
    for i, g in enumerate(gates):
        corners = g.corners()
        corners_plot = np.vstack([corners, corners[0]])
        ax.plot(corners_plot[:, 0], corners_plot[:, 1], color="orange", linewidth=2, alpha=0.7)
        ax.scatter(g.center[0], g.center[1], color="orange", s=20)
        ax.text(g.center[0], g.center[1], f"G{i}", color="black")

        nx, ny = g.normal[0], g.normal[1]
        ax.arrow(g.center[0], g.center[1], nx * 0.5, ny * 0.5, head_width=0.2, color="orange", alpha=0.5)


def _plot_wind_zone_topdown(
    ax,
    wind_zone_xy: np.ndarray | None = None,
    wind_accel_world: np.ndarray | None = None,
    wind_force_n: float | None = None,
    wind_accel_norm: float | None = None,
) -> None:
    if wind_zone_xy is None:
        return

    zone = np.asarray(wind_zone_xy, dtype=np.float32)
    if zone.ndim != 2 or zone.shape[0] < 3 or zone.shape[1] != 2:
        return

    poly = Polygon(
        zone,
        closed=True,
        facecolor="deepskyblue",
        edgecolor="deepskyblue",
        linewidth=1.8,
        alpha=0.15,
        zorder=1,
        label="Wind zone",
    )
    ax.add_patch(poly)

    center = zone.mean(axis=0)
    if wind_accel_world is None:
        return

    a = np.asarray(wind_accel_world, dtype=np.float32).reshape(-1)
    if a.size < 2:
        return
    a_xy = a[:2]
    a_xy_norm = float(np.linalg.norm(a_xy))
    if a_xy_norm < 1e-6:
        return

    extent = np.linalg.norm(zone.max(axis=0) - zone.min(axis=0))
    arrow_len = max(0.8, 0.22 * float(extent))
    u = (a_xy / a_xy_norm) * arrow_len
    ax.arrow(
        center[0],
        center[1],
        float(u[0]),
        float(u[1]),
        head_width=0.18,
        head_length=0.25,
        color="deepskyblue",
        alpha=0.9,
        length_includes_head=True,
        zorder=8,
    )

    a_norm_txt = float(a_xy_norm if wind_accel_norm is None else wind_accel_norm)
    if wind_force_n is None:
        txt = f"Wind |a|={a_norm_txt:.2f} m/s^2"
    else:
        txt = f"Wind |a|={a_norm_txt:.2f} m/s^2 |F|={float(wind_force_n):.2f} N"
    ax.text(
        center[0],
        center[1],
        txt,
        fontsize=9,
        color="deepskyblue",
        ha="center",
        va="bottom",
        zorder=9,
    )


def plot_trajectory_3d(
    states: np.ndarray,
    gates,
    title: str = "Drone Trajectory",
    color_by_speed: bool = True,
    cmap: str = "turbo",
):
    """
    Plot 3D trajectory and gates.
    If color_by_speed=True, the trajectory uses a speed colormap.
    """
    states = _as_states_array(states)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    path_coords = states[:, :3] if states.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
    speed = _speed_from_states(states)

    if states.shape[0] > 1:
        x, y, z = path_coords[:, 0], path_coords[:, 1], path_coords[:, 2]

        if color_by_speed:
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            vmin = float(np.min(speed))
            vmax = float(np.max(speed))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2.4)
            lc.set_array(speed[:-1])
            ax.add_collection3d(lc)
            cbar = fig.colorbar(lc, ax=ax, pad=0.12, fraction=0.03)
            cbar.set_label("Speed [m/s]")
        else:
            ax.plot(x, y, z, label="Drone Path", color="blue", linewidth=2)

        # Sparse velocity direction arrows in 3D
        vel = states[:, 7:10]
        step = max(1, states.shape[0] // 24)
        for i in range(0, states.shape[0], step):
            v = vel[i]
            vn = np.linalg.norm(v)
            if vn < 1e-6:
                continue
            u = (v / vn) * 0.8
            ax.quiver(
                x[i],
                y[i],
                z[i],
                u[0],
                u[1],
                u[2],
                color="black",
                linewidth=0.8,
                alpha=0.35,
                arrow_length_ratio=0.25,
            )

        ax.scatter(x[0], y[0], z[0], color="green", marker="o", label="Start")
        ax.scatter(x[-1], y[-1], z[-1], color="red", marker="x", label="End")

    all_coords = _gate_corners_array(gates)
    if path_coords.size > 0 and all_coords.size > 0:
        all_coords = np.concatenate([path_coords, all_coords], axis=0)
    elif path_coords.size > 0:
        all_coords = path_coords

    _set_equal_3d_limits(ax, all_coords)
    _plot_gates_3d(ax, gates)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig


def plot_states(states: np.ndarray, dt=0.02, title="States"):
    """
    Plot position, velocity, orientation.
    """
    states = np.asarray(states)
    t = np.arange(len(states)) * dt
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    if states.size == 0:
        for ax in axs:
            ax.text(0.5, 0.5, "No state samples", transform=ax.transAxes, ha="center", va="center")
            ax.grid(True)
        axs[2].set_xlabel("Time [s]")
        return fig

    # Position
    axs[0].plot(t, states[:, 0], label="x")
    axs[0].plot(t, states[:, 1], label="y")
    axs[0].plot(t, states[:, 2], label="z")
    axs[0].set_ylabel("Position [m]")
    axs[0].legend()
    axs[0].set_title(f"{title} - Position")
    axs[0].grid(True)

    # Velocity
    axs[1].plot(t, states[:, 7], label="vx")
    axs[1].plot(t, states[:, 8], label="vy")
    axs[1].plot(t, states[:, 9], label="vz")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].legend()
    axs[1].grid(True)

    # Quaternion
    axs[2].plot(t, states[:, 3], label="qw")
    axs[2].plot(t, states[:, 4], label="qx")
    axs[2].plot(t, states[:, 5], label="qy")
    axs[2].plot(t, states[:, 6], label="qz")
    axs[2].set_ylabel("Quaternion")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()
    axs[2].grid(True)

    return fig


def plot_controls(
    actions: np.ndarray,
    dt=0.02,
    title="Controls",
    normalized: bool = True,
    omega_max: np.ndarray | None = None,
):
    """
    Plot controls.
    actions: [T, 4] (thrust, w1, w2, w3)
      - normalized=True: actions in [-1, 1]
      - normalized=False: physical actions [N, rad/s, rad/s, rad/s]
    """
    actions = np.asarray(actions)
    t = np.arange(len(actions)) * dt
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if actions.size == 0:
        for ax in axs:
            ax.text(0.5, 0.5, "No control samples", transform=ax.transAxes, ha="center", va="center")
            ax.grid(True)
        axs[1].set_xlabel("Time [s]")
        return fig

    # Thrust
    axs[0].plot(t, actions[:, 0], label="Thrust", color="k")
    if normalized:
        axs[0].set_ylabel("Thrust [-1, 1]")
        axs[0].set_ylim(-1.05, 1.05)
    else:
        axs[0].set_ylabel("Thrust [N]")
    axs[0].set_title(f"{title} - Thrust")
    axs[0].grid(True)

    # Rates
    axs[1].plot(t, actions[:, 1], label="wx")
    axs[1].plot(t, actions[:, 2], label="wy")
    axs[1].plot(t, actions[:, 3], label="wz")
    if normalized:
        axs[1].set_ylabel("Body Rates (Norm) [-1, 1]")
        axs[1].set_ylim(-1.05, 1.05)
    else:
        axs[1].set_ylabel("Body Rates [rad/s]")
        if omega_max is not None:
            lim = float(np.max(np.abs(np.asarray(omega_max).reshape(-1))))
            axs[1].set_ylim(-1.05 * lim, 1.05 * lim)
    axs[1].set_xlabel("Time [s]")
    axs[1].legend()
    axs[1].grid(True)

    return fig


def plot_trajectory_top_down(
    states: np.ndarray,
    gates,
    title="Top-Down View (XY)",
    color_by_speed: bool = True,
    cmap: str = "turbo",
    wind_zone_xy: np.ndarray | None = None,
    wind_accel_world: np.ndarray | None = None,
    wind_force_n: float | None = None,
    wind_accel_norm: float | None = None,
    wind_active_mask: np.ndarray | None = None,
):
    """
    Plot top-down XY trajectory.
    If color_by_speed=True, the trajectory uses a speed colormap and velocity direction arrows.
    """
    states = _as_states_array(states)
    fig, ax = plt.subplots(figsize=(10, 10))

    if states.shape[0] > 1:
        xy = states[:, 0:2]
        speed = _speed_from_states(states)

        if color_by_speed:
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            vmin = float(np.min(speed))
            vmax = float(np.max(speed))
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-6
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.6)
            lc.set_array(speed[:-1])
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("Speed [m/s]")
        else:
            ax.plot(xy[:, 0], xy[:, 1], label="Drone Path", color="blue", linewidth=2)

        if wind_active_mask is not None:
            wind_mask = np.asarray(wind_active_mask).reshape(-1).astype(bool)
            if wind_mask.size == max(0, xy.shape[0] - 1) and np.any(wind_mask):
                segments = np.stack([xy[:-1], xy[1:]], axis=1)
                wind_segments = segments[wind_mask]
                if wind_segments.shape[0] > 0:
                    wl = LineCollection(
                        wind_segments,
                        colors="deepskyblue",
                        linewidths=4.2,
                        alpha=0.35,
                        zorder=4,
                        label="Wind-active path",
                    )
                    ax.add_collection(wl)

        # Sparse velocity direction arrows (top view)
        vel_xy = states[:, 7:9]
        step = max(1, states.shape[0] // 28)
        for i in range(0, states.shape[0], step):
            v = vel_xy[i]
            vn = np.linalg.norm(v)
            if vn < 1e-6:
                continue
            u = (v / vn) * 0.8
            ax.arrow(
                xy[i, 0],
                xy[i, 1],
                u[0],
                u[1],
                head_width=0.16,
                head_length=0.22,
                color="black",
                alpha=0.35,
                length_includes_head=True,
                zorder=5,
            )

        ax.scatter(xy[0, 0], xy[0, 1], color="green", marker="o", label="Start")
        ax.scatter(xy[-1, 0], xy[-1, 1], color="red", marker="x", label="End")

    _plot_gates_topdown(ax, gates)
    _plot_wind_zone_topdown(
        ax,
        wind_zone_xy=wind_zone_xy,
        wind_accel_world=wind_accel_world,
        wind_force_n=wind_force_n,
        wind_accel_norm=wind_accel_norm,
    )

    gate_xy = np.array([g.center[:2] for g in gates], dtype=np.float32) if len(gates) > 0 else np.zeros((0, 2), dtype=np.float32)
    path_xy = states[:, :2] if states.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    all_parts = []
    if path_xy.size > 0:
        all_parts.append(path_xy)
    if gate_xy.size > 0:
        all_parts.append(gate_xy)
    if wind_zone_xy is not None:
        zxy = np.asarray(wind_zone_xy, dtype=np.float32)
        if zxy.ndim == 2 and zxy.shape[1] == 2 and zxy.shape[0] > 0:
            all_parts.append(zxy)
    all_xy = np.concatenate(all_parts, axis=0) if len(all_parts) > 0 else np.zeros((0, 2), dtype=np.float32)
    if all_xy.size > 0:
        x_min, y_min = all_xy.min(axis=0)
        x_max, y_max = all_xy.max(axis=0)
        pad = max(1.0, 0.1 * max(x_max - x_min, y_max - y_min, 1.0))
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    return fig


def save_speed_direction_video_topdown(
    states: np.ndarray,
    gates,
    out_path: str | Path,
    dt: float = 0.02,
    fps: int = 30,
    title: str = "Eval rollout",
    cmap: str = "turbo",
    trail_max_points: int = 2500,
) -> str:
    """
    Save a top-down trajectory video with:
    - trajectory colored by speed,
    - current velocity vector,
    - speed colorbar.
    Returns the path of the generated video (or GIF fallback).
    """
    states = _as_states_array(states)
    if states.shape[0] < 2:
        raise ValueError("At least 2 states are needed to render a video.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xy = states[:, :2]
    vel = states[:, 7:9]
    speed = _speed_from_states(states)

    vmin = float(np.min(speed))
    vmax = float(np.max(speed))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_gates_topdown(ax, gates)

    gate_xy = np.array([g.center[:2] for g in gates], dtype=np.float32) if len(gates) > 0 else np.zeros((0, 2), dtype=np.float32)
    all_xy = np.concatenate([xy, gate_xy], axis=0) if gate_xy.size > 0 else xy
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    pad = max(1.0, 0.1 * max(x_max - x_min, y_max - y_min, 1.0))
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    lc = LineCollection([], cmap=cmap, norm=norm, linewidth=2.8)
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Speed [m/s]")

    point_artist = ax.scatter([xy[0, 0]], [xy[0, 1]], color="black", s=30, zorder=6)
    arrow_holder = {"artist": None}

    def _update_arrow(i: int) -> None:
        if arrow_holder["artist"] is not None:
            arrow_holder["artist"].remove()
            arrow_holder["artist"] = None

        v = vel[i]
        vn = np.linalg.norm(v)
        if vn < 1e-6:
            return

        arrow_len = 0.9
        u = (v / vn) * arrow_len
        arrow_holder["artist"] = ax.arrow(
            xy[i, 0],
            xy[i, 1],
            u[0],
            u[1],
            head_width=0.20,
            head_length=0.26,
            color="black",
            alpha=0.9,
            length_includes_head=True,
            zorder=7,
        )

    def _init():
        lc.set_segments([])
        lc.set_array(np.array([], dtype=np.float32))
        point_artist.set_offsets(np.array([[xy[0, 0], xy[0, 1]]], dtype=np.float32))
        _update_arrow(0)
        ax.set_title(f"{title} | t=0.00s | speed={speed[0]:.2f} m/s")
        return (lc, point_artist)

    def _update(frame_idx: int):
        frame_idx = int(frame_idx)
        start = max(0, frame_idx - int(trail_max_points))

        if frame_idx - start >= 1:
            pts = xy[start : frame_idx + 1]
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(speed[start:frame_idx])
        else:
            lc.set_segments([])
            lc.set_array(np.array([], dtype=np.float32))

        point_artist.set_offsets(np.array([[xy[frame_idx, 0], xy[frame_idx, 1]]], dtype=np.float32))
        _update_arrow(frame_idx)

        t_s = frame_idx * float(dt)
        ax.set_title(f"{title} | t={t_s:.2f}s | speed={speed[frame_idx]:.2f} m/s")
        return (lc, point_artist)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=np.arange(states.shape[0]),
        init_func=_init,
        interval=1000.0 / max(1, int(fps)),
        blit=False,
        repeat=False,
    )

    written_path = out_path
    try:
        writer = animation.FFMpegWriter(fps=max(1, int(fps)), bitrate=2200)
        ani.save(str(out_path), writer=writer)
    except Exception:
        # Fallback when ffmpeg is not available.
        written_path = out_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=max(1, min(int(fps), 20)))
        ani.save(str(written_path), writer=writer)

    plt.close(fig)
    return str(written_path)


def plot_metrics(
    distances: np.ndarray,
    cost_params: np.ndarray,
    dt=0.02,
    title="Metrics Evolution",
    inference_ms: np.ndarray | None = None,
    wind_accel_norm: np.ndarray | None = None,
    wind_force_n: np.ndarray | None = None,
    wind_active_mask: np.ndarray | None = None,
):
    """
    Plot distance to next gate and evolution of Q/R cost weights.
    cost_params: [T_steps, 28] (Taking only the first timestep of the horizon for plotting)
                 Indices 0-13 are Quadratic (Q), 14-27 are Linear (p).
                 Within Q (first 14):
                 0-2: Pos, 3-6: Quat, 7-9: Vel, 10-12: Omega(R), 13: Thrust(R)
    """
    distances = np.asarray(distances).reshape(-1)
    cost_params = np.asarray(cost_params)
    if cost_params.ndim == 1:
        cost_params = cost_params.reshape(1, -1)
    if cost_params.shape[1] < 28:
        padded = np.zeros((cost_params.shape[0], 28), dtype=np.float32)
        padded[:, : cost_params.shape[1]] = cost_params
        cost_params = padded
    elif cost_params.shape[1] > 28:
        cost_params = cost_params[:, :28]

    n = min(len(distances), len(cost_params))
    inference = None
    if inference_ms is not None:
        inference = np.asarray(inference_ms).reshape(-1)
        n = min(n, len(inference))

    wind_acc = None
    if wind_accel_norm is not None:
        wind_acc = np.asarray(wind_accel_norm).reshape(-1)
        n = min(n, len(wind_acc))
    wind_force = None
    if wind_force_n is not None:
        wind_force = np.asarray(wind_force_n).reshape(-1)
        n = min(n, len(wind_force))
    wind_mask = None
    if wind_active_mask is not None:
        wind_mask = np.asarray(wind_active_mask).reshape(-1).astype(bool)
        n = min(n, len(wind_mask))

    distances = distances[:n]
    cost_params = cost_params[:n]
    if inference is not None:
        inference = inference[:n]
    if wind_acc is not None:
        wind_acc = wind_acc[:n]
    if wind_force is not None:
        wind_force = wind_force[:n]
    if wind_mask is not None:
        wind_mask = wind_mask[:n]
    t = np.arange(n) * dt
    has_wind = wind_acc is not None or wind_force is not None
    n_rows = 3 + (1 if inference is not None else 0) + (1 if has_wind else 0)
    fig, axs = plt.subplots(n_rows, 1, figsize=(10, 3.6 * n_rows), sharex=True)

    if n == 0:
        for ax in np.atleast_1d(axs):
            ax.text(0.5, 0.5, "No metrics samples", transform=ax.transAxes, ha="center", va="center")
            ax.grid(True)
        axs[-1].set_xlabel("Time [s]")
        return fig

    axs[0].plot(t, distances, color="purple", label="Dist to Target Gate")
    axs[0].set_ylabel("Distance [m]")
    axs[0].set_title(f"{title} - Distance")
    axs[0].grid(True)
    axs[0].legend()

    q_weights = cost_params[:, :14]

    q_pos = np.mean(q_weights[:, 0:3], axis=1)
    q_att = np.mean(q_weights[:, 3:7], axis=1)
    q_vel = np.mean(q_weights[:, 7:10], axis=1)

    axs[1].plot(t, q_pos, label="Q Position")
    axs[1].plot(t, q_att, label="Q Attitude")
    axs[1].plot(t, q_vel, label="Q Velocity")
    axs[1].set_ylabel("Cost Weight (Q)")
    axs[1].set_title("Adaptive State Costs (Q)")
    axs[1].legend()
    axs[1].grid(True)

    q_omega = np.mean(q_weights[:, 10:13], axis=1)
    q_thrust = q_weights[:, 13]

    axs[2].plot(t, q_omega, label="R Rates (Omega)")
    axs[2].plot(t, q_thrust, label="R Thrust")
    axs[2].set_ylabel("Cost Weight (R)")
    axs[2].set_title("Adaptive Control Costs (R)")
    axs[2].legend()
    axs[2].grid(True)

    row_idx = 3
    if inference is not None:
        axs[row_idx].plot(t, inference, label="Policy Inference", color="tab:red")
        axs[row_idx].set_ylabel("Inference [ms]")
        axs[row_idx].set_title("Policy Inference Time")
        axs[row_idx].legend()
        axs[row_idx].grid(True)
        row_idx += 1

    if has_wind:
        wind_ax = axs[row_idx]
        if wind_acc is not None:
            wind_ax.plot(t, wind_acc, label="|a_wind| [m/s^2]", color="deepskyblue", linewidth=1.8)
        if wind_force is not None:
            wind_ax.plot(t, wind_force, label="|F_wind| [N]", color="tab:cyan", linestyle="--", linewidth=1.8)

        if wind_mask is not None and np.any(wind_mask):
            start = None
            for i, active in enumerate(wind_mask):
                if active and start is None:
                    start = i
                if (not active or i == len(wind_mask) - 1) and start is not None:
                    end = i if not active else i + 1
                    wind_ax.axvspan(start * dt, end * dt, color="deepskyblue", alpha=0.12, linewidth=0)
                    start = None

        wind_ax.set_ylabel("Wind")
        wind_ax.set_title("Wind Disturbance Intensity")
        wind_ax.legend()
        wind_ax.grid(True)

    axs[-1].set_xlabel("Time [s]")

    return fig
