import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import gym
import numpy as np
import torch as th

# Setup paths for vendored MPC dynamics.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "mpc.pytorch"))
sys.path.insert(0, str(ROOT / "acmpc_public-master" / "diff_mpc_drones"))

from drone import DroneDx
from envs.track_presets import make_straight_track, make_track
from utils.track_generator import load_track_from_yaml, Gate
import utils.math_utils as mu


class GateRacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        track_config_path: str | None = None,
        track: str = "splits",
        n_future_gates: int = 2,
        max_steps: int = 1000,
        loop_track_on_finish: bool = True,
        max_laps_per_episode: int | None = None,
        init_pos_cube: float = 1.0,
        spawn_distance: float = 1.0,
        spawn_plane_margin: float = 0.2,
        crash_z_min: float = 0.0,
        min_gate_z: float = 0.5,
        gate_pass_radius: float = 0.0,
        gate_pass_radius_requires_plane: bool = False,
        gate_thickness: float = 0.1,
        reward_body_rate_coeff: float = 0.01,
        reward_collision_penalty: float = 10.0,
        reward_gate_bonus: float = 10.0,
        obs_mode: str = "paper",
        # Parameters for internal generators
        straight_n_gates: int = 4,
        straight_start_x: float = 3.0,
        straight_gate_spacing: float = 2.0,
        straight_gate_y: float = 0.0,
        straight_gate_z: float = 1.0,
        wind_accel_world: List[float] | None = None,
        wind_active_steps: int = 0,
        wind_pre_gate_only: bool = True,
        wind_pre_gate_margin: float = 0.0,
        wind_zone_depth: float = 0.0,
        wind_zone_width: float = 0.0,
        random_n_gates: int = 8,
        random_xy_range: float = 8.0,
        random_z_range: float = 3.0,
        random_step_min: float = 1.0,
        random_step_max: float = 3.0,
        generative_n_gates: int = 10,
        generative_start_x: float = 4.0,
        generative_start_y: float = 0.0,
        generative_start_z: float = 1.5,
        generative_gate_size_m: float = 1.524,
        generative_size_decay: float = 0.95,
        generative_min_size_ratio: float = 0.5,
        generative_spacing_x: float = 2.0,
        generative_yz_jitter_start: float = 0.5,
        generative_yz_jitter_growth: float = 0.1,
        generative_yaw_start_deg: float = 0.0,
        generative_base_yaw_random_deg: float = 180.0,
        generative_yaw_step_deg: float = 5.0,
        generative_yaw_max_deg: float = 45.0,
        # Override Params
        override_spawn_pos: List[float] | None = None,
        override_spawn_quat: List[float] | None = None,
        override_spawn_yaw_deg: float | None = None,
        use_yaml_start_pos: bool = True,
        debug_info: bool = False,
    ):
        super().__init__()
        
        self.track_metadata = {}
        self.track_source = track

        self.straight_params = {
            'n_gates': straight_n_gates, 'start_x': straight_start_x,
            'spacing': straight_gate_spacing, 'y': straight_gate_y, 'z': straight_gate_z
        }
        if self.track_source == "straight_wind":
            if wind_accel_world is None:
                # Cross-wind style disturbance in world frame [ax, ay, az] m/s^2.
                wind_accel_world = [0.0, -5.0, 0.0]
            if int(wind_active_steps) <= 0:
                wind_active_steps = 80
            if float(wind_zone_depth) <= 0.0:
                wind_zone_depth = 3.0
            if float(wind_zone_width) <= 0.0:
                wind_zone_width = 3.5

        self.random_n_gates = random_n_gates
        self.random_xy_range = random_xy_range
        self.random_z_range = random_z_range
        self.random_step_min = random_step_min
        self.random_step_max = random_step_max
        
        self.generative_params = {
            'n': generative_n_gates, 'sx': generative_start_x, 'sy': generative_start_y, 'sz': generative_start_z,
            'size': generative_gate_size_m, 'decay': generative_size_decay, 'min_ratio': generative_min_size_ratio,
            'spacing': generative_spacing_x, 'jitter_start': generative_yz_jitter_start, 'jitter_growth': generative_yz_jitter_growth,
            'yaw_start': generative_yaw_start_deg, 'base_rnd': generative_base_yaw_random_deg, 
            'yaw_step': generative_yaw_step_deg, 'yaw_max': generative_yaw_max_deg
        }

        self.min_gate_z = float(min_gate_z)
        self.use_yaml_start_pos = bool(use_yaml_start_pos)
        
        if track_config_path:
            self.gates, self.track_metadata = load_track_from_yaml(track_config_path)
            self.current_track = self.track_metadata.get("track_id", "yaml_track")

            # Track YAML can provide the default spawn pose unless explicitly overridden.
            if self.use_yaml_start_pos and "start_pos" in self.track_metadata and override_spawn_pos is None:
                sp = self.track_metadata["start_pos"]
                self.override_spawn_pos = sp[0:3]
                self.override_spawn_yaw_deg = sp[3]
                self.override_spawn_quat = None
            else:
                self.override_spawn_pos = override_spawn_pos
                self.override_spawn_yaw_deg = override_spawn_yaw_deg
                self.override_spawn_quat = override_spawn_quat
        else:
            self.current_track = self.track_source
            self.override_spawn_pos = override_spawn_pos
            self.override_spawn_yaw_deg = override_spawn_yaw_deg
            self.override_spawn_quat = override_spawn_quat

            self._select_track()

        self.n_future_gates = n_future_gates
        self.max_steps = max_steps
        self.loop_track_on_finish = bool(loop_track_on_finish)
        self.max_laps_per_episode = None if max_laps_per_episode is None else int(max_laps_per_episode)
        if self.max_laps_per_episode is not None and self.max_laps_per_episode <= 0:
            raise ValueError("max_laps_per_episode must be > 0 when set.")
        
        self.init_pos_cube = init_pos_cube
        self.spawn_distance = spawn_distance
        self.spawn_plane_margin = float(spawn_plane_margin)
        self.crash_z_min = float(crash_z_min)
        self.gate_thickness = gate_thickness
        self.gate_pass_radius = float(gate_pass_radius)
        self.gate_pass_radius_requires_plane = bool(gate_pass_radius_requires_plane)
        self.reward_body_rate_coeff = float(reward_body_rate_coeff)
        self.reward_collision_penalty = float(reward_collision_penalty)
        self.reward_gate_bonus = float(reward_gate_bonus)
        self.obs_mode = obs_mode
        self.debug_info = bool(debug_info)

        self.device = "cpu"
        self.dx = DroneDx(device=self.device)

        # Action: normalized thrust/body rates
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = self._obs_dim()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.state = th.zeros(10, dtype=th.float32)
        self.prev_pos = np.zeros(3, dtype=np.float32)
        self.current_step = 0
        self.target_gate_idx = 0
        self.lap_count = 0
        self.last_gate_step = 0

        # Physical params
        normalization_max = 8.5 
        self.max_acc = (normalization_max * 4) / float(self.dx.mass)
        self.force_mean = 9.8066
        self.force_std = max(1e-6, self.max_acc - self.force_mean)
        self.omega_max = th.tensor([10.0, 10.0, 4.0], dtype=th.float32)
        self.wind_accel_world = th.tensor(
            np.asarray(wind_accel_world if wind_accel_world is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        )
        if self.wind_accel_world.shape != (3,):
            raise ValueError("wind_accel_world must be a 3-element vector [ax, ay, az].")
        self.wind_active_steps = max(0, int(wind_active_steps))
        self.wind_pre_gate_only = bool(wind_pre_gate_only)
        self.wind_pre_gate_margin = float(wind_pre_gate_margin)
        self.wind_zone_depth = max(0.0, float(wind_zone_depth))
        self.wind_zone_width = max(0.0, float(wind_zone_width))

    def _obs_dim(self) -> int:
        base_state = 10
        track_feat = self.n_future_gates * 12
        if self.obs_mode == "paper":
            quad_feat = 12  # v (3) + R (9)
            return base_state + quad_feat + track_feat
        return base_state + track_feat

    def seed(self, seed: int | None = None):
        if seed is not None:
            th.manual_seed(seed)
            np.random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
            
        if self.current_track in ["random", "generative"]:
            self._select_track()

        self.state.zero_()
        
        if self.override_spawn_pos is not None:
            self.state[0:3] = th.tensor(self.override_spawn_pos, dtype=th.float32)

            if self.override_spawn_quat is not None:
                self.state[3:7] = th.tensor(self.override_spawn_quat, dtype=th.float32)
            elif self.override_spawn_yaw_deg is not None:
                rad = math.radians(self.override_spawn_yaw_deg)
                self.state[3] = math.cos(rad / 2)  # w
                self.state[6] = math.sin(rad / 2)  # z
            else:
                self.state[3] = 1.0 # Identity (Default)

            self.target_gate_idx = 0

        else:
            num_gates = len(self.gates)
            if num_gates > 0:
                # For straight_wind we always spawn before gate 0, so the wind zone
                # is guaranteed to happen before the first gate.
                if self.current_track == "straight_wind":
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, num_gates)
                self.target_gate_idx = start_idx
                current_gate = self.gates[start_idx]

                gate_center = th.from_numpy(current_gate.center).float()
                gate_normal = th.from_numpy(current_gate.normal).float()

                gate_yaw = float(current_gate.pose[3])

                half_yaw = gate_yaw / 2.0
                self.state[3] = math.cos(half_yaw)
                self.state[4] = 0.0
                self.state[5] = 0.0
                self.state[6] = math.sin(half_yaw)

                spawn_center = gate_center - self.spawn_distance * gate_normal

                spawn_pos = spawn_center + (th.rand(3) - 0.5) * self.init_pos_cube

                if spawn_pos[2] < self.min_gate_z:
                    spawn_pos[2] = self.min_gate_z

                # Keep the spawn point before the gate plane.
                d = th.dot(spawn_pos - gate_center, gate_normal)
                if d > -self.spawn_plane_margin:
                    spawn_pos = spawn_pos - (d + self.spawn_plane_margin) * gate_normal

                self.state[0:3] = spawn_pos
            else:
                self.state[0:3] = th.zeros(3)
                self.state[2] = 1.0
                self.state[3] = 1.0

        self.prev_pos = self.state[0:3].numpy().copy()
        self.current_step = 0
        self.lap_count = 0
        self.last_gate_step = 0

        return self._get_obs()

    def step(self, action):
        u = th.from_numpy(action).float()
        thrust_mass_norm = u[0] * self.force_std + self.force_mean
        thrust_mass_norm = th.clamp(thrust_mass_norm, 0.0, float(self.max_acc))
        thrust = thrust_mass_norm * self.dx.mass
        omegas = u[1:] * self.omega_max
        physical_u = th.cat([thrust.view(1), omegas], dim=0)

        with th.no_grad():
            x_next = self.dx.forward(self.state.unsqueeze(0), physical_u.unsqueeze(0))
        self.state = x_next.squeeze(0)
        wind_active, wind_accel = self._apply_wind_disturbance()

        pos = self.state[0:3].numpy().copy()
        info = {
            "track": self.current_track,
            "lap_count": int(self.lap_count),
            "target_gate_idx": int(self.target_gate_idx),
            "num_gates": int(len(self.gates)),
            "wind_active": bool(wind_active),
            "wind_accel_world": [float(v) for v in wind_accel.tolist()],
            "wind_accel_norm": float(np.linalg.norm(wind_accel)),
            "wind_force_n": float(np.linalg.norm(wind_accel) * float(self.dx.mass)),
            "wind_zone_depth": float(self.wind_zone_depth),
            "wind_zone_width": float(self.wind_zone_width),
        }

        # Diagnostics
        gate = self.gates[self.target_gate_idx]
        prev_dist = float(np.linalg.norm(gate.center - self.prev_pos))
        curr_dist = float(np.linalg.norm(gate.center - pos))
        progress = prev_dist - curr_dist
        plane_dist = float(abs(np.dot(pos - gate.center, gate.normal)))
        crossed, in_gate = self._check_gate_cross(self.prev_pos, pos, gate)
        
        within_radius = False
        if self.gate_pass_radius > 0.0:
            if self.gate_pass_radius_requires_plane:
                within_radius = (curr_dist <= self.gate_pass_radius) and (plane_dist <= self.gate_thickness)
            else:
                within_radius = curr_dist <= self.gate_pass_radius

        if pos[2] <= self.crash_z_min:
            info.update({"crash": "ground", "event": "ground"})
            if self.debug_info:
                self._add_debug_info(info, curr_dist, progress, plane_dist, crossed, in_gate, within_radius, omegas, pos, action, thrust_mass_norm)
            info["target_gate_idx"] = int(self.target_gate_idx)
            info["num_gates"] = int(len(self.gates))
            return self._get_obs(), -self.reward_collision_penalty, True, info

        reward, done = self._compute_reward(pos, omegas.numpy())

        lap_finished = False
        race_finished = False
        
        if done and reward > 0:
            if self.target_gate_idx >= len(self.gates):
                if self.loop_track_on_finish:
                    self.lap_count += 1
                    reached_max_laps = (self.max_laps_per_episode is not None and self.lap_count >= self.max_laps_per_episode)
                    
                    if reached_max_laps:
                        race_finished = True
                    else:
                        done = False
                        self.target_gate_idx = 0
                        lap_finished = True
                else:
                    race_finished = True
        self.prev_pos = pos
        self.current_step += 1
        did_timeout = False
        
        if self.current_step >= self.max_steps:
            done = True
            did_timeout = True

        if done and not did_timeout and reward <= -self.reward_collision_penalty:
            info["event"] = "miss"
        elif lap_finished:
            info["event"] = "lap_finished"
        elif race_finished:
            info["event"] = "race_finished"
        elif reward > 1.0 and not done:
            info["event"] = "gate_passed"
        elif reward > 1.0 and done:
            info["event"] = "gate_passed_at_timeout"
        elif done and did_timeout:
            info["event"] = "timeout"

        if self.debug_info:
            self._add_debug_info(info, curr_dist, progress, plane_dist, crossed, in_gate, within_radius, omegas, pos, action, thrust_mass_norm)

        info["target_gate_idx"] = int(self.target_gate_idx)
        info["num_gates"] = int(len(self.gates))

        return self._get_obs(), reward, done, info

    def _compute_reward(self, pos: np.ndarray, omegas: np.ndarray) -> Tuple[float, bool]:
        done = False
        gate = self.gates[self.target_gate_idx]
        gate_passed = False

        if self.gate_pass_radius > 0.0:
            dist = float(np.linalg.norm(gate.center - pos))
            ok = False
            if self.gate_pass_radius_requires_plane:
                plane_dist = float(abs(np.dot(pos - gate.center, gate.normal)))
                ok = (dist <= self.gate_pass_radius) and (plane_dist <= self.gate_thickness)
            else:
                ok = dist <= self.gate_pass_radius
            if ok:
                gate_passed = True

        if not gate_passed:
            crossed, in_gate = self._check_gate_cross(self.prev_pos, pos, gate)
            if crossed:
                if not in_gate:
                    return -self.reward_collision_penalty, True
                gate_passed = True

        if gate_passed:
            self.last_gate_step = self.current_step
            self.target_gate_idx += 1
            if self.target_gate_idx >= len(self.gates):
                return self.reward_gate_bonus, True
            return self.reward_gate_bonus, False

        gk = gate.center
        prev_dist = np.linalg.norm(gk - self.prev_pos)
        curr_dist = np.linalg.norm(gk - pos)
        progress = prev_dist - curr_dist
        body_rate_penalty = self.reward_body_rate_coeff * np.linalg.norm(omegas)

        return float(progress - body_rate_penalty), done

    def _check_gate_cross(self, p0: np.ndarray, p1: np.ndarray, gate: Gate) -> Tuple[bool, bool]:
        n = gate.normal
        c = gate.center
        
        d0 = np.dot(p0 - c, n)
        d1 = np.dot(p1 - c, n)

        # Passage is valid when the drone crosses along the gate normal.
        if not (d0 < 0 and d1 >= 0):
            return False, False

        if abs(d0 - d1) < 1e-8:
            return False, False

        t = d0 / (d0 - d1)
        t = np.clip(t, 0.0, 1.0)
        
        p_int = p0 + t * (p1 - p0)
        rel = p_int - c
        right = gate.R[:, 0]
        up = gate.R[:, 1]        
        x_local = np.dot(rel, right)
        y_local = np.dot(rel, up)
        
        in_gate = (abs(x_local) <= gate.width * 0.5) and (abs(y_local) <= gate.height * 0.5)
        
        return True, in_gate

    def _get_obs(self) -> np.ndarray:
        state_np = self.state.numpy()
        pos = state_np[0:3]
        quat = state_np[3:7]
        vel = state_np[7:10]

        track_feat = []
        for i in range(self.n_future_gates):
            idx = self.target_gate_idx + i
            if self.loop_track_on_finish and idx >= len(self.gates):
                 idx = idx % len(self.gates)
            
            if idx < len(self.gates):
                corners = self.gates[idx].corners()
                rel = corners - pos.reshape(1, 3)
                track_feat.append(rel.reshape(-1))
            else:
                track_feat.append(np.zeros(12, dtype=np.float32))
        
        if len(track_feat) > 0:
            track_feat = np.concatenate(track_feat, axis=0)
        else:
            track_feat = np.array([], dtype=np.float32)

        if self.obs_mode == "paper":
            R = mu.quat_to_rot(quat)
            obs = np.concatenate([state_np, vel, R.reshape(-1), track_feat], axis=0)
        else:
            obs = np.concatenate([state_np, track_feat], axis=0)
        return obs.astype(np.float32)

    def _is_wind_active_now(self) -> bool:
        if self.wind_active_steps <= 0:
            return False
        if self.current_step >= self.wind_active_steps:
            return False
        if float(th.norm(self.wind_accel_world).item()) <= 1e-8:
            return False
        if not self.wind_pre_gate_only:
            return True
        if self.target_gate_idx != 0 or len(self.gates) == 0:
            return False

        gate0 = self.gates[0]
        pos = self.state[0:3].numpy()
        rel = pos - gate0.center
        signed_plane_dist = float(np.dot(rel, gate0.normal))
        if signed_plane_dist >= self.wind_pre_gate_margin:
            return False
        if self.wind_zone_depth > 0.0 and signed_plane_dist < (self.wind_pre_gate_margin - self.wind_zone_depth):
            return False
        if self.wind_zone_width > 0.0:
            right_axis = gate0.R[:, 0]
            lateral = float(abs(np.dot(rel, right_axis)))
            if lateral > self.wind_zone_width * 0.5:
                return False
        return True

    def _apply_wind_disturbance(self) -> Tuple[bool, np.ndarray]:
        if not self._is_wind_active_now():
            return False, np.zeros(3, dtype=np.float32)

        dt = float(getattr(self.dx, "dt", 0.02))
        a = self.wind_accel_world.to(dtype=th.float32)

        # Integrate constant acceleration in world frame for one env step.
        self.state[7:10] = self.state[7:10] + a * dt
        self.state[0:3] = self.state[0:3] + 0.5 * a * (dt * dt)
        return True, a.detach().cpu().numpy().astype(np.float32)

    def get_wind_vector_world(self) -> np.ndarray:
        return self.wind_accel_world.detach().cpu().numpy().astype(np.float32)

    def get_wind_strength_n(self) -> float:
        return float(np.linalg.norm(self.get_wind_vector_world()) * float(self.dx.mass))

    def get_wind_zone_xy_polygon(self) -> Optional[np.ndarray]:
        if len(self.gates) == 0:
            return None
        if self.wind_zone_width <= 0.0:
            return None

        gate0 = self.gates[0]
        center_xy = gate0.center[:2].astype(np.float32)
        normal_xy = gate0.normal[:2].astype(np.float32)
        right_xy = gate0.R[:, 0][:2].astype(np.float32)

        norm_n = float(np.linalg.norm(normal_xy))
        norm_r = float(np.linalg.norm(right_xy))
        if norm_n < 1e-6 or norm_r < 1e-6:
            return None

        normal_xy = normal_xy / norm_n
        right_xy = right_xy / norm_r

        depth = self.wind_zone_depth if self.wind_zone_depth > 0.0 else max(1.0, self.spawn_distance + self.init_pos_cube)
        front_d = float(self.wind_pre_gate_margin)
        back_d = front_d - float(depth)
        half_w = 0.5 * float(self.wind_zone_width)

        front_center = center_xy + front_d * normal_xy
        back_center = center_xy + back_d * normal_xy
        polygon = np.stack(
            [
                back_center - half_w * right_xy,
                back_center + half_w * right_xy,
                front_center + half_w * right_xy,
                front_center - half_w * right_xy,
            ],
            axis=0,
        ).astype(np.float32)
        return polygon

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _select_track(self) -> None:
        if self.track_source == "random":
            self.gates = self._make_random_track()
            self._clamp_gate_heights(self.gates)
        elif self.track_source == "generative":
            self.gates = self._make_generative_track()
            self._clamp_gate_heights(self.gates)
        elif self.track_source in ["straight", "straight_wind"]:
             p = self.straight_params
             gates = make_straight_track(p['n_gates'], p['start_x'], p['spacing'], p['y'], p['z'])
             self._clamp_gate_heights(gates)
             self.gates = gates
        elif self.track_source in ["splits", "circle", "horizontal", "vertical"]:
            gates = make_track(self.track_source)
            self._clamp_gate_heights(gates)
            self.gates = gates
    def _z_floor(self) -> float:
        return max(float(self.min_gate_z), 1e-3)

    def _enforce_positive_z(self, z: float) -> float:
        z = float(z)
        if z <= 0.0:
            z = abs(z)
        return max(z, self._z_floor())

    def _clamp_gate_heights(self, gates: List[Gate]) -> None:
        for g in gates:
            if g.pose[2] < self._z_floor():
                 g.pose[2] = self._enforce_positive_z(g.pose[2])
                 g.__post_init__()

    def _make_random_track(self) -> List[Gate]:
        gates: List[Gate] = []
        x, y, z = 4.0, 0.0, 1.0

        poses = [[x,y,z,0.0]] # x,y,z,yaw
        
        for _ in range(self.random_n_gates - 1):
            step = np.random.uniform(self.random_step_min, self.random_step_max)
            yaw = np.random.uniform(-math.pi, math.pi)
            dz = np.random.uniform(-self.random_z_range, self.random_z_range)
            
            prev_x, prev_y, prev_z, _ = poses[-1]
            
            next_x = prev_x + step * math.cos(yaw)
            next_y = prev_y + step * math.sin(yaw)
            next_z = prev_z + dz
            
            next_x = np.clip(next_x, -self.random_xy_range, self.random_xy_range)
            next_y = np.clip(next_y, -self.random_xy_range, self.random_xy_range)
            next_z = np.clip(next_z, self._z_floor(), 1.0 + self.random_z_range)
            
            poses.append([next_x, next_y, next_z, 0.0]) # Yaw placeholder

        for i in range(len(poses)):
            cx, cy, cz, _ = poses[i]
            if i < len(poses) - 1:
                nx, ny, nz, _ = poses[i+1]
                dx, dy = nx - cx, ny - cy
            else:
                px, py, pz, _ = poses[i-1]
                dx, dy = cx - px, cy - py
            
            yaw = math.atan2(dy, dx)
            pose = np.array([cx, cy, cz, yaw], dtype=np.float32)
            gates.append(Gate(width=1.0, height=1.0, pose=pose))
            
        return gates

    def _make_generative_track(self) -> List[Gate]:
        gates: List[Gate] = []
        p = self.generative_params
        
        curr_x, curr_y, curr_z = p['sx'], p['sy'], p['sz']
        curr_z = self._enforce_positive_z(curr_z)
        
        base_yaw_deg = p['yaw_start']
        if p['base_rnd'] > 0.0:
            base_yaw_deg += np.random.uniform(-p['base_rnd'], p['base_rnd'])
            
        prev_yaw_rad = 0.0

        for i in range(p['n']):
            yaw_bound = min(p['yaw_max'], p['yaw_step'] * float(i))
            yaw_offset = np.random.uniform(-yaw_bound, yaw_bound) if yaw_bound > 0.0 else 0.0
            yaw_deg = base_yaw_deg + yaw_offset
            yaw_rad = math.radians(yaw_deg)

            if i > 0:
                # Advance along previous direction
                curr_x += p['spacing'] * math.cos(prev_yaw_rad)
                curr_y += p['spacing'] * math.sin(prev_yaw_rad)
                
                # Jitter
                amp = p['jitter_start'] + p['jitter_growth'] * float(i - 1)
                curr_y += np.random.uniform(-amp, amp)
                curr_z += np.random.uniform(-amp, amp)
                curr_z = self._enforce_positive_z(curr_z)

            ratio = max(p['min_ratio'], p['decay'] ** float(i))
            size = p['size'] * ratio
            
            pose = np.array([curr_x, curr_y, curr_z, yaw_rad], dtype=np.float32)
            gates.append(Gate(width=size, height=size, pose=pose))
            
            prev_yaw_rad = yaw_rad

        return gates

    def _add_debug_info(self, info, curr_dist, progress, plane_dist, crossed, in_gate, within_radius, omegas, pos, action, thrust_mass_norm):
        info.update({
            "target_gate_idx": int(self.target_gate_idx),
            "lap_count": int(self.lap_count),
            "dist_to_gate": curr_dist,
            "progress": progress,
            "plane_dist": plane_dist,
            "crossed_plane": bool(crossed),
            "in_gate_frame": bool(in_gate),
            "within_radius": bool(within_radius),
            "omega_norm": float(np.linalg.norm(omegas.numpy())),
            "z": float(pos[2]),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "u_norm": [float(x) for x in action.tolist()],
            "thrust_mass_norm": float(thrust_mass_norm.item()),
            "omegas_cmd": [float(x) for x in omegas.numpy().tolist()],
        })
