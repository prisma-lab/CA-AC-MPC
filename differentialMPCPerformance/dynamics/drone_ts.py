import torch
import torch.nn as nn

class DroneDynamics(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.mass = 0.752
        self.g = torch.tensor([0.0, 0.0, -9.8066], device=device)
        self.inertia_diag = torch.tensor([0.0025, 0.0021, 0.0043], device=device)
        self.inertia_diag_inv = 1.0 / self.inertia_diag
        
        # Precomputed matrices for Jacobian calculation
        self._init_jacobian_matrices()
        
    def _init_jacobian_matrices(self):
        # Q_dot_dw
        self.aux_q_dot_dw_1 = torch.tensor([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], device=self.device, dtype=torch.float32)
        self.aux_q_dot_dw_2 = torch.tensor([[0, 0, -1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]], device=self.device, dtype=torch.float32)
        self.aux_q_dot_dw_3 = torch.tensor([[0, 0, 0, -1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], device=self.device, dtype=torch.float32)

        # d_q_dot_dq
        self.aux_d_q_dot_dq_1 = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device, dtype=torch.float32)
        self.aux_d_q_dot_dq_2 = torch.tensor([[-1, 0, 0], [0, 0, 0], [0, 0, -1], [0, 1, 0]], device=self.device, dtype=torch.float32)
        self.aux_d_q_dot_dq_3 = torch.tensor([[0, -1, 0], [0, 0, 1], [0, 0, 0], [-1, 0, 0]], device=self.device, dtype=torch.float32)
        self.aux_d_q_dot_dq_4 = torch.tensor([[0, 0, -1], [0, -1, 0], [1, 0, 0], [0, 0, 0]], device=self.device, dtype=torch.float32)

        # d_v_dot_d_q
        self.aux_d_v_dot_d_q_1 = torch.tensor([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], device=self.device, dtype=torch.float32)
        self.aux_d_v_dot_d_q_2 = torch.tensor([[0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]], device=self.device, dtype=torch.float32)
        self.aux_d_v_dot_d_q_3 = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], device=self.device, dtype=torch.float32)
        self.aux_d_v_dot_d_q_4 = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=self.device, dtype=torch.float32)

        # w_dot_dw
        i0, i1, i2 = self.inertia_diag[0], self.inertia_diag[1], self.inertia_diag[2]
        self.aux_w_dot_dw_1 = torch.tensor([[0, 0, 0], [0, 0, (i2 - i0)/i1], [0, (i0 - i1)/i2, 0]], device=self.device, dtype=torch.float32)
        self.aux_w_dot_dw_2 = torch.tensor([[0, 0, (i1 - i2)/i0], [0, 0, 0], [(i0 - i1)/i2, 0, 0]], device=self.device, dtype=torch.float32)
        self.aux_w_dot_dw_3 = torch.tensor([[0, (i1 - i2)/i0, 0], [(i2 - i0)/i1, 0, 0], [0, 0, 0]], device=self.device, dtype=torch.float32)

        # d_v_dot_du
        self.aux_dv_dot_du_1 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], device=self.device, dtype=torch.float32)
        self.aux_dv_dot_du_2 = torch.tensor([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], device=self.device, dtype=torch.float32)
        self.aux_dv_dot_du_3 = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=self.device, dtype=torch.float32)

    def quat_mult(self, q1, q2):
        # q1: [4, B], q2: [4, B] -> [4, B]
        ans = torch.stack([
            q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2] - q2[3] * q1[3],
            q2[0] * q1[1] + q2[1] * q1[0] - q2[2] * q1[3] + q2[3] * q1[2],
            q2[0] * q1[2] + q2[2] * q1[0] + q2[1] * q1[3] - q2[3] * q1[1],
            q2[0] * q1[3] - q2[1] * q1[2] + q2[2] * q1[1] + q2[3] * q1[0]
        ])
        return ans

    def exp_map(self, w, dt):
        # w: [3, B]
        w_norm = torch.norm(w, dim=0) + 1e-8
        w_normalized = w / w_norm
        
        # half_angle = w_norm * dt * 0.5
        s = torch.sin(w_norm * dt * 0.5)
        c = torch.cos(w_norm * dt * 0.5)
        
        return torch.stack([
            c,
            w_normalized[0] * s,
            w_normalized[1] * s,
            w_normalized[2] * s
        ])

    def rotate_quat(self, q, v):
        # q: [4, B] or [4], v: [3, B] or [3]
        if q.ndim == 1:
            q = q.unsqueeze(1)
        if v.ndim == 1:
            v = v.unsqueeze(1)
        # v_aug = [0, v]
        v_aug = torch.cat([torch.zeros(1, v.shape[1], device=q.device, dtype=q.dtype), v], dim=0)
        
        # q_inv = [q0, -q1, -q2, -q3]
        q_inv = torch.stack([q[0], -q[1], -q[2], -q[3]])
        
        # q * v_aug * q_inv
        tmp = self.quat_mult(q, v_aug)
        ans = self.quat_mult(tmp, q_inv)
        
        return ans[1:]

    def forward(self, x, u, dt=0.05):
        # x: [B, 10] or [10], u: [B, 4] or [4]
        squeeze = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        if u.ndim == 1:
            u = u.unsqueeze(0)
        device = x.device
        dtype = x.dtype
        # Transpose for easier indexing similar to original
        x_t = x.t()
        u_t = u.t()
        
        p = x_t[0:3]
        q = x_t[3:7]
        v = x_t[7:10]
        
        fc = u_t[0]
        w = u_t[1:4]
        
        # Position update
        new_p = p + dt * v
        
        # Quaternion update (exp map)
        em = self.exp_map(w, dt)
        new_q = self.quat_mult(q, em)
        new_q = new_q / (torch.norm(new_q, dim=0, keepdim=True) + 1e-8)
        
        # Velocity update thrust vector is [0, 0, fc] in body frame.
        thrust_body = torch.stack([torch.zeros_like(fc), torch.zeros_like(fc), fc])

        g_vec = self.g.to(device=device, dtype=dtype)
        acc = self.rotate_quat(q, thrust_body) / self.mass + g_vec.unsqueeze(1)
        new_v = v + dt * acc
        
        out = torch.cat([new_p, new_q, new_v]).t() # [B, 10]
        if squeeze:
            out = out.squeeze(0)
        return out

    def get_jacobian(self, x, u, dt=0.05):
        # Analytic Jacobian
        # x: [B, 10], u: [B, 4]
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        to_x = lambda t: t.to(device=device, dtype=dtype)
        
        q = x[:, 3:7].to(device=device, dtype=dtype)     # [B, 4]
        w = u[:, 1:4].to(device=device, dtype=dtype)     # [B, 3]
        fc = u[:, 0].to(device=device, dtype=dtype)      # [B]
        
        # --- R (State Jacobian) ---
        
        # d(q_dot)/d(q) = 0.5 * G(w)
        # G(w) uses aux_d_q_dot_dq matrices
        # [B, 4, 4]
        w_t = w.t() # [3, B]
        dqdot_dq = 0.5 * torch.stack([
            torch.matmul(to_x(self.aux_d_q_dot_dq_1), w_t).t(),
            torch.matmul(to_x(self.aux_d_q_dot_dq_2), w_t).t(),
            torch.matmul(to_x(self.aux_d_q_dot_dq_3), w_t).t(),
            torch.matmul(to_x(self.aux_d_q_dot_dq_4), w_t).t()
        ], dim=1)
        
        # d(v_dot)/d(q)
        # uses aux_d_v_dot_d_q matrices
        # qfcm = q * fc / m
        qfcm = (q * fc.unsqueeze(1) / self.mass).t() # [4, B]
        
        # aux @ qfcm -> [3, B]
        v1 = torch.matmul(to_x(self.aux_d_v_dot_d_q_1), qfcm)
        v2 = torch.matmul(to_x(self.aux_d_v_dot_d_q_2), qfcm)
        v3 = torch.matmul(to_x(self.aux_d_v_dot_d_q_3), qfcm)
        v4 = torch.matmul(to_x(self.aux_d_v_dot_d_q_4), qfcm)
        
        # Stack 4 vectors (columns) to form [3, 4, B] -> permute to [B, 3, 4]
        dvdot_dq = 2.0 * torch.stack([v1, v2, v3, v4], dim=0).permute(2, 1, 0)
        
        # Construct R [B, 10, 10]
        R = torch.eye(10, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        
        # Block (0:3, 7:10) -> p_dot depends on v
        # R[:, 0:3, 7:10] += dt * I
        R[:, 0:3, 7:10] = R[:, 0:3, 7:10] + dt * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        
        # Block (3:7, 3:7) -> q_dot depends on q
        # R[:, 3:7, 3:7] += dt * dqdot_dq
        R[:, 3:7, 3:7] = R[:, 3:7, 3:7] + dt * dqdot_dq
        
        # Block (7:10, 3:7) -> v_dot depends on q
        R[:, 7:10, 3:7] = dt * dvdot_dq
        
        # --- S (Control Jacobian) ---
        S = torch.zeros(B, 10, 4, device=device, dtype=dtype)
        
        # d(q_dot)/dw
        # uses aux_q_dot_dw matrices
        # [B, 4, 3]
        q_t = q.t() # [4, B]
        dqdot_dw = 0.5 * torch.stack([
            torch.matmul(to_x(self.aux_q_dot_dw_1), q_t).t(),
            torch.matmul(to_x(self.aux_q_dot_dw_2), q_t).t(),
            torch.matmul(to_x(self.aux_q_dot_dw_3), q_t).t()
        ], dim=2) # Note stack dim 2 because we want [B, 4, 3] from [B, 4] vectors
        
        S[:, 3:7, 1:4] = dt * dqdot_dw
        
        # d(v_dot)/dfc
        # uses aux_dv_dot_du matrices
        # tmp = q.T @ aux @ q
        # This is a bit complex to batch manually without loop or einsum
        # tmp1 = q @ aux1 @ q.T (roughly)
        
        # Batched quadratic form: q_i^T A q_i
        # einsum 'bi,ij,bj->b'
        val1 = torch.einsum('bi,ij,bj->b', q, to_x(self.aux_dv_dot_du_1), q)
        val2 = torch.einsum('bi,ij,bj->b', q, to_x(self.aux_dv_dot_du_2), q)
        val3 = torch.einsum('bi,ij,bj->b', q, to_x(self.aux_dv_dot_du_3), q)
        
        dvdot_dfc = torch.stack([val1, val2, val3], dim=1) / self.mass # [B, 3]
        
        S[:, 7:10, 0] = dt * dvdot_dfc
        
        return R, S

    def __call__(self, x, u, dt=0.05):
        return self.forward(x, u, dt)

    def get_traced_forward(self, example_batch_size=1):
        """Returns a JIT-traced version of the forward dynamics for faster execution.

        Usage:
            dynamics = DroneDynamics(device='cuda')
            traced_fwd = dynamics.get_traced_forward()
            x_next = traced_fwd(x, u, torch.tensor(0.05))  # dt must be a tensor!
        """
        # Create example inputs
        x_example = torch.zeros(example_batch_size, 10, device=self.device)
        x_example[:, 3] = 1.0  # valid quaternion
        u_example = torch.zeros(example_batch_size, 4, device=self.device)
        dt_example = torch.tensor(0.05, device=self.device)

        # Cache constants
        mass = self.mass
        g = self.g

        def forward_pure(x, u, dt_tensor):
            # x: [B, 10], u: [B, 4], dt_tensor: scalar tensor
            # Use tensor operations throughout for tracing compatibility

            # Extract components
            p = x[:, 0:3]
            q = x[:, 3:7]
            v = x[:, 7:10]

            fc = u[:, 0:1]
            w = u[:, 1:4]

            # Position update: p_new = p + dt * v
            new_p = p + dt_tensor * v

            # Quaternion update via exp map
            # exp(w*dt/2) = [cos(|w|*dt/2), sin(|w|*dt/2) * w/|w|]
            w_norm = torch.norm(w, dim=-1, keepdim=True) + 1e-8
            w_normalized = w / w_norm
            half_angle = w_norm * dt_tensor * 0.5
            s = torch.sin(half_angle)
            c = torch.cos(half_angle)
            em = torch.cat([c, w_normalized * s], dim=-1)

            # Quaternion multiplication: new_q = q * em
            # Original code uses convention: ans = q2 * q1 (second arg multiplies first)
            # So for quat_mult(q, em), result is em * q
            q1w, q1x, q1y, q1z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
            q2w, q2x, q2y, q2z = em[:, 0:1], em[:, 1:2], em[:, 2:3], em[:, 3:4]

            # Following exact formula from original quat_mult
            new_qw = q2w*q1w - q2x*q1x - q2y*q1y - q2z*q1z
            new_qx = q2w*q1x + q2x*q1w - q2y*q1z + q2z*q1y
            new_qy = q2w*q1y + q2y*q1w + q2x*q1z - q2z*q1x
            new_qz = q2w*q1z - q2x*q1y + q2y*q1x + q2z*q1w
            new_q = torch.cat([new_qw, new_qx, new_qy, new_qz], dim=-1)
            new_q = new_q / (torch.norm(new_q, dim=-1, keepdim=True) + 1e-8)

            # Velocity update
            # thrust_body = [0, 0, fc] in body frame
            # thrust_world = R(q) * thrust_body
            # Using quaternion rotation formula

            # For v = [0, 0, fc], the rotated vector is:
            # R(q) * [0,0,fc] where R is rotation matrix from quaternion
            qw_s = q[:, 0:1]
            qx_s = q[:, 1:2]
            qy_s = q[:, 2:3]
            qz_s = q[:, 3:4]

            # Rotation matrix applied to [0, 0, fc]
            # [R31, R32, R33] * fc = [2(qx*qz + qw*qy), 2(qy*qz - qw*qx), 1-2(qx^2+qy^2)] * fc
            rotated_x = 2 * (qx_s * qz_s + qw_s * qy_s) * fc
            rotated_y = 2 * (qy_s * qz_s - qw_s * qx_s) * fc
            rotated_z = (1 - 2 * qx_s * qx_s - 2 * qy_s * qy_s) * fc
            thrust_world = torch.cat([rotated_x, rotated_y, rotated_z], dim=-1)

            acc = thrust_world / mass + g.unsqueeze(0)
            new_v = v + dt_tensor * acc

            return torch.cat([new_p, new_q, new_v], dim=-1)

        # Trace and return
        traced = torch.jit.trace(forward_pure, (x_example, u_example, dt_example))
        return traced
