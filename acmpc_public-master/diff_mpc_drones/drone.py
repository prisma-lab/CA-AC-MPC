import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

def exp_map(w, delta_t):
    w_norm = torch.norm(w, dim=0)
    w_normalized = torch.nn.functional.normalize(w, dim=0)
    ans  = torch.stack([torch.cos(torch.mul(w_norm, delta_t * 0.5)),
                        w_normalized[0,:] * torch.sin(torch.mul(w_norm, delta_t * 0.5)),
                        w_normalized[1,:] * torch.sin(torch.mul(w_norm, delta_t * 0.5)),
                        w_normalized[2,:] * torch.sin(torch.mul(w_norm, delta_t * 0.5)),
                        ])
    return ans

def quat_mult(q1,q2):
    ans = torch.stack([q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
                     q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
                     q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
                     q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:]])
    return ans

# def quat_mult(q1, q2):
#     ans = quat2_lmatrix(q1) * q2
#     return ans

def quat2_rmatrix(q):
    return torch.tensor([[q[0], -q[1], -q[2], -q[3]],
                         [q[1], q[0], q[3], -q[2]],
                         [q[2], -q[3], q[0], q[1]],
                         [q[3], q[2], -q[1], q[0]]
                         ])

def quat2_lmatrix(q):
    return torch.tensor([[q[0], -q[1], -q[2], -q[3]],
                         [q[1], q[0], -q[3], q[2]],
                         [q[2], q[3], q[0], -q[1]],
                         [q[3], -q[2], q[1], q[0]]
                         ])

def jac_d_w_dot_du(lx, ly, k, inertia_diag_inv):
    return torch.Tensor([[ly[0]*inertia_diag_inv[0], ly[1]*inertia_diag_inv[0], ly[2]*inertia_diag_inv[0], ly[3]*inertia_diag_inv[0]],
                         [-lx[0]*inertia_diag_inv[1], -lx[1]*inertia_diag_inv[1], -lx[2]*inertia_diag_inv[1], -lx[3]*inertia_diag_inv[1]],
                         [-k*inertia_diag_inv[2], -k*inertia_diag_inv[2], k*inertia_diag_inv[2], k*inertia_diag_inv[2]]])


    
    

    # tmp[0] = (2*q[0]*q[2] + 2*q[1]*q[3])/m
    # tmp[1] = -(2*q[0]*q[1] - 2*q[2]*q[3])/m
    # tmp[2] = (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)/m

    return tmp.repeat(1, 4)


def jac_w_dot_dw(omega, inertia_diag):
    # ans = torch.zeros(3,3)

    ans = torch.Tensor([[0.0, (inertia_diag[1] - inertia_diag[2])/inertia_diag[0] * omega[2], (inertia_diag[1] - inertia_diag[2])/inertia_diag[0] * omega[1]],
                        [(inertia_diag[2] - inertia_diag[0])/inertia_diag[1] * omega[2], 0.0, (inertia_diag[2] - inertia_diag[0])/inertia_diag[1] * omega[0]],
                        [(inertia_diag[0] - inertia_diag[1])/inertia_diag[2] * omega[1], (inertia_diag[0] - inertia_diag[1])/inertia_diag[2] * omega[0], 0.0]])
    # first row
    # ans[0,1] = (inertia_diag[1] - inertia_diag[2]) * omega[2]
    # ans[0,2] = (inertia_diag[1] - inertia_diag[2]) * omega[1]
    # # second row
    # ans[1,0] = (inertia_diag[2] - inertia_diag[0]) * omega[2]
    # ans[1,2] = (inertia_diag[2] - inertia_diag[0]) * omega[0]

    # # third row
    # ans[2,0] = (inertia_diag[0] - inertia_diag[1]) * omega[1]
    # ans[2,1] = (inertia_diag[0] - inertia_diag[1]) * omega[0]

    return ans



def jac_q_dot_dw(q):
    # ans = torch.zeros(4, 3)

    ans = 0.5*torch.tensor([[-q[1], -q[2], -q[3]],
                            [q[0], -q[3], -q[2]],
                            [q[3], q[0], -q[1]],
                            [-q[2], q[1], q[0]]])

    return ans

def jac_d_v_dot_dq(q, f_c, m):

    q0fcm = (q[0]*f_c)/m
    q1fcm = (q[1]*f_c)/m
    q2fcm = (q[2]*f_c)/m
    q3fcm = (q[3]*f_c)/m
    
    ans = 2 * torch.tensor([[q2fcm, q3fcm, q0fcm, q1fcm],
                            [-q1fcm, -q0fcm, q3fcm, q2fcm],
                            [q0fcm, -q1fcm, -q2fcm, q3fcm]])
    # first row
    # ans[0,0] = (2*q[2]*f_c)/m
    # ans[0,1] = (2*q[3]*f_c)/m
    # ans[0,2] = (2*q[0]*f_c)/m
    # ans[0,3] = (2*q[1]*f_c)/m

    # # second row
    # ans[1,0] = (-2*q[1]*f_c)/m
    # ans[1,1] = (-2*q[0]*f_c)/m
    # ans[1,2] = (2*q[3]*f_c)/m
    # ans[1,3] = (2*q[2]*f_c)/m

    # # third row
    # ans[2,0] = (2*q[0]*f_c)/m
    # ans[2,1] = (-2*q[1]*f_c)/m
    # ans[2,2] = (-2*q[2]*f_c)/m
    # ans[2,3] = (2*q[3]*f_c)/m

    return ans


def quat_error(q, q_ref):
    q_aux = torch.stack([q[0,:] * q_ref[0,:] + q[1,:] * q_ref[1,:] + q[2,:] * q_ref[2,:] + q[3,:] * q_ref[3,:],
                         -q[1,:] * q_ref[0,:] + q[0,:] * q_ref[1,:] + q[3,:] * q_ref[2,:] - q[2,:] * q_ref[3,:],
                         -q[2,:] * q_ref[0,:] - q[3,:] * q_ref[1,:] + q[0,:] * q_ref[2,:] + q[1,:] * q_ref[3,:],
                         -q[3,:] * q_ref[0,:] + q[2,:] * q_ref[1,:] - q[1,:] * q_ref[2,:] + q[0,:] * q_ref[3,:]])
    # attitude errors. SQRT have small quantities added (1e-3) to alleviate the derivative
    # not being defined at zero, and also because it's in the denominator
    q_att_denom = torch.sqrt(q_aux[0] * q_aux[0] + q_aux[3] * q_aux[3] + 1e-3)
    q_att = torch.stack([
        q_aux[0] * q_aux[1] - q_aux[2] * q_aux[3],
        q_aux[0] * q_aux[2] + q_aux[1] * q_aux[3],
        q_aux[3]]) / q_att_denom
    return q_att

def rotate_quat(q1,v1):
    if q1.is_cuda:
        this_device = "cuda:0"
    else:
        this_device = "cpu"

    n_batch = q1.shape[1]
    
    ans = quat_mult(quat_mult(q1, torch.cat([torch.full((1, n_batch), 0, device=this_device), v1])),
                    torch.stack([q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]]))

    return torch.stack([ans[1,:], ans[2,:], ans[3,:]])

class DroneDx(nn.Module):
    def __init__(self, params=None, device='cpu'):
        super().__init__()

        self.params = Variable(torch.tensor([]))
        self.this_device = device


        self.mass = 0.752       # kg
        self.dt = 0.02          # s
        self.l_x = torch.tensor([0.075, -0.075, -0.075, 0.075])
        self.l_y = torch.tensor([-0.10, 0.10, -0.10, 0.10])
        self.kappa = 0.022
        self.inertia_diag = torch.tensor([0.0025, 0.0021, 0.0043])
        self.inertia_diag_inv = torch.div(torch.tensor([1.0, 1.0, 1.0]), self.inertia_diag)
        self.inertia = torch.diag(self.inertia_diag)
        self.inertia_inv = torch.diag(self.inertia_diag_inv)
        self.motor_tau = 0.033

        self.omega_max = torch.tensor([10.0, 10.0, 4.0])
        self.thrust_min = 0.0
        self.thrust_max = 8.5

        self.n_state = 10
        self.n_ctrl = 4
        self.POS_IDX = slice(0,3)
        self.QUAT_IDX = slice(3,7)
        self.VEL_IDX = slice(7,10)

        self.g = torch.tensor([0, 0, -9.8066])

        self.goal_tau = torch.Tensor([0.0, 0.0, 0.0,
                                      0.7071, 0.0, 0.0, 0.7071,
                                      0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0,
                                      -self.g[2]*self.mass/4.0,-self.g[2]*self.mass/4.0, -self.g[2]*self.mass/4.0, -self.g[2]*self.mass/4.0 ])

        self.goal_weights = torch.Tensor([0.6, 0.6, 0.9,
                                          0.2, 0.2, 0.2, 0.2,
                                          0.6, 0.6, 0.6,
                                          0.1, 0.1, 0.1,
                                          0.3, 0.3, 0.3, 0.3])

        self.mpc_eps = 1e-1
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        # Declaration of vmaps
        # State jacobian
        self.vect_jac_q_dot_dw = torch.func.vmap(self.jac_q_dot_dw_mat)
        self.vect_jac_q_dot_dq = torch.func.vmap(self.jac_q_dot_dq_mat)
        self.vect_jac_d_v_dot_dq = torch.func.vmap(self.jac_d_v_dot_dq_mat)
        self.vect_jac_w_dot_dw = torch.func.vmap(self.jac_w_dot_dw_mat)


        # For CTBR dynamics
        self.vect_jac_d_v_dot_df = torch.func.vmap(self.jac_d_v_dot_df_mat)


        # Allocation matrices

        # Q_dot_dw
        self.aux_q_dot_dw_1 = torch.Tensor([[0, -1, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 0, 1],
                                         [0, 0, -1, 0]]).to(device=self.this_device)
        
        self.aux_q_dot_dw_2 = torch.Tensor([[0, 0, -1, 0],
                                         [0, 0, 0, -1],
                                         [1, 0, 0, 0],
                                         [0, 1, 0, 0]]).to(device=self.this_device)
        
        self.aux_q_dot_dw_3 = torch.Tensor([[0, 0, 0, -1],
                                         [0, 0, -1, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, 0, 0]]).to(device=self.this_device)

        # d_q_dot_dq
        self.aux_d_q_dot_dq_1 = torch.Tensor([[0, 0, 0],
                                           [1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]]).to(device=self.this_device)

        self.aux_d_q_dot_dq_2 = torch.Tensor([[-1, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, -1],
                                           [0, 1, 0]]).to(device=self.this_device)

        self.aux_d_q_dot_dq_3 = torch.Tensor([[0, -1, 0],
                                             [0, 0, 1],
                                             [0, 0, 0],
                                             [-1, 0, 0]]).to(device=self.this_device)

        self.aux_d_q_dot_dq_4 = torch.Tensor([[0, 0, -1],
                                             [0, -1, 0],
                                             [1, 0, 0],
                                             [0, 0, 0]]).to(device=self.this_device)

        # d_v_dot_d_q
        self.aux_d_v_dot_d_q_1 = torch.Tensor([[0, 0, 1, 0],
                                             [0, -1, 0, 0],
                                             [1, 0, 0, 0]]).to(device=self.this_device)

        self.aux_d_v_dot_d_q_2 = torch.Tensor([[0, 0, 0, 1],
                                               [-1, 0, 0, 0],
                                               [0, -1, 0, 0]]).to(device=self.this_device)

        self.aux_d_v_dot_d_q_3 = torch.Tensor([[1, 0, 0, 0],
                                               [0, 0, 0, 1],
                                               [0, 0, -1, 0]]).to(device=self.this_device)

        self.aux_d_v_dot_d_q_4 = torch.Tensor([[0, 1, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]]).to(device=self.this_device)

        # w_dot_dw

        self.aux_w_dot_dw_1 = torch.Tensor([[0, 0, 0],
                                            [0, 0, (self.inertia_diag[2] - self.inertia_diag[0])/self.inertia_diag[1]],
                                            [0, (self.inertia_diag[0] - self.inertia_diag[1])/self.inertia_diag[2], 0]]).to(device=self.this_device)

        self.aux_w_dot_dw_2 = torch.Tensor([[0, 0, (self.inertia_diag[1] - self.inertia_diag[2])/self.inertia_diag[0]],
                                            [0, 0, 0],
                                            [(self.inertia_diag[0] - self.inertia_diag[1])/self.inertia_diag[2], 0, 0]]).to(device=self.this_device)

        self.aux_w_dot_dw_3 = torch.Tensor([[0, (self.inertia_diag[1] - self.inertia_diag[2])/self.inertia_diag[0], 0],
                                            [(self.inertia_diag[2] - self.inertia_diag[0])/self.inertia_diag[1], 0, 0],
                                            [0, 0, 0]]).to(device=self.this_device)

        # d_v_dot_du
        self.aux_dv_dot_du_1 = torch.Tensor([[0, 0, 1, 0],
                                             [0, 0, 0, 1],
                                             [1, 0, 0, 0],
                                             [0, 1, 0, 0]]).to(device=self.this_device)

        self.aux_dv_dot_du_2 = torch.Tensor([[0, -1, 0, 0],
                                             [-1, 0, 0, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 1, 0]]).to(device=self.this_device)

        self.aux_dv_dot_du_3 = torch.Tensor([[1, 0, 0, 0],
                                             [0, -1, 0, 0],
                                             [0, 0, -1, 0],
                                             [0, 0, 0, 1]]).to(device=self.this_device)



    def jac_q_dot_dw_mat(self, q):
        return 0.5*torch.stack((torch.matmul(self.aux_q_dot_dw_1, q), torch.matmul(self.aux_q_dot_dw_2,q), torch.matmul(self.aux_q_dot_dw_3,q)), dim=1)

    def jac_q_dot_dq_mat(self, omega):
        aux1 = torch.matmul(self.aux_d_q_dot_dq_1, omega)
        aux2 = torch.matmul(self.aux_d_q_dot_dq_2, omega)
        aux3 = torch.matmul(self.aux_d_q_dot_dq_3, omega)
        aux4 = torch.matmul(self.aux_d_q_dot_dq_4, omega)

        return 0.5 * torch.stack((aux1, aux2, aux3, aux4), dim=1)

    def jac_d_v_dot_dq_mat(self, q, f_c):
        m = self.mass
        # f_c = torch.sum(u, dim=0)
        qfcm = q * f_c / m

        aux1 = torch.matmul(self.aux_d_v_dot_d_q_1, qfcm)
        aux2 = torch.matmul(self.aux_d_v_dot_d_q_2, qfcm)
        aux3 = torch.matmul(self.aux_d_v_dot_d_q_3, qfcm)
        aux4 = torch.matmul(self.aux_d_v_dot_d_q_4, qfcm)
        
        return 2 * torch.stack((aux1, aux2, aux3, aux4), dim=1)

    def jac_w_dot_dw_mat(self, omega):
        # ans = torch.zeros(3,3)
        aux1 = torch.matmul(self.aux_w_dot_dw_1, omega)
        aux2 = torch.matmul(self.aux_w_dot_dw_2, omega)
        aux3 = torch.matmul(self.aux_w_dot_dw_3, omega)

        return torch.stack((aux1, aux2, aux3), dim=1)

    def jac_d_v_dot_df_mat(self, q):
        m = self.mass       # kg

        tmp1 = torch.matmul(torch.matmul(q.T, self.aux_dv_dot_du_1), q)
        tmp2 = torch.matmul(torch.matmul(q.T, self.aux_dv_dot_du_2), q)
        tmp3 = torch.matmul(torch.matmul(q.T, self.aux_dv_dot_du_3), q)
        tmp = torch.stack((tmp1, tmp2, tmp3), dim=0)/m

        return tmp.T


    def nonlinear_dynamics(self, x, u):
        return self.nonlinear_dynamics_CTBR(x, u)


    def nonlinear_dynamics_CTBR(self, x, u):
        if x.is_cuda:
            this_device = "cuda:0"
        else:
            this_device = "cpu"

        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == self.n_state
        assert u.shape[1] == self.n_ctrl
        assert u.ndimension() == 2

        n_batch = x.shape[0]

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z = torch.unbind(x, dim=1)
        p = torch.stack([p_x, p_y, p_z])
        q = torch.stack([q_w, q_x, q_y, q_z])
        v = torch.stack([v_x, v_y, v_z])

        w = u[:, 1:4]

        fc = u[:, 0]


        p_dot = v


        # print(0.5*quat_mult(q, torch.cat([torch.full((1, n_batch), 0), w])))
        # Naive integration of quaternion:
        # new_q = q + self.dt * (0.5*quat_mult(q, torch.cat([torch.full((1, n_batch), 0), w])))
        # print(torch.norm(new_q, dim=0))

        # Better integration:
        q_dot = 0.5*quat_mult(q, torch.cat([torch.full((1, n_batch), 0, device=this_device), w.t()]))
        # print(torch.norm(new_q, dim=0))
        batched_c_thrust = torch.cat([torch.full((2, n_batch), 0).to(device=this_device), fc.unsqueeze(0)])
        v_dot = rotate_quat(q, batched_c_thrust)/self.mass + self.g.repeat(n_batch,1).t().to(device=this_device)

        state = torch.cat([p_dot, q_dot, v_dot]).t()
        # print(state)

        if squeeze:
            state = state.squeeze(0)
        return state


    def forward(self, x, u):
        return self.forward_CTBR(x, u)

    def forward_CTBR(self, x, u):
        if x.is_cuda:
            this_device = "cuda:0"
        else:
            this_device = "cpu"

        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == self.n_state
        assert u.shape[1] == self.n_ctrl
        assert u.ndimension() == 2

        n_batch = x.shape[0]

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()


        p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z = torch.unbind(x, dim=1)
        p = torch.stack([p_x, p_y, p_z])

        # u[:, 0] = torch.clamp(u[:, 0], self.thrust_min*4, self.thrust_max*4)
        q = torch.stack([q_w, q_x, q_y, q_z])
        v = torch.stack([v_x, v_y, v_z])

        w = u[:, 1:4]
        fc = u[:, 0]


        # print(0.5*quat_mult(q, torch.cat([torch.full((1, n_batch), 0), w])))
        # Naive integration of quaternion:
        # new_q = q + self.dt * (0.5*quat_mult(q, torch.cat([torch.full((1, n_batch), 0), w])))
        # print(torch.norm(new_q, dim=0))

        new_p = p + self.dt * (v)

        # Better integration:
        em = exp_map(w.t(), self.dt)
        new_q = quat_mult(q, em)
        # print(torch.norm(new_q, dim=0))

        batched_c_thrust = torch.cat([torch.full((2, n_batch), 0).to(device=this_device), fc.unsqueeze(0)])
        new_v = v + self.dt * (rotate_quat(q, batched_c_thrust)/self.mass + self.g.repeat(n_batch,1).t().to(device=this_device))

        state = torch.cat([new_p, new_q, new_v]).t()

        # print(state)

        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x, ax=None):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == self.n_state


        p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z = torch.unbind(x)


        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()


        ax.plot((0,p_x), (0, p_z), color='k')
        ax.set_xlim((-10.0, 10.0))
        ax.set_ylim((-10.0, 10.0))
        return fig, ax

    def get_A_B_mat(self, x, u):
        return self.get_A_B_mat_CTBR(x, u)

    def get_A_B_mat_CTBR(self, x, u):
        if x.is_cuda:
            this_device = "cuda:0"
        else:
            this_device = "cpu"

        n_grad = x.shape[0]

        FC_IDX = 0
        OMEGA_RED_IDX = slice(1,4)

        R_vec = torch.zeros(n_grad, self.n_state, self.n_state, device=this_device)
        S_vec = torch.zeros(n_grad, self.n_state, self.n_ctrl, device=this_device)


        R_vec[:, self.POS_IDX, self.VEL_IDX] = torch.eye(3).reshape(1, 3, 3).repeat(n_grad, 1, 1).to(device=this_device)
        # d(q_dot)/dq
        omegas = u[:, OMEGA_RED_IDX]
        R_vec[:, self.QUAT_IDX, self.QUAT_IDX] = self.vect_jac_q_dot_dq(omegas).to(device=this_device)

        fc = u[:, FC_IDX]

        # d(v_dot)/dq
        R_vec[:, self.VEL_IDX, self.QUAT_IDX] = self.vect_jac_d_v_dot_dq(x[:, self.QUAT_IDX], fc).to(device=this_device)

        # d(v_dot)/dthrust
        S_vec[:, self.VEL_IDX, FC_IDX] = self.vect_jac_d_v_dot_df(x[:, self.QUAT_IDX]).to(device=this_device)

        # d(q_dot)/dw
        S_vec[:, self.QUAT_IDX, OMEGA_RED_IDX] = self.vect_jac_q_dot_dw(x[:, self.QUAT_IDX]).to(device=this_device)

        return R_vec, S_vec



    def grad_input(self, x, u):
        if x.is_cuda:
            this_device = "cuda:0"
        else:
            this_device = "cpu"

        n_grad = x.shape[0]

        R_vec, S_vec = self.get_A_B_mat_CTBR(x, u)

        # Discretization
        R_vec = (R_vec * self.dt + torch.eye(self.n_state).to(device=this_device).reshape(1, self.n_state, self.n_state).repeat(n_grad, 1, 1))
        S_vec = (S_vec * self.dt)



        return R_vec, S_vec

    def get_true_obj(self):
        q = self.goal_weights
        assert not hasattr(self, 'mpc_lin')
        q_ref = self.goal_tau[3:7]
        M = quat2_rmatrix(q_ref)
        q_quat = M.t() * self.goal_weights[3:7] * M
        Q = torch.diag(q)
        Q[3:7, 3:7] = q_quat
        # px = -torch.sqrt(self.goal_weights) * self.goal_state #+ self.mpc_lin
        # px[3:7] = torch.zeros(4)
        p = -torch.sqrt(self.goal_weights) * self.goal_tau #+ self.mpc_lin
        p[3:7] = torch.zeros(4)
        # p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(Q), Variable(p)


if __name__ == '__main__':
    # For debugging
    dx = DroneDx()
    n_batch, T = 8, 50
    # print(dx.get_true_obj())
    # u = torch.zeros(T, n_batch, dx.n_ctrl)
    u = torch.full((T, n_batch, dx.n_ctrl), 0.0)
    u[:, :, 0] += 0.752 * 9.806
    u[:, :, 2] += 0.1
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit[:, 3] = 1.0   # q_w = 1.0
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
        print("x")
        print(x)
        fig, ax = dx.get_frame(x[0])
        fig.savefig('img/{:03d}.png'.format(t))
        plt.close(fig)

    vid_file = 'drone_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('(/usr/bin/ffmpeg '
            '-r 32 -f image2 -i img/%03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}) &').format(
        vid_file
    )
    os.system(cmd)
    # for t in range(T):
    #     os.remove('img/{:03d}.png'.format(t))
