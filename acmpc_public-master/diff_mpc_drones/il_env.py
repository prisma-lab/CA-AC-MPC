import torch
torch.set_printoptions(linewidth=200)

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost
from mpc.dynamics import NNDynamics
import mpc.util as eutil

import numpy as np
import numpy.random as npr

import os
import sys
import shutil
import time

import pickle as pkl

from setproctitle import setproctitle

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector
from scipy import interpolate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import drone



class IL_Env:
    def __init__(self, env, lqr_iter=500, mpc_T=5, slew_rate_penalty=None):
        self.env = env


        if self.env == 'drone':
            self.true_dx = drone.DroneDx()
        else:
            assert False

        self.lqr_iter = lqr_iter
        self.mpc_T = mpc_T
        self.slew_rate_penalty = slew_rate_penalty

        self.grad_method = GradMethods.ANALYTIC
        # self.grad_method = GradMethods.AUTO_DIFF
        # self.grad_method = GradMethods.FINITE_DIFF
        # self.grad_method = GradMethods.ANALYTIC_CHECK

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.original_trajs = []


    def mpc(self, dx, xinit, Q, p, u_init=None, eps_override=False,
            lqr_iter_override=None):
        n_batch = xinit.shape[0]

        if xinit.is_cuda:
            this_device = "cuda:0"
        else:
            this_device = "cpu"

        n_sc = self.true_dx.n_state + self.true_dx.n_ctrl

        # p = p.unsqueeze(0).repeat(self.mpc_T, n_batch, 1)

        if eps_override:
            eps = eps_override
        else:
            eps = self.true_dx.mpc_eps

        if lqr_iter_override:
            lqr_iter = lqr_iter_override
        else:
            lqr_iter = self.lqr_iter


        lower = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
        lower[:,:,0] = self.true_dx.thrust_min*4
        lower[:,:,1] = -self.true_dx.omega_max[0]
        lower[:,:,2] = -self.true_dx.omega_max[1]
        lower[:,:,3] = -self.true_dx.omega_max[2]
        upper = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
        upper[:,:,0] = self.true_dx.thrust_max*4
        upper[:,:,1] = self.true_dx.omega_max[0]
        upper[:,:,2] = self.true_dx.omega_max[1]
        upper[:,:,3] = self.true_dx.omega_max[2]
        # Allow passing a warm-start sequence (used by AC-MPC to speed up MPC solves).
        # Fallback: hover thrust initialization.
        if u_init is None:
            u_init = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
            u_init[:, :, 0] = self.true_dx.mass * 9.8066
        else:
            u_init = u_init.to(device=this_device)



        x_mpc, u_mpc, objs_mpc = mpc.MPC(
            self.true_dx.n_state, self.true_dx.n_ctrl, self.mpc_T,
            u_lower=lower, u_upper=upper, u_init=u_init,
            lqr_iter=lqr_iter,
            verbose=-1,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=self.true_dx.linesearch_decay,
            max_linesearch_iter=self.true_dx.max_linesearch_iter,
            grad_method=self.grad_method,
            eps=eps,
            # slew_rate_penalty=self.slew_rate_penalty,
            # prev_ctrl=prev_ctrl,
            # best_cost_eps=1e0
        )(xinit, QuadCost(Q, p), dx)
        return x_mpc, u_mpc
