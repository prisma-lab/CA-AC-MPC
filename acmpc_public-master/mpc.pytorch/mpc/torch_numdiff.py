#!/usr/bin/env python3

import torch
from torch.autograd import Variable

import itertools

def grad(net, inputs, eps=1e-4):
    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xp, xn = [], []
    e = 0.5*eps*torch.eye(nDim).type_as(inputs.data)
    for b in range(nBatch):
        for i in range(nDim):
            xp.append((inputs.data[b].clone()+e[i]).unsqueeze(0))
            xn.append((inputs.data[b].clone()-e[i]).unsqueeze(0))
    xs = Variable(torch.cat(xp+xn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fs_p, fs_n = torch.split(fs, nBatch*nDim)
    g = ((fs_p-fs_n)/eps).view(nBatch, nDim, fDim).squeeze(2)
    return g

def hess(net, inputs, eps=1e-4):
    assert(inputs.ndimension() == 2)
    nBatch, nDim = inputs.size()
    xpp, xpn, xnp, xnn = [], [], [], []
    e = eps*torch.eye(nDim).type_as(inputs.data)
    for b,i,j in itertools.product(range(nBatch), range(nDim), range(nDim)):
        xpp.append((inputs.data[b].clone()+e[i]+e[j]).unsqueeze(0))
        xpn.append((inputs.data[b].clone()+e[i]-e[j]).unsqueeze(0))
        xnp.append((inputs.data[b].clone()-e[i]+e[j]).unsqueeze(0))
        xnn.append((inputs.data[b].clone()-e[i]-e[j]).unsqueeze(0))
    xs = Variable(torch.cat(xpp+xpn+xnp+xnn))
    fs = net(xs)
    fDim = fs.size(1) if fs.ndimension() > 1 else 1
    fpp, fpn, fnp, fnn = torch.split(fs, nBatch*nDim*nDim)
    h = ((fpp-fpn-fnp+fnn)/(4*eps*eps)).view(nBatch, nDim, nDim, fDim).squeeze(3)
    return h
