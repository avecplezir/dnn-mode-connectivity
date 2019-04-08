from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
import math

import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import torch.nn
from torch import nn

from flows.layers import BatchNorm



class SNet(nn.Module):
    def __init__(self, dim_in, dim_middle):
        super(SNet, self).__init__()
        affine = True
        self.h = nn.Tanh()  # nn.LeakyReLU() #nn.Tanh()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_middle),
            self.h,
            # nn.BatchNorm1d(dim_middle, affine=affine),
            nn.Linear(dim_middle, dim_middle),
            self.h,
            # nn.BatchNorm1d(dim_middle, affine=affine),
            nn.Linear(dim_middle, dim_in)
        )

    def forward(self, x):
        x = self.fc(x)
        #         x = torch.clamp(x, -1, 1)
        return x


class TNet(nn.Module):
    def __init__(self, dim_in, dim_middle):
        super(TNet, self).__init__()
        affine = True
        self.h = nn.Tanh()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_middle),
            self.h,
            nn.BatchNorm1d(dim_middle, affine=affine),
            nn.Linear(dim_middle, dim_middle),
            self.h,
            nn.BatchNorm1d(dim_middle, affine=affine),
            nn.Linear(dim_middle, dim_in),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class RealNVP(nn.Module):
    def __init__(self, mask, prior):
        super(RealNVP, self).__init__()

        # Create a flow
        # nets:  a function that return a pytocrn neurel network e.g., nn.Sequential, s = nets(), s: dim(X) -> dim(X)
        # nett:  a function that return a pytocrn neurel network e.g., nn.Sequential, t = nett(), t: dim(X) -> dim(X)
        # mask:  a torch.Tensor of size #number_of_coupling_layers x #dim(X)
        # prior: an object from torch.distributions e.g., torch.distributions.MultivariateNormal

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([SNet(dim_in=795, dim_middle=795) for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([TNet(dim_in=795, dim_middle=795) for _ in range(len(mask))])
        self.b = torch.nn.ModuleList([BatchNorm(dim_in=795) for _ in range(len(mask))])
        self.batch_norm = True
        self.verbose = False

    def g(self, z):
        # Compute and return g(z) = x,
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer
        # z: a torch.Tensor of shape batchSize x 1 x dim(X)
        # return x: a torch.Tensor of shape batchSize x 1 x dim(X)
        for i, (s, t, b) in enumerate(zip(reversed(self.s), reversed(self.t), reversed(self.b))):
            m = self.mask[-i - 1]
            #             print('i', i, 'm', m)

            if self.verbose:
                print('z1', z)
            z = (m * z + (1 - m) * (z - t(m * z)) * (-s(m * z)).exp()).detach()
            #             print('z1', z)
            if self.batch_norm:
                z = (z * (b.sig2 + 1.0e-6).sqrt() + b.mu).detach()
            if self.verbose:
                print('z2', z)

        x = z
        return x

    def f(self, x):
        # Compute f(x) = z and log_det_Jakobian of f,
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer
        # x: a torch.Tensor, of shape batchSize x dim(X), is a datapoint
        # return z: a torch.Tensor of shape batchSize x dim(X), a hidden representations
        # return log_det_J: a torch.Tensor of len batchSize

        z = x
        log_det_J = 0
        for s, t, m, b in zip(self.s, self.t, self.mask, self.b):

            if self.batch_norm:
                z, sig2 = b(z)

            s_res = s(m * z)
            z = m * z + (1 - m) * (z * s_res.exp() + t(m * z))

            if self.batch_norm:
                log_det_J += ((1 - m) * s_res - 0.5 * sig2.log()).sum(-1)
            else:
                log_det_J += ((1 - m) * s_res).sum(-1)

        return z, log_det_J

    def log_prob(self, x):
        # Compute and return log p(x)
        # using the change of variable formula and log_det_J computed by f
        # return logp: torch.Tensor of len batchSize
        z, log_det_J = self.f(x)

        #         logp = -0.5*np.log(np.pi*2)-0.5*z.pow(2)
        #         logp = logp.sum(-1)

        logp = self.prior.log_prob(z.cpu()).cuda()


        return logp + log_det_J

    def sample(self, K):
        # Draw and return batchSize samples from flow using implementation of g
        # return x: torch.Tensor of shape batchSize x 1 x dim(X)

        z = self.prior.sample((K,)).cuda()
        x = self.g(z)

        return x