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

import data
import models
import curves
import utils

import pickle

from flows.layers import BatchNorm

class AutoregressiveLinear(nn.Module):
    def __init__(self, dim_in, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = dim_in
        self.out_size = out_size

        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output


class AutoregressiveLinearU(nn.Module):
    def __init__(self, dim_in, out_size, bias=True, ):
        super(AutoregressiveLinearU, self).__init__()

        self.in_size = dim_in
        self.out_size = out_size

        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.triu(1))

        output = input @ self.weight.triu(1)
        if self.bias is not None:
            output += self.bias
        return output


class IAF_block(nn.Module):
    def __init__(self, dim_in, dim_middle, Low=True):
        super(IAF_block, self).__init__()

        self.z_size = dim_in
        self.dim_middle = dim_middle
        self.Low = Low
        self.h = nn.LeakyReLU()
        affine = True

        if Low:
            self.m = nn.Sequential(
                AutoregressiveLinear(self.z_size, self.dim_middle),
                #                 nn.BatchNorm1d(dim_middle, affine=affine),
                #                 self.h,
                #                 AutoregressiveLinear(self.dim_middle, self.dim_middle),
                self.h,
                AutoregressiveLinear(self.dim_middle, self.z_size)
            )

            self.s = nn.Sequential(
                AutoregressiveLinear(self.z_size, self.dim_middle),
                #                 nn.BatchNorm1d(dim_middle, affine=affine),
                #                 self.h,
                #                 AutoregressiveLinear(self.dim_middle, self.dim_middle),
                self.h,
                AutoregressiveLinear(self.dim_middle, self.z_size)
            )
        else:
            self.m = nn.Sequential(
                AutoregressiveLinearU(self.z_size, self.dim_middle),
                #                 self.h,
                #                 AutoregressiveLinearU(self.dim_middle, self.dim_middle),
                # nn.BatchNorm1d(dim_middle, affine=affine),
                self.h,
                AutoregressiveLinearU(self.dim_middle, self.z_size)
            )

            self.s = nn.Sequential(
                AutoregressiveLinearU(self.z_size, self.dim_middle),
                #                 self.h,
                #                 AutoregressiveLinearU(self.dim_middle, self.dim_middle),
                # nn.BatchNorm1d(dim_middle, affine=affine),
                self.h,
                AutoregressiveLinearU(self.dim_middle, self.z_size)
            )

    def forward(self, z):

        self.mu_z = self.m(z)
        self.log_sigma_z = self.s(z)
        self.log_sigma_z = torch.clamp(self.log_sigma_z, -1, 1)

        #         self.sigma_z = nn.Sigmoid()(s)

        #         z = self.sigma_z*z+(1-self.sigma_z)*self.mu_z
        z = self.log_sigma_z.exp() * z + self.mu_z

        return z, self.log_sigma_z



class Flow_IAF(nn.Module):
    def __init__(self, prior):
        super(Flow_IAF, self).__init__()

        # Create a flow
        # nets:  a function that return a pytocrn neurel network e.g., nn.Sequential, s = nets(), s: dim(X) -> dim(X)
        # nett:  a function that return a pytocrn neurel network e.g., nn.Sequential, t = nett(), t: dim(X) -> dim(X)
        # mask:  a torch.Tensor of size #number_of_coupling_layers x #dim(X)
        # prior: an object from torch.distributions e.g., torch.distributions.MultivariateNormal

        self.prior = prior
        self.len = 20
        self.s = torch.nn.ModuleList([IAF_block(dim_in=795, dim_middle=795, Low=i % 2) for i in range(self.len)])
        self.b = torch.nn.ModuleList([BatchNorm(dim_in=795) for _ in range(self.len)])

        self.verbose = False
        self.batch_norm = True

    def g(self, z):
        # Compute and return g(z) = x,
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer
        # z: a torch.Tensor of shape batchSize x 1 x dim(X)
        # return x: a torch.Tensor of shape batchSize x 1 x dim(X)
        z = z.detach()
        for s, b in zip(reversed(self.s), reversed(self.b)):
            if s.Low:
                crange = reversed(range(z.size()[1]))
            else:
                crange = range(z.size()[1])

            for i in crange:

                _, log_sigma = s(z.detach())
                mu = s.mu_z
                if self.verbose:
                    print(i)
                    print('z1', z[:, i])
                z[:, i] = ((z[:, i] - mu[:, i]) * (-log_sigma[:, i]).exp()).detach()
                if self.verbose:
                    print('mu', mu[:, i])
                    print('sigma', (-log_sigma[:, i]).exp())
                    print('z2', z[:, i])

            if self.verbose:
                print('z1-bn', z)
            if self.batch_norm:
                z = z * (b.sig2 + 1.0e-6).sqrt() + b.mu
            #             print('sigma, mu', b.sig2[0], b.mu[0])
            if self.verbose:
                print('z2-bn', z)

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

        #         batch_size = z.size()[0]

        for i, (s, b) in enumerate(zip(self.s, self.b)):

            if self.batch_norm:
                z, sig2 = b(z)

            z, log_sigma = s(z)
            if self.batch_norm:
                log_det_J += (log_sigma - 0.5 * sig2.log()).sum(-1)
            else:
                log_det_J += log_sigma.sum(-1)

        return z, log_det_J

    def log_prob(self, x):
        # Compute and return log p(x)
        # using the change of variable formula and log_det_J computed by f
        # return logp: torch.Tensor of len batchSize
        z, log_det_J = self.f(x)

        logp = -0.5 * np.log(np.pi * 2) - 0.5 * z.pow(2)
        logp = logp.sum(-1)

        #         shape = torch.Size((K, self.in_dim))
        #         logp = torch.cuda.FloatTensor(x.shape[0])
        #         self.prior.log_prob(z.cpu(), out=logp)

        #         logp = self.prior.log_prob(z.cpu()).cuda()
        #         print('logp', logp.shape)

        return logp + log_det_J

    def sample(self, K):
        # Draw and return batchSize samples from flow using implementation of g
        # return x: torch.Tensor of shape batchSize x 1 x dim(X)

        z = self.prior.sample((K,)).cuda()
        x = self.g(z)

        return x

