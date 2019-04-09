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


class BatchNorm(nn.Module):
    def __init__(self, dim_in):
        super(BatchNorm, self).__init__()

        self.mu = torch.zeros(dim_in).cuda()
        self.sig2 = torch.zeros(dim_in).cuda() + 0.1
        self.momentum = 0.1

    def forward(self, x):

        if self.training:
            mu = x.mean(0)
            sig2 = (x - mu).pow(2).mean(0)
            x = (x - mu) / (sig2 + 1.0e-6).sqrt()
            self.mu = self.momentum * mu + (1 - self.momentum) * self.mu
            self.sig2 = self.momentum * sig2 + (1 - self.momentum) * self.sig2
            return x, sig2 + 1.0e-6
        else:
            x = (x - self.mu) / (self.sig2 + 1.0e-6).sqrt()
            return x, self.sig2 + 1.0e-6
