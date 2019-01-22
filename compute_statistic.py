import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils

ckpt = 'curves/curve13/checkpoint-100.pt'

loaders, num_classes = data.loaders(
    "CIFAR10",
    "data",
    128,
    1,
    "VGG",
    False)

architecture = getattr(models, "VGG16")
curve = getattr(curves, 'PolyChain')

model = curves.CurveNet(
            10,
            curve,
            architecture.curve,
            3,
            True,
            True,
            architecture_kwargs=architecture.kwargs,
            )


has_bn = utils.check_bn(model)
test_res = {'loss': None, 'accuracy': None, 'nll': None}

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(1e-4)


checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint['model_state'])

train_res = utils.test(loaders['train'], model, criterion, regularizer)
test_res = utils.test(loaders['test'], model, criterion, regularizer)

values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
          test_res['accuracy'], time_ep]

table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
if epoch % 40 == 1 or epoch == start_epoch:
    table = table.split('\n')
    table = '\n'.join([table[1]] + table)
else:
    table = table.split('\n')[2]
print(table)