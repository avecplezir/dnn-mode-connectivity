import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils
import copy

architecture = getattr(models, "VGG16")

model = architecture.base(num_classes=10, **architecture.kwargs)
checkpoint_name = "checkpoint-100.pt"

model.load_state_dict(torch.load("curve/"+checkpoint_name)['model_state'])

loaders, num_classes = data.loaders(
    "CIFAR10",
    "data",
    128,
    1,
    "VGG",
    False)

for X, y in loaders['test']:
    break

y_pred = torch.argmax(model(X), dim=-1)

print("y ", y)

print("y_pred ", y_pred)

def rescale(l, scale):
    list(model.modules())[l].weight=torch.nn.Parameter(list(model.modules())[l].weight*scale)
    list(model.modules())[l].bias=torch.nn.Parameter(list(model.modules())[l].bias*scale)

rescale(17, 10)
rescale(18, 0.1)

checkpoints = torch.load("curve/"+checkpoint_name)

print("Saving new rescaling checkpoint")

utils.save_checkpoint(
            'curve',
            100,
            name='checkpoint_rConve17-18',
            model_state=model.state_dict(),
            optimizer_state=checkpoints['optimizer_state']
        )

y_pred_r = torch.argmax(model(X), dim=-1)

print("eq ", ~(y_pred==y_pred_r))
print("err ", (~(y_pred==y_pred_r)).sum())

print("y_pred_r ", y_pred_r)
