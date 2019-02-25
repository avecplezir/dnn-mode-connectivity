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

parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--name', type=str, default='/tmp/name/', metavar='NAME',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--l1', type=int, default=-1, metavar='L1')
parser.add_argument('--l2', type=int, default=-2, metavar='L2')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')




args = parser.parse_args()

architecture = getattr(models, args.model)

model = architecture.base(num_classes=10, **architecture.kwargs)

model.load_state_dict(torch.load(args.ckpt)['model_state'])

loaders, num_classes = data.loaders(
    "CIFAR10",
    "data",
    128,
    1,
    "VGG",
    False
)

for X, y in loaders['train']:
    break

y_pred = torch.argmax(model(X), dim=-1)

print("y ", y)

print("y_pred ", y_pred)

def change_node(l1, l2, i, j):
    
    c = copy.deepcopy(torch.nn.Parameter(list(model.modules())[l1].weight[j]))
    list(model.modules())[l1].weight[j] = list(model.modules())[l1].weight[i]
    list(model.modules())[l1].weight[i] = c
    
    c = copy.deepcopy(torch.nn.Parameter(list(model.modules())[l2].weight.transpose(0,1)[j]))
    list(model.modules())[l2].weight.transpose(0,1)[j] = list(model.modules())[l2].weight.transpose(0,1)[i]
    list(model.modules())[l2].weight.transpose(0,1)[i] = c
    
for i in range(400):
    change_node(args.l1, args.l2, i, 2*i);


y_pred_n = torch.argmax(model(X), dim=-1)

print("eq ", ~(y_pred_n==y_pred))
print("err ", (~(y_pred_n==y_pred)).sum())

print("y_pred_n ", y_pred_n)

# saving changed checkpoints

checkpoints = torch.load(args.ckpt)

print("Saving checkpoint for node changing")

utils.save_checkpoint(
            args.dir,
            100,
            name=args.name,
            model_state=model.state_dict(),
            optimizer_state=checkpoints['optimizer_state']
            )
