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
import pickle

import copy

parser = argparse.ArgumentParser(description='Computes values for plane visualization')
parser.add_argument('--dir', type=str, default='points2plane/new', metavar='DIR',
                    help='training directory (default: /tmp/plane)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--init_E_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--init_E_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')


args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
num_classes = 10

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)

model = curves.CurveNet(
    10,
    curve,
    architecture.curve,
    4,
    True,
    True,
    architecture_kwargs=architecture.kwargs,
)

base_model = None
for path, k in [(args.init_start, 0), (args.init_E_start, 1), (args.init_E_end, 2), (args.init_end, 3)]:
    if path is not None:
        if base_model is None:
            base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        checkpoint = torch.load(path)
        print('Loading %s as point #%d' % (path, k))
        base_model.load_state_dict(checkpoint['model_state'])
        model.import_base_parameters(base_model, k)

utils.save_checkpoint(
    args.dir,
    -1,
    model_state=model.state_dict(),
    optimizer_state=-1
)
