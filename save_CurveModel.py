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

d = 'three_eq_points_nConve'
os.makedirs(d, exist_ok=True)

init_start = 'curve/checkpoint-100.pt'
init_middle = 'curve/checkpoint_n200Conve-100.pt'
init_end = 'curve/checkpoint_n200-100.pt'
num_classes = 10

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

base_model = None
for path, k in [(init_start, 0), (init_middle, 1), (init_end, 2)]:
    if path is not None:
        if base_model is None:
            base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        checkpoint = torch.load(path)
        print('Loading %s as point #%d' % (path, k))
        base_model.load_state_dict(checkpoint['model_state'])
        model.import_base_parameters(base_model, k)
            
utils.save_checkpoint(
                     d,
                     -1,
                     model_state=model.state_dict(),
                     optimizer_state=-1
                     )
