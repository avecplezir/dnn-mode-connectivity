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
import pickle

loaders, num_classes = data.loaders(
    "CIFAR10",
    "data",
    128,
    1,
    "VGG",
    False)

architecture = getattr(models, "VGG16")

model = architecture.base(num_classes=10, **architecture.kwargs)
model.cuda()

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(1e-4)

statistic = []

for i in range(1, 44):
    ckpt = 'curves/curve'+str(i)+'/checkpoint-100.pt'

    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state'])

    train_res = utils.test(loaders['train'], model, criterion, regularizer)
    test_res = utils.test(loaders['test'], model, criterion, regularizer)

    columns = ['model', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc']

    values = [i, train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy']]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')

    print(table)

    statistic.append(values)


pickle.dump( statistic, open( "stats/point_stat.p", "wb"))