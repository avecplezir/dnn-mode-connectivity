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

model1 = architecture.base(num_classes=10, **architecture.kwargs)
model1.cuda()
has_bn1 = utils.check_bn(model1)

model2 = architecture.base(num_classes=10, **architecture.kwargs)
model2.cuda()
has_bn2 = utils.check_bn(model2)

base_model = architecture.base(10, **architecture.kwargs)
base_model.cuda()

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(1e-4)


statistic = []

for i in range(47, 100):

    ckpt = 'curves/curve' + str(i) + '/checkpoint-0.pt'
    checkpoint = torch.load(ckpt)
    model1.load_state_dict(checkpoint['model_state'])

    for j in range(i, i+1):

            ckpt = 'curves/curve' + str(j) + '/checkpoint-100.pt'
            checkpoint = torch.load(ckpt)
            model2.load_state_dict(checkpoint['model_state'])

            for parameter, p1, p2 in zip(base_model.parameters(), model1.parameters(), model2.parameters()):
                parameter.data.copy_((p1+p2)/2) #(p1+p2)/2

            par1 = np.concatenate([p.data.cpu().numpy().ravel() for p in model1.parameters()])
            par2 = np.concatenate([p.data.cpu().numpy().ravel() for p in model2.parameters()])
            u = par2 - par1
            dx = np.linalg.norm(u)

            utils.update_bn(loaders['train'], base_model)

            train_res = utils.test(loaders['train'], base_model, criterion, regularizer)
            test_res = utils.test(loaders['test'], base_model, criterion, regularizer)

            columns = ['models', 'dist', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc']

            values = [ [i, j], dx, train_res['loss'], train_res['accuracy'], test_res['nll'],
                      test_res['accuracy']]

            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')

            print(table)

            statistic.append(values)

pickle.dump(statistic, open("stats2/100middle_point_stat.p", "wb"))