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

parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--number_points', type=int, default=1, metavar='NM',
                    help='for how many points compute centre mass')


args = parser.parse_args()

loaders, num_classes = data.loaders(
    "CIFAR10",
    "data",
    128,
    1,
    "VGG",
    False)

architecture = getattr(models, "VGG16")

number_points = args.number_points

models = [ architecture.base(num_classes=10, **architecture.kwargs) for i in range(number_points)]

for m in models:
    m.cuda()

base_model = architecture.base(10, **architecture.kwargs)
base_model.cuda()

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(1e-4)


statistic = []

ind = 47
T = True

# index = list(range(number_points))


while ind < 57-number_points+1:

    l = []
    for m in models:

        ckpt = 'curves/curve' + str(ind) + '/checkpoint-100.pt'
        checkpoint = torch.load(ckpt)
        m.load_state_dict(checkpoint['model_state'])
        ind+=1
        l.append(ind)

    for i, m in enumerate(models):
        for parameter, p in zip(base_model.parameters(), m.parameters()):
            if i == 0:
                parameter.data.copy_((p))
            else:
                parameter.data.copy_((parameter+p))

    for parameter in base_model.parameters():
        parameter.data.copy_(parameter/len(models))

    utils.update_bn(loaders['train'], base_model)

    train_res = utils.test(loaders['train'], base_model, criterion, regularizer)
    test_res = utils.test(loaders['test'], base_model, criterion, regularizer)

    columns = ['models', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc']

    values = [ l, train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy']]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')

    print(table)

    statistic.append(values)

    ind -= (number_points-1)

pickle.dump( statistic, open( "stats/centre_mass_stat_"+str(number_points)+".p", "wb"))
