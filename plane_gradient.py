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

parser = argparse.ArgumentParser(description='Computes values for plane visualization')
parser.add_argument('--dir', type=str, default='/tmp/plane', metavar='DIR',
                    help='training directory (default: /tmp/plane)')

parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                    help='number of points in the grid (default: 21)')
parser.add_argument('--margin_left', type=float, default=0.2, metavar='M',
                    help='left margin (default: 0.2)')
parser.add_argument('--margin_right', type=float, default=0.2, metavar='M',
                    help='right margin (default: 0.2)')
parser.add_argument('--margin_bottom', type=float, default=0.2, metavar='M',
                    help='bottom margin (default: 0.)')
parser.add_argument('--margin_top', type=float, default=0.2, metavar='M',
                    help='top margin (default: 0.2)')

parser.add_argument('--curve_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')

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
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False
)

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)

curve_model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
curve_model.cuda()

checkpoint = torch.load(args.ckpt)
curve_model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = utils.l2_regularizer(args.wd)


def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])


w = list()
curve_parameters = list(curve_model.net.parameters())
for i in range(args.num_bends):
    w.append(np.concatenate([
        p.data.cpu().numpy().ravel() for p in curve_parameters[i::args.num_bends]
    ]))

print('Weight space dimensionality: %d' % w[0].shape[0])

u = w[2] - w[0]
dx = np.linalg.norm(u)
u /= dx

print('Computing gradient vector')

def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            utils.adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    grad = np.concatenate([p.grad.data.cpu().numpy().ravel() for p in model.parameters()])

    return grad

model1 = architecture.base(num_classes=10, **architecture.kwargs)
model2 = architecture.base(num_classes=10, **architecture.kwargs)
model3 = architecture.base(num_classes=10, **architecture.kwargs)
model3.cuda()


def init_model(p, model):
    offset = 0
    for parameter in model.parameters():
        size = np.prod(parameter.size())
        value = p[offset:offset + size].reshape(parameter.size())
        parameter.data.copy_(torch.from_numpy(value))
        offset += size


init_model(w[0], model1)
init_model(w[2], model2)

criterion = F.cross_entropy
regularizer = None
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model3.parameters()),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0
)

num_points = 15
l_grad = []
for j in range(0, num_points):
    for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
        alpha = j * 1.0 / (num_points-1)
        p3.data.copy_(alpha * p1.data + (1.0 - alpha) * p2.data)
    grad = train(loaders['train'], model3, optimizer, criterion, )
    l_grad.append(grad)

for i, v in enumerate(l_grad):
    print(np.linalg.norm(v))
    l_grad[i] = v/np.linalg.norm(v)

s = 0
for v in l_grad:
    s += v
print(np.linalg.norm(s)/num_points)
s = s/np.linalg.norm(s)


v = w[1] - w[0]
v -= np.dot(u, v) * u
dy = np.linalg.norm(v)
v /= dy

v = s

bend_coordinates = np.stack(get_xy(p, w[0], u, v) for p in w)

ts = np.linspace(0.0, 1.0, args.curve_points)
curve_coordinates = []
for t in np.linspace(0.0, 1.0, args.curve_points):
    weights = curve_model.weights(torch.Tensor([t]).cuda())
    curve_coordinates.append(get_xy(weights, w[0], u, v))
curve_coordinates = np.stack(curve_coordinates)

G = args.grid_points
alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

tr_loss = np.zeros((G, G))
tr_nll = np.zeros((G, G))
tr_acc = np.zeros((G, G))
tr_err = np.zeros((G, G))

te_loss = np.zeros((G, G))
te_nll = np.zeros((G, G))
te_acc = np.zeros((G, G))
te_err = np.zeros((G, G))

grid = np.zeros((G, G, 2))

base_model = architecture.base(num_classes, **architecture.kwargs)
base_model.cuda()

print("Computing grid")

columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        p = w[0] + alpha * dx * u + beta * dy * v

        offset = 0
        for parameter in base_model.parameters():
            size = np.prod(parameter.size())
            value = p[offset:offset+size].reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size


        utils.update_bn(loaders['train'], base_model)

        tr_res = utils.test(loaders['train'], base_model, criterion, regularizer)
        te_res = utils.test(loaders['test'], base_model, criterion, regularizer)

        tr_loss_v, tr_nll_v, tr_acc_v = tr_res['loss'], tr_res['nll'], tr_res['accuracy']
        te_loss_v, te_nll_v, te_acc_v = te_res['loss'], te_res['nll'], te_res['accuracy']

        c = get_xy(p, w[0], u, v)
        grid[i, j] = [alpha * dx, beta * dy]

        tr_loss[i, j] = tr_loss_v
        tr_nll[i, j] = tr_nll_v
        tr_acc[i, j] = tr_acc_v
        tr_err[i, j] = 100.0 - tr_acc[i, j]

        te_loss[i, j] = te_loss_v
        te_nll[i, j] = te_nll_v
        te_acc[i, j] = te_acc_v
        te_err[i, j] = 100.0 - te_acc[i, j]

        values = [
            grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
            te_nll[i, j], te_err[i, j]
        ]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if j == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

np.savez(
    os.path.join(args.dir, 'plane.npz'),
    ts=ts,
    bend_coordinates=bend_coordinates,
    curve_coordinates=curve_coordinates,
    alphas=alphas,
    betas=betas,
    grid=grid,
    tr_loss=tr_loss,
    tr_acc=tr_acc,
    tr_nll=tr_nll,
    tr_err=tr_err,
    te_loss=te_loss,
    te_acc=te_acc,
    te_nll=te_nll,
    te_err=te_err
)
