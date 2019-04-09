import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import torch.nn

import data
import models
import curves
import utils

import pickle

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.stats import norm

import utils
import time
from torch import nn
import seaborn as sns
from sklearn.manifold import TSNE
from pylab import rcParams

import flows

parser = argparse.ArgumentParser(description='flow training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--flow', type=str, default=None, metavar='FLOW', required=True,
                    help='flow name (default: None)')
parser.add_argument('--model1', type=str, default='curves_mnist/LinearOneLayer/LongTraining/curve1/checkpoint-30.pt', metavar='CKPT',
                    help='checkpoint to eval (default: None)')
parser.add_argument('--model2', type=str, default='curves_mnist/LinearOneLayer/LongTraining/curve2/checkpoint-30.pt', metavar='CKPT',
                    help='checkpoint to eval (default: None)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
args = parser.parse_args()

architecture = getattr(models, "LinearOneLayer")

model = architecture.base(num_classes=10, **architecture.kwargs)

loaders, num_classes = data.loaders(
    "MNIST",
    "data",
    128,
    1,
    "VGG",
    True)

architecture = getattr(models, "LinearOneLayer") #LinearOneLayer LogRegression
model1 = architecture.base(num_classes=10, **architecture.kwargs)
model2 = architecture.base(num_classes=10, **architecture.kwargs)

m = architecture.base(num_classes=10, **architecture.kwargs)

def distance(model1, model2):
    par1 = np.concatenate([p.data.cpu().numpy().ravel() for p in model1.parameters()])
    par2 = np.concatenate([p.data.cpu().numpy().ravel() for p in model2.parameters()])
    u = par2 - par1
    dx = np.linalg.norm(u)
    return dx

def get_vector(model1):
    par1 = np.concatenate([p.data.cpu().numpy().ravel() for p in model1.parameters()])
    return par1


def samples(model):
    p1 = list(model.parameters())[0].data.cpu().numpy()
    p2 = list(model.parameters())[1].data.cpu().numpy()
    p3 = list(model.parameters())[2].transpose(0, 1).data.cpu().numpy()
    samples = np.hstack([p1, p2[:, None], p3])

    return samples

print("making dataset...")
ind = 1
T = True
S = []
B = []
while ind < 20:
    ckpt = 'curves_mnist/LinearOneLayer/LongTraining/curve' + str(ind) + '/checkpoint-30.pt'
    checkpoint = torch.load(ckpt)
    m.load_state_dict(checkpoint['model_state'])

    S.append(samples(m))
    B.append(list(m.parameters())[-1].data.numpy())
    ind += 1

S = np.concatenate(S)


def iterate_minibatches(train_data, batchsize):
    indices = np.random.permutation(np.arange(len(train_data)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield torch.FloatTensor(train_data[ix]).cuda()


def train(flow, epochs=20, lr=1e-5, batchsize=512, step=10):
    flow.cuda();
    flow.train();
    trainable_parametrs = filter(lambda param: param.requires_grad,
                                 flow.parameters())  # list of all trainable parameters in a flow
    optimizer = torch.optim.Adam(trainable_parametrs, lr=lr)  # choose an optimizer, use module torch.optim

    for epoch in range(epochs + 1):

        t = time.time()
        total_loss = 0

        for X in iterate_minibatches(S, batchsize):
            loss = -flow.log_prob(X).mean()  # compute the maximum-likelihood loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        total_loss /= (len(S) // batchsize)
        if epoch % step == 0:
            print('epoch %s:' % epoch, 'loss = %.3f' % total_loss, 'time = %.2f' % (time.time() - t))

from torch import distributions
prior = distributions.MultivariateNormal(torch.zeros(795), torch.eye(795))

N = 2
X = torch.FloatTensor(S[:N]).cuda()


def get_model(W, B):
    model_sampled = architecture.base(num_classes=10, **architecture.kwargs)
    model_samples = W.cpu().data.numpy()
    SIZE = 2000

    offset = 0
    for parameter in list(model_sampled.parameters())[:-1]:
        size = int(np.prod(parameter.size()) / SIZE)
        value = model_samples[:, offset:offset + size]
        if size == 10 or size == 1:
            value = value.T
        #         print(value.shape)
        value = value.reshape(parameter.size())
        #         print(value.shape)
        parameter.data.copy_(torch.from_numpy(value))
        offset += size

    list(model_sampled.parameters())[-1].data.copy_(torch.from_numpy(B.mean(0)))

    return model_sampled


def test(model):
    criterion = F.cross_entropy
    regularizer = None
    train_res = utils.test(loaders['train'], model, criterion, regularizer)
    test_res = utils.test(loaders['test'], model, criterion, regularizer)
    print(train_res)
    print(test_res)


def test_flow(S, model1, model2, flow, N=2000):
    rcParams['figure.figsize'] = 12, 10
    rcParams['figure.dpi'] = 100

    model1.cuda(), model2.cuda(), flow.cuda()
    model1.eval(), model2.eval(), flow.eval()
    # print('copmuting samples...')
    # X = torch.FloatTensor(S[:N]).cuda()
    # X_sample = X.data.cpu().numpy()
    # X_prior = prior.sample((N,)).cpu().data.numpy()
    # X_flow = flow.sample(N, ).data.cpu().numpy()
    # X_sample_prior = flow.f(torch.FloatTensor(X_sample).cuda())[0].data.cpu().numpy()

    # print('drawing...')
    # i, j = 500, -1
    # fig, axes = plt.subplots(2, 2, )
    # axes[0, 0].set_title('Samples')
    # axes[0, 0].scatter(X_sample[:, i], X_sample[:, j])
    # axes[0, 1].set_title('Prior')
    # axes[0, 1].scatter(X_prior[:, i], X_prior[:, j])
    # axes[1, 0].set_title('Flow sampling')
    # axes[1, 0].scatter(X_flow[:, i], X_flow[:, j])
    # axes[1, 1].set_title('Map from samples to prior')
    # axes[1, 1].scatter(X_sample_prior[:, i], X_sample_prior[:, j])
    # plt.show()

    print('computing Arc model...')
    W1 = samples(model1)
    W2 = samples(model2)

    #     flow.cpu()
    W_pre = 1 / np.sqrt(2) * flow.f(torch.FloatTensor(W1).cuda())[0] + 1 / np.sqrt(2) * \
            flow.f(torch.FloatTensor(W2).cuda())[0]
    W = flow.g(W_pre)
    B = []
    B.append(list(model1.parameters())[-1].data.cpu().numpy())
    B.append(list(model2.parameters())[-1].data.cpu().numpy())
    B = np.array(B)

    model_sampled = get_model(W, B)
    test(model_sampled)

    if N == 2000:
        print('computing Sampling from flow model...')
        X_flow = flow.sample(N, ).data.cpu()

        model_flow = get_model(X_flow, B)
        test(model_flow)
    #         return model_sampled, model_flow

    return model_sampled


prior = distributions.MultivariateNormal(torch.zeros(795), torch.eye(795))

flow_architecture = getattr(flows, args.flow)

if args.flow == 'Flow_IAF':
    flow = flow_architecture(prior)
elif args.flow == 'RealNVP':
    N_layers = 15
    n_dim = 795
    onezero = [0, 1] * n_dim
    masks = torch.Tensor([[onezero[:n_dim], onezero[1:n_dim + 1]]] * N_layers)
    masks = masks.view(2 * N_layers, -1)

    flow = flow_architecture(masks, prior)
else:
    print('error, the flow not found')

train(flow, epochs=10, lr=args.lr, batchsize=1024, step=1)

model1.load_state_dict(torch.load(args.model1)['model_state'])
model2.load_state_dict(torch.load(args.model2)['model_state'])

test_flow(S, model1, model2, flow, N=2000);

os.makedirs(args.dir, exist_ok=True)

utils.save_checkpoint(
    args.dir,
    100,
    model_state=flow.state_dict(),
    optimizer_state=None
)