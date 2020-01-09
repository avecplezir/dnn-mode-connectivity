import sys
sys.path.append('/home/ivan/distribution_connector')
import os
import numpy as np
import torch
from tqdm import tqdm
from connector_utils import test_models, gather_statistics, test_func
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from sklearn.decomposition import PCA, KernelPCA

from connector import Connector
# from one_layer_utils import samples, make_dataset, get_model, get_b
from utils import test_model
from tqdm import tqdm

import models

print('getting loaders')
import data
loaders, num_classes = data.loaders(
    "CIFAR10",
    "data",
    1024,
    1,
    "VGG",
    True,
    train_random=False,
    shuffle_train=False)


def get_data(data_type='train'):
    targ = []
    data = []
    for X, y in loaders[data_type]:
        data.append(X.view(-1, 3 * 32 * 32).cpu().data.numpy())
        targ.append(y)
    data = np.concatenate(data)
    targ = np.concatenate(targ)

    return data, targ

print('getting data')
data, targ = get_data(data_type='train')
data_test, targ_test = get_data(data_type='test')

# data, targ = data[:10], targ[:10]

print('train len', len(data), 'test len', len(data_test))

def next_layer(W, data):
    funcs = np.maximum(data @ W.T, 0)
    return funcs


def accuracy(pred, targ):
    ens_acc = 100.0 * np.mean(np.argmax(pred, axis=1) == targ)
    return ens_acc


def get_model_weights(model):
    p = [list(model.parameters())[i].data.cpu().numpy() for i in range(len(list(model.parameters())))]
    return p


def get_model_from_weights(W, architecture):
    model_sampled = architecture.base(num_classes=10, **architecture.kwargs)
    for parameter, w in zip(model_sampled.parameters(), W):
        parameter.data.copy_(torch.from_numpy(w))
    return model_sampled

def get_end_point_stat(architecture, model_name, beg_ind=1, end_ind=8):
    stat = {'train': [], 'test': []}
    m = architecture.base(num_classes=10, **architecture.kwargs)
    m.cuda();
    for i in tqdm(range(beg_ind, end_ind)):
        m.load_state_dict(
            torch.load('curves/' + model_name + '/curve' + str(i) + '/checkpoint-400.pt')['model_state'])
        res = test_model(m, loaders, cuda=True)
        stat['train'].append(res[0]['accuracy'])
        stat['test'].append(res[1]['accuracy'])
    return stat

def get_stat_plus_zeroing(pointfinder, model_name, architecture, t, method='arc_connect',
             beg_ind=1, end_ind=7):
    stat = {'test': [], 'train': []}
    stat_zero = {'test': [], 'train': []}
    model1 = architecture.base(num_classes=10, **architecture.kwargs)
    model2 = architecture.base(num_classes=10, **architecture.kwargs)
    for i in tqdm(range(beg_ind, end_ind)):
        model1.load_state_dict(torch.load('curves/'+model_name+'/curve'+str(i)+'/checkpoint-400.pt')['model_state'])
        model2.load_state_dict(torch.load('curves/'+model_name+'/curve'+str(i+1)+'/checkpoint-400.pt')['model_state'])
        finder = pointfinder(model1, model2, architecture)
        point = finder.find_point(t=t, method=method)
        stat['test'].append(point['test'])
        stat['train'].append(point['train'])
        point = finder.test_zeroing()
        stat_zero['test'].append(point['test'])
        stat_zero['train'].append(point['train'])
    return stat, stat_zero

def get_stat(pointfinder, model_name, architecture, t, method='arc_connect',
             beg_ind=1, end_ind=7):
    stat = {'test': [], 'train': []}
    model1 = architecture.base(num_classes=10, **architecture.kwargs)
    model2 = architecture.base(num_classes=10, **architecture.kwargs)
    for i in tqdm(range(beg_ind, end_ind)):
        model1.load_state_dict(torch.load('curves/'+model_name+'/curve'+str(i)+'/checkpoint-400.pt')['model_state'])
        model2.load_state_dict(torch.load('curves/'+model_name+'/curve'+str(i+1)+'/checkpoint-400.pt')['model_state'])
        finder = pointfinder(model1, model2, architecture)
        point = finder.find_point(t=t, method=method)
        stat['test'].append(point['test'])
        stat['train'].append(point['train'])
    return stat

def test_zeroing(pointfinder, model_name, architecture, beg_ind=1, end_ind=7):
    stat = {'test': [], 'train': []}
    model1 = architecture.base(num_classes=10, **architecture.kwargs)
    model2 = architecture.base(num_classes=10, **architecture.kwargs)
    for i in tqdm(range(beg_ind, end_ind)):
        model1.load_state_dict(torch.load('curves/'+model_name+'/curve'+str(i)+'/checkpoint-400.pt')['model_state'])
        model2.load_state_dict(torch.load('curves/'+model_name+'/curve'+str(i+1)+'/checkpoint-400.pt')['model_state'])
        finder = pointfinder(model1, model2, architecture)
        point = finder.test_zeroing()
        stat['test'].append(point['test'])
        stat['train'].append(point['train'])
    return stat

def find_path(time, finder, method='arc_connect'):
    path = {'test': [], 'train': []}
    for t in tqdm(time):
        point = finder.find_point(t=t, method=method)
        path['test'].append(100 - point['test'])
        path['train'].append(100 - point['train'])
    return path

def get_mean_svd(stat):
    train = np.array(stat['train'])
    test = np.array(stat['test'])
    return train.mean(), train.std(), test.mean(), test.std()


class PointFinderSimultaneous():
    def __init__(self, model1, model2, architecture):
        self.architecture = architecture
        self.model1 = model1
        self.model2 = model2
        self.weights_model1 = get_model_weights(model1)
        self.weights_model2 = get_model_weights(model2)
        self.depth = len(self.weights_model1)

    def find_point(self, t, method='arc_connect'):
        weights_model_new = []
        assert 0 <= t <= 1, 't must be between 0 and 1'
        for W1, W2 in zip(self.weights_model1, self.weights_model2):
            Wn = getattr(Connector(W1, W2), method)(t=t)[1]
            weights_model_new.append(Wn)

        m = get_model_from_weights(weights_model_new, self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}

        return out


class PointFinderStepWiseButterfly(PointFinderSimultaneous):
    def __init__(self, model1, model2, architecture):
        super().__init__(model1, model2, architecture)
        self.funcs1 = self.find_feature_maps(self.weights_model1, data)
        self.funcs2 = self.find_feature_maps(self.weights_model2, data)
        self.weights_adjusted = self.adjust_all_weights()

    def find_feature_maps(self, weights_model, data):
        """find feature maps for 2, 3 ,..., N-2 layers of network"""
        print('finding feature maps')
        funcs_list = []
        funcs = data
        for W in tqdm(list(weights_model)[:-2]):
            funcs = next_layer(W, data=funcs)
            funcs_list.append(funcs)
        return funcs_list

    def connect_butterflies(self, W10, W20, W11, W11b2,
                            t=0.5, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        Wn1 = getattr(Connector(W11.T, W11b2.T), method)(t=t)[1].T
        return Wn0, Wn1

    def adjust_weights(self, f1, f2, W):
        f_inv2 = np.linalg.pinv(f2.T)
        Wb2 = W @ f1.T @ f_inv2
        return Wb2

    def adjust_all_weights(self, ):
        """find intermidiate weights between \Theta^A and \Theta^B (see the the paper for the notation) """
        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1,
                                                 self.funcs2,
                                                 self.weights_model1[1:-1]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-1])
        return Wb2_list

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        W11b2 = self.weights_adjusted[layer + 1]
        Wn0, Wn1 = self.connect_butterflies(W10, W20, W11, W11b2,
                                            t=t, method=method)
        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}

        return out


class PointFinderStepWiseInverse(PointFinderStepWiseButterfly):
    def __init__(self, model1, model2, architecture):
        #         self.data = data
        super().__init__(model1, model2, architecture)

    def find_feature_maps(self, weights_model, data):
        """find feature maps of functions \theta_2^AB, ..., \theta_{N-1}^AB
        (see the the paper for the notation)"""

        print('finding feature maps')
        funcs_list = []
        funcs = data
        funcs_list.append(funcs)
        for W in tqdm(list(weights_model)[:-1]):
            funcs = next_layer(W, data=funcs)
            funcs_list.append(funcs)
        return funcs_list

    def connect(self, W10, W20, t, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        return Wn0

    def adjust_all_weights(self, ):
        """adjust weights of the first model (model1) according to feature maps of model1, model2
        in a way that resulting model will have the output of the model1 """

        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1[1:],
                                                 self.funcs2[1:],
                                                 self.weights_model1[1:]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-1])
        return Wb2_list

    def find_intermediate_point(self, t, layer, method):
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        Wn0 = self.connect(W10, W20, t=t, method=method)
        #         if t in [0, 1]:
        #             print('not computing')
        #             if t == 0:
        #                 Wn1 = W11
        #             else:
        #                 Wn1 = self.weights_adjusted[layer]
        #         else:
        f1 = self.funcs1[layer + 1]
        f2 = next_layer(Wn0, data=self.funcs2[layer])
        Wn1 = self.adjust_weights(f1, f2, W11)
        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]
        return weights_model_t

    def last_layer_interpolation(self, t, layer, method):
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        Wn0 = self.connect(W10, W20, t=t, method=method)
        weights_model_t = self.weights_model2[:layer] + [Wn0]
        return weights_model_t

    def find_point(self, t, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        assert layer <= self.depth + 0.1, 'the network is shot for this t value'

        if layer == self.depth:
            layer -= 1
            t = 1
        if layer == self.depth - 1:
            weights_model_t = self.last_layer_interpolation(t, layer, method=method)
        else:
            weights_model_t = self.find_intermediate_point(t, layer, method=method)
        m = get_model_from_weights(weights_model_t, self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}
        return out


class PointFinderWithBias():
    def __init__(self, model1, model2, architecture):
        self.architecture = architecture
        self.model1 = model1
        self.model2 = model2
        self.weights_model1 = self.get_model_weights(model1)
        self.weights_model2 = self.get_model_weights(model2)
        self.depth = len(self.weights_model1)

    def get_model_from_weights(self, W, B, architecture):
        model_sampled = architecture.base(num_classes=10, **architecture.kwargs)
        model_samples = np.array(W)  # .cpu().data.numpy()
        SIZE = model_sampled.middle_dim

        offset = 0
        for parameter in list(model_sampled.parameters())[:-1]:
            size = int(np.prod(parameter.size()) / SIZE)
            value = model_samples[:, offset:offset + size]
            if size == 10 or size == 1:
                value = value.T
            value = value.reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        list(model_sampled.parameters())[-1].data.copy_(torch.tensor(B))

        return model_sampled

    def get_b(self, model1, model2):
        B = []
        B.append(list(model1.parameters())[-1].cpu().data.numpy())
        B.append(list(model2.parameters())[-1].cpu().data.numpy())
        B = torch.tensor(np.array(B))
        return B

    def get_model_weights(self, model):
        p1 = list(model.parameters())[0].data.cpu().numpy()
        p2 = list(model.parameters())[1].data.cpu().numpy()
        p3 = list(model.parameters())[2].transpose(0, 1).data.cpu().numpy()
        samples = np.hstack([p1, p2[:, None], p3])
        return samples

    def find_point(self, t, method='arc_connect'):
        assert 0 <= t <= 1, 't must be between 0 and 1'
        weights_model_new = getattr(Connector(self.weights_model1, self.weights_model2), method)(t=t)[1]
        B = self.get_b(self.model1, self.model2)
        B = getattr(Connector(B[:1], B[1:]), method)(t=t)[1]
        #         print('B', B.shape)
        #         B = B.mean(0, keepdim=True)
        m = self.get_model_from_weights(weights_model_new, B[0], self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}

        return out

class PointFinderByOne(PointFinderWithBias):
    def __init__(self, model1, model2, architecture):
        super().__init__(model1, model2, architecture)

    def find_point(self, t, N, method='arc_connect'):
        assert 0 <= t <= 1, 't must be between 0 and 1'
        weights_model_new = getattr(Connector(self.weights_model1[N:N+1], self.weights_model2[N:N+1]), method)(t=t)[1]
        weights_model_new = list(self.weights_model2)[:N] + list(weights_model_new) + list(self.weights_model1[N+1:])
        B = self.get_b(self.model1, self.model2)
        B = getattr(Connector(B[:1], B[1:]), method)(t=t)[1]
        m = self.get_model_from_weights(weights_model_new, B[0], self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}

        return out


class PointFinderTransportation(PointFinderWithBias):
    def __init__(self, model1, model2, architecture):
        super().__init__(model1, model2, architecture)
        self.solve_optimal_transport_problem()
        # find bijection
        self.indices = np.argmax(self.G0, axis=-1)
        self.weights_model2_permuted = self.weights_model2[self.indices]

    def solve_optimal_transport_problem(self, ):
        self.n = len(self.weights_model1)
        self.a, self.b = np.ones((self.n,)) / self.n, np.ones((self.n,)) / self.n  # uniform distribution on samples
        # loss matrix
        self.M = ot.dist(self.weights_model1, self.weights_model2)
        self.M /= self.M.max()
        self.G0 = ot.emd(self.a, self.b, self.M)

    def find_point(self, t, method='arc_connect'):
        assert 0 <= t <= 1, 't must be between 0 and 1'
        weights_model_new = getattr(Connector(self.weights_model1, self.weights_model2_permuted), method)(t=t)[1]
        B = self.get_b(self.model1, self.model2)
        B = getattr(Connector(B[:1], B[1:]), method)(t=t)[1]
        m = self.get_model_from_weights(weights_model_new, B[0], self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}
        return out

    def test_zeroing(self, ):
        weights_model_new = self.weights_model2
        weights_model_new[0] = 0
        B = self.get_b(self.model1, self.model2)
        m = self.get_model_from_weights(weights_model_new, B[1], self.architecture)
        m.cuda();
        res = test_model(m, loaders, cuda=True)
        out = {'train': res[0]['accuracy'], 'test': res[1]['accuracy']}
        return out