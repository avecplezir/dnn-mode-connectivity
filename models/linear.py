import math
import torch.nn as nn

import curves

__all__ = [
    'Linear', 'LinearMNIST', 'LogRegression', 'LinearOneLayer', 'LinearOneLayerCF', 'LinearNoBias',
    'Linear3NoBias', 'LinearOneLayer60k', 'Linear2NoBias', 'Linear3NoBiasW',
    'LinearOneLayer100', 'LinearOneLayer500', 'LinearOneLayer1000',
    'Linear5NoBias', 'Linear7NoBias', 'Linear9NoBias',
]


class LinearBase(nn.Module):
    def __init__(self, num_classes, in_dim, middle_dim,  bias=True):
        super(LinearBase, self).__init__()

        self.dims = [in_dim] + list(middle_dim) + [num_classes]
        self.linear_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        for i in range(len(self.dims)-2):
            self.linear_layers.append(nn.Linear(self.dims[i], self.dims[i+1], bias=bias))
            self.relu_layers.append(nn.ReLU(True))

        self.linear_layers.append(nn.Linear(self.dims[-2], self.dims[-1], bias=bias))

    def last_layers(self, x, N=-1):

        for i in range(N, len(self.dims) - 2):
            x = self.linear_layers[i](x)
            x = self.relu_layers[i](x)

        x = self.linear_layers[-1](x)

        return x

    def forward(self, x, N=-1):
        x = x.view(x.size(0), -1)
        for i in range(len(self.dims) - 2):
            x = self.linear_layers[i](x)
            x = self.relu_layers[i](x)
            if N == i:
                return x

        x = self.linear_layers[-1](x)

        return x


class LinearCurve(nn.Module):
    def __init__(self, num_classes, fix_points, in_dim, middle_dim,  bias=True):
        super(LinearCurve, self).__init__()

        self.dims = [in_dim]+list(middle_dim) + [num_classes]
        self.linear_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        for i in range(len(self.dims)-2):
            # print(self.dims[i], self.dims[i+1])
            self.linear_layers.append(curves.Linear(self.dims[i], self.dims[i+1], fix_points=fix_points, bias=bias))
            self.relu_layers.append(nn.ReLU(True))

        self.linear_layers.append(curves.Linear(self.dims[-2], self.dims[-1], fix_points=fix_points, bias=bias))


    def forward(self, x, coeffs_t):

        x = x.view(x.size(0), -1)

        for i in range(len(self.dims) - 2):
            x = self.linear_layers[i](x, coeffs_t)
            x = self.relu_layers[i](x)

        x = self.linear_layers[-1](x, coeffs_t)

        return x

class LogRegressionBase(nn.Module):
    def __init__(self, num_classes, in_dim):
        super(LogRegressionBase, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LogRegressionCurve(nn.Module):
    def __init__(self, num_classes, fix_points, in_dim):
        super(LogRegressionCurve, self).__init__()
        self.fc = curves.Linear(in_dim, num_classes, fix_points=fix_points)

    def forward(self, x, coeffs_t):

        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)
        return x

class LinearOneLayerBase(nn.Module):
    def __init__(self, num_classes, in_dim, middle_dim):
        super(LinearOneLayerBase, self).__init__()
        self.in_dim = in_dim
        self.middle_dim = middle_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, middle_dim),
            nn.ReLU(True),
            nn.Linear(middle_dim, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LinearOneLayerCurve(nn.Module):
    def __init__(self, num_classes, fix_points, in_dim, middle_dim):
        super(LinearOneLayerCurve, self).__init__()
        self.fc1 = curves.Linear(in_dim, middle_dim, fix_points=fix_points)
        self.relu1 = nn.ReLU(True)
        self.fc2 = curves.Linear(middle_dim,num_classes, fix_points=fix_points)

    def forward(self, x, coeffs_t):

        x = x.view(x.size(0), -1)

        x = self.fc1(x, coeffs_t)
        x = self.relu1(x)
        x = self.fc2(x, coeffs_t)
        return x


class Linear:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': True,
        'middle_dim': [2*3072, 2*3072, 1152, 1000, 1000],
    }

class LinearNoBias:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2 * 3072, 1152, 1000, 1000],
    }

class Linear3NoBias:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000],
    }

class Linear5NoBias:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000, 1000, 1000],
    }

class Linear7NoBias:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000, 1000, 1000, 1000, 1000],
    }

class Linear9NoBias:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000, 1000, 1000, 1000, 1000, 1000, 1000],
    }

class Linear3NoBiasW:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2 * 3072],
    }

class Linear2NoBias:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072],
    }


class LinearMNIST:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 784,
        'bias': True,
        'middle_dim': [2 * 3072, 2 * 3072, 1152, 1000, 1000],
    }


class LogRegression:
    base = LogRegressionBase
    curve = LogRegressionCurve
    kwargs = {
        'in_dim': 784
    }

class LinearOneLayer:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 784,
        'middle_dim': 2000
    }

class LinearOneLayer100:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 100
    }

class LinearOneLayer500:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 500
    }

class LinearOneLayer1000:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 1000
    }

class LinearOneLayer60k:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 60000
    }

class LinearOneLayerCF:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 2000
    }