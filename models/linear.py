import math
import torch.nn as nn

import curves

__all__ = [
    'Linear', 'LinearMNIST', 'LogRegression', 'LinearOneLayer', 'LinearOneLayerCF'
]


class LinearBase(nn.Module):
    def __init__(self, num_classes, in_dim):
        super(LinearBase, self).__init__()
        self.fc1_part = nn.Sequential(
            nn.Linear(in_dim, 2*3072),
            nn.ReLU(True),
            nn.Linear(2*3072, 2*3072),
            nn.ReLU(True),
            nn.Linear(2*3072, 1152),
            nn.ReLU(True),
        )
        self.fc2_part = nn.Sequential(
            nn.Linear(1152, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1_part(x)
        x = self.fc2_part(x)
        return x


class LinearCurve(nn.Module):
    def __init__(self, num_classes, fix_points, in_dim):
        super(LinearCurve, self).__init__()
        self.fc1 = curves.Linear(in_dim, 2*3072, fix_points=fix_points)
        self.relu1 = nn.ReLU(True)

        self.fc2 = curves.Linear(2*3072, 2*3072, fix_points=fix_points)
        self.relu2 = nn.ReLU(True)

        self.fc3 = curves.Linear(2*3072, 1152, fix_points=fix_points)
        self.relu3 = nn.ReLU(True)

        self.fc4 = curves.Linear(1152, 1000, fix_points=fix_points)
        self.relu4 = nn.ReLU(True)

        self.fc5 = curves.Linear(1000, 1000, fix_points=fix_points)
        self.relu5 = nn.ReLU(True)

        self.fc6 = curves.Linear(1000, num_classes, fix_points=fix_points)


    def forward(self, x, coeffs_t):

        x = x.view(x.size(0), -1)

        x = self.fc1(x, coeffs_t)
        x = self.relu1(x)

        x = self.fc2(x, coeffs_t)
        x = self.relu2(x)

        x = self.fc3(x, coeffs_t)
        x = self.relu3(x)

        x = self.fc4(x, coeffs_t)
        x = self.relu4(x)

        x = self.fc5(x, coeffs_t)
        x = self.relu5(x)

        x = self.fc6(x, coeffs_t)

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
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         n = m.in_features
        #         m.weight.data.normal_(0, math.sqrt(1. / n))
        #         m.bias.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         n = m.in_features
        #         m.weight.data.normal_(0, math.sqrt(1. / n))
        #         m.bias.data.zero_()

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
        'in_dim': 3072
    }


class LinearMNIST:
    base = LinearBase
    curve = LinearCurve
    kwargs = {
        'in_dim': 784
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

class LinearOneLayerCF:
    base = LinearOneLayerBase
    curve = LinearOneLayerCurve
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 10000
    }