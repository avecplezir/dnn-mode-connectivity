import math
import torch.nn as nn

import curves

__all__ = [
    'Linear', 'LinearMNIST',
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