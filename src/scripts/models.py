import numpy as np
import torch.nn as nn
import torch
import math

# tensorflow sucks
# build a similar thing in pytorch


class de_no_var(nn.Module):
    def __init__(self):
        super().__init__()
        drop_percent = 0.1
        self.ln_1 = nn.Linear(3, 100)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop_percent)
        self.ln_2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop_percent)
        self.ln_3 = nn.Linear(100, 100)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(drop_percent)
        self.ln_4 = nn.Linear(100,1) # needs to be 2 if using the GaussianNLLoss

    def forward(self, x):
        x = self.drop1(self.act1(self.ln_1(x)))
        x = self.drop2(self.act2(self.ln_2(x)))
        x = self.drop3(self.act3(self.ln_3(x)))
        x = self.ln_4(x)
        return x


class de_var(nn.Module):
    def __init__(self):
        super().__init__()
        drop_percent = 0.1
        self.ln_1 = nn.Linear(3, 100)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop_percent)
        self.ln_2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop_percent)
        self.ln_3 = nn.Linear(100, 100)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(drop_percent)
        self.ln_4 = nn.Linear(100,2) # needs to be 2 if using the GaussianNLLoss

    def forward(self, x):
        x = self.drop1(self.act1(self.ln_1(x)))
        x = self.drop2(self.act2(self.ln_2(x)))
        x = self.drop3(self.act3(self.ln_3(x)))
        x = self.ln_4(x)
        return x

## in numpyro, you must specify number of sampling chains you will use upfront

# words of wisdom from Tian Li and crew:
# on gpu, don't use conda, use pip install
# HMC after SBI to look at degeneracies between params
# different guides (some are slower but better at showing degeneracies)

# This is from PasteurLabs - 
# https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py

class Model(nn.Module):
    def __init__(self, n_output, n_hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.model(x)


class DERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class SDERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nu + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


def loss_der(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    omega = 2.0 * beta * (1.0 + nu)

    return torch.mean(
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error**2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * nu + alpha)
    )


def loss_sder(y, y_pred, coeff):
    gamma, nu, _, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    var = beta / nu

    return torch.mean(torch.log(var) + (1. + coeff * nu) * error**2 / var)
