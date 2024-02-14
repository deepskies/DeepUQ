import numpy as np
import torch.nn as nn
import torch
import math


def model_setup_DER(DER_type, DEVICE):
    # initialize the model from scratch
    if DER_type == "SDER":
        # model = models.de_no_var().to(device)
        DERLayer = models.SDERLayer

        # initialize our loss function
        lossFn = models.loss_sder
    else:
        # model = models.de_var().to(device)
        DERLayer = models.DERLayer
        # initialize our loss function
        lossFn = models.loss_der

    # from https://github.com/pasteurlabs/unreasonable_effective_der
    # /blob/main/x3_indepth.ipynb
    model = torch.nn.Sequential(models.Model(4), DERLayer())
    model = model.to(DEVICE)
    return model, lossFn


def model_setup_DE(DE_type, DEVICE):
    # initialize the model from scratch

    if DE_type == "no_var_loss":
        model = models.de_no_var().to(DEVICE)
        # initialize our optimizer and loss function
        lossFn = torch.nn.MSELoss(reduction="mean")
    else:
        model = models.de_var().to(DEVICE)
        # initialize our optimizer and loss function
        lossFn = torch.nn.GaussianNLLLoss(full=False,
                                            eps=1e-06,
                                            reduction="mean")
    return model, lossFn


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
        self.ln_4 = nn.Linear(100, 1)
        # this last dim needs to be 2 if using the GaussianNLLoss

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
        self.ln_4 = nn.Linear(100, 2)
        # this last dim needs to be 2 if using the GaussianNLLoss

    def forward(self, x):
        x = self.drop1(self.act1(self.ln_1(x)))
        x = self.drop2(self.act2(self.ln_2(x)))
        x = self.drop3(self.act3(self.ln_3(x)))
        x = self.ln_4(x)
        return x


# This following is from PasteurLabs -
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
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    var = beta / nu

    # define aleatoric and epistemic uncert
    u_al = np.sqrt(
        beta.detach().numpy()
        * (1 + nu.detach().numpy())
        / (alpha.detach().numpy() * nu.detach().numpy())
    )
    u_ep = 1 / np.sqrt(nu.detach().numpy())

    return torch.mean(torch.log(var)
                      + (1.0 + coeff * nu) * error**2 / var), u_al, u_ep
