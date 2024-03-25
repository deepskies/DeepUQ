import numpy as np
import torch.nn as nn
import torch
import math


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


def model_setup_DER(DER_type, DEVICE):
    # initialize the model from scratch
    if DER_type == "SDER":
        Layer = SDERLayer
        # initialize our loss function
        lossFn = loss_sder
    if DER_type == "DER":
        Layer = DERLayer
        # initialize our loss function
        lossFn = loss_der

    # from https://github.com/pasteurlabs/unreasonable_effective_der
    # /blob/main/x3_indepth.ipynb
    model = torch.nn.Sequential(Model(4), Layer())
    model = model.to(DEVICE)
    return model, lossFn


class MuVarLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mu = x[:, 0]
        # softplus enforces positivity
        var = nn.functional.softplus(x[:, 1])
        # var = x[:, 1]
        return torch.stack((mu, var), dim=1)


def model_setup_DE(loss_type, DEVICE):  # , INIT_LR=0.001):
    # initialize the model from scratch
    if loss_type == "no_var_loss":
        # model = de_no_var().to(DEVICE)
        lossFn = torch.nn.MSELoss(reduction="mean")
    if loss_type == "var_loss":
        # model = de_var().to(DEVICE)
        Layer = MuVarLayer
        lossFn = torch.nn.GaussianNLLLoss(full=False,
                                          eps=1e-06,
                                          reduction="mean")
    if loss_type == "bnll_loss":
        # model = de_var().to(DEVICE)
        Layer = MuVarLayer
        lossFn = loss_bnll
    model = torch.nn.Sequential(Model(2), Layer())
    model = model.to(DEVICE)
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


def loss_der(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    omega = 2.0 * beta * (1.0 + nu)

    # define aleatoric and epistemic uncert
    u_al = np.sqrt(
        beta.detach().numpy()
        * (1 + nu.detach().numpy())
        / (alpha.detach().numpy() * nu.detach().numpy())
    )
    u_ep = 1 / np.sqrt(nu.detach().numpy())
    return (
        torch.mean(
            0.5 * torch.log(math.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(error**2 * nu + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
            + coeff * torch.abs(error) * (2.0 * nu + alpha)
        ),
        u_al,
        u_ep,
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

    return torch.mean(torch.log(var) + (1.0 + coeff * nu) * error**2 / var), \
        u_al, u_ep


# from martius lab
# https://github.com/martius-lab/beta-nll
# and Seitzer+2020


def loss_bnll(mean, variance, target, beta):  # beta=0.5):
    """Compute beta-NLL loss

    :param mean: Predicted mean of shape B x D
    :param variance: Predicted variance of shape B x D
    :param target: Target of shape B x D
    :param beta: Parameter from range [0, 1] controlling relative
        weighting between data points, where `0` corresponds to
        high weight on low error points and `1` to an equal weighting.
    :returns: Loss per batch element of shape B
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())
    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    return loss.sum(axis=-1)


'''
def get_loss(transform, beta=None):
    if beta:
        def beta_nll_loss(targets, outputs, beta=beta):
            """Compute beta-NLL loss
            """
            mu = outputs[..., 0:1]
            var = transform(outputs[..., 1:2])
            loss = (K.square((targets - mu)) / var + K.log(var))
            loss = loss * K.stop_gradient(var) ** beta
            return loss
        return beta_nll_loss
    else:
        def negative_log_likelihood(targets, outputs):
            """Calculate the negative loglikelihood."""
            mu = outputs[..., 0:1]
            var = transform(outputs[..., 1:2])
            y = targets[..., 0:1]
            loglik = - K.log(var) - K.square((y - mu)) / var
            return - loglik
    return negative_log_likelihood
'''
