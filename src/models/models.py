# Contains modules used to prepare a dataset
# with varying noise properties
import numpy as np
import pickle
import torch
import torch.nn as nn
import math


class ModelLoader:
    def save_model_pkl(self, path, model_name, posterior):
        """
        Save the pkl'ed saved posterior model

        :param path: Location to save the model
        :param model_name: Name of the model
        :param posterior: Model object to be saved
        """
        file_name = path + model_name + ".pkl"
        with open(file_name, "wb") as file:
            pickle.dump(posterior, file)

    def load_model_pkl(self, path, model_name):
        """
        Load the pkl'ed saved posterior model

        :param path: Location to load the model from
        :param model_name: Name of the model
        :return: Loaded model object that can be used with the predict function
        """
        print(path)
        with open(path + model_name + ".pkl", "rb") as file:
            posterior = pickle.load(file)
        return posterior

    def predict(input, model):
        """

        :param input: loaded object used for inference
        :param model: loaded model
        :return: Prediction
        """
        return 0


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

class ConvLayers(nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        # a little strange = # of filters, usually goes from small to large
        # double check on architecture decisions
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(10, 5, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # print('input shape', x.shape)
        if x.dim() == 3:  # Check if the input is of shape (batchsize, 32, 32)
            x = x.unsqueeze(1)  # Add channel dimension, becomes (batchsize, 1, 32, 32)
        # print('shape after potential unsqeeze', x.shape)
        x = nn.functional.relu(self.conv1(x))
        # print('shape after conv1', x.shape)
        x = nn.functional.relu(self.conv2(x))
        # print('shape after conv2', x.shape)
        x = self.pool1(x)
        # print('shape after pool1', x.shape)
        x = nn.functional.relu(self.conv3(x))
        # print('shape after conv3', x.shape)
        x = self.pool2(x)
        # print('shape after pool2', x.shape)
        x = nn.functional.relu(self.conv4(x))
        # print('shape after conv4', x.shape)
        x = nn.functional.relu(self.conv5(x))
        # print('shape after conv5', x.shape)
        x = self.flatten(x)
        # print('shape after flatten', x.shape)
        return x


def model_setup_DER(loss_type,
                    DEVICE,
                    n_hidden=64,
                    data_type="0D"):
    # initialize the model from scratch
    if loss_type == "SDER":
        Layer = SDERLayer
        # initialize our loss function
        lossFn = loss_sder
    if loss_type == "DER":
        Layer = DERLayer
        # initialize our loss function
        lossFn = loss_der
    if data_type == "2D":
        # Define the convolutional layers
        conv_layers = ConvLayers()
        
        # Initialize the rest of the model
        model = torch.nn.Sequential(
            conv_layers,
            Model(n_hidden=n_hidden, n_input=405, n_output=4),  # Adjust input size according to the flattened output size
            Layer()
        )
    elif data_type == "0D":
        # from https://github.com/pasteurlabs/unreasonable_effective_der
        # /blob/main/x3_indepth.ipynb
        model = torch.nn.Sequential(Model(
            n_hidden=n_hidden, n_input=3, n_output=4), Layer())
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


def model_setup_DE(loss_type,
                   DEVICE,
                   n_hidden=64,
                   data_type="0D"):
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
    if data_type == "2D":
        # Define the convolutional layers
        conv_layers = ConvLayers()
        # Initialize the rest of the model
        model = torch.nn.Sequential(
            conv_layers,
            Model(n_hidden=n_hidden, n_input=405, n_output=2),  # Adjust input size according to the flattened output size
            Layer()
        )
    elif data_type == "0D":
        # from https://github.com/pasteurlabs/unreasonable_effective_der
        # /blob/main/x3_indepth.ipynb
        model = torch.nn.Sequential(Model(
            n_hidden=n_hidden, n_input=3, n_output=2), Layer())
    model = model.to(DEVICE)
    return model, lossFn


# This following is from PasteurLabs -
# https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py


class Model(nn.Module):
    def __init__(self,
                 n_output=4,
                 n_hidden=64,
                 n_input=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
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
    w_st = torch.sqrt(beta * (1 + nu) / (alpha * nu))
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
            + (coeff * torch.abs(error / w_st) * (2.0 * nu + alpha))
        ),
        u_al,
        u_ep,
    )


# simplified DER loss (from Meinert)
def loss_sder(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    var = beta / nu

    # define aleatoric and epistemic uncert
    u_al = np.sqrt(
        (beta.detach().numpy() * (1 + nu.detach().numpy()))
        / (alpha.detach().numpy() * nu.detach().numpy())
    )
    u_ep = 1 / np.sqrt(nu.detach().numpy())

    return torch.mean(torch.log(var) +
                      (1.0 + coeff * nu) * error**2 / var), u_al, u_ep


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
    return loss.sum(axis=-1) / len(mean)
