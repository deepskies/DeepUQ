# Contains modules used to prepare a dataset
# with varying noise properties
import numpy as np
import pickle
import torch
import torch.nn as nn
import math


class ModelLoader:
    """A class for saving and loading machine learning models in pickle format.

    This class provides methods to save a trained model to a specified location
    and to load a model from a saved file. It utilizes Python's pickle module
    to serialize and deserialize model objects.

    Methods:
        save_model_pkl(path, model_name, posterior):
            Saves the provided model object as a pickle file.

        load_model_pkl(path, model_name):
            Loads a model object from a specified pickle file.

    Attributes:
        None
    """

    def save_model_pkl(self, path, model_name, model):
        """
        Save the pkl'ed saved posterior model

        :param path: Location to save the model
        :param model_name: Name of the model
        :param posterior: Model object to be saved
        """
        file_name = path + model_name + ".pkl"
        with open(file_name, "wb") as file:
            pickle.dump(model, file)

    def load_model_pkl(self, path, model_name):
        """
        Load the pkl'ed saved model

        :param path: Location to load the model from
        :param model_name: Name of the model
        :return: Loaded model object that can be used with the predict function
        """
        print(path)
        with open(path + model_name + ".pkl", "rb") as file:
            model = pickle.load(file)
        return model


class DERLayer(nn.Module):
    """A layer that processes inputs to produce parameters for the DER loss
    function.

    This layer takes input features and transforms them into parameters
    required for the DER (Deep Ensemble Regression) model. The output includes
    gamma, nu, alpha, and beta parameters, with nu, alpha, and beta enforced
    to be positive using the softplus function.

    Methods:
        forward(x):
            Defines the forward pass through the layer, producing the DER
            parameters.

    Attributes:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass to compute DER parameters.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features),
                              where n_features must be at least 4.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 4) containing the
                          parameters:
                          - gamma
                          - nu (softplus applied)
                          - alpha (softplus applied + 1)
                          - beta (softplus applied)
        """
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class SDERLayer(nn.Module):
    """A layer that processes inputs to produce parameters for the SDER loss
    function.

    This layer takes input features and transforms them into parameters
    required for the SDER (Simplified Deep Ensemble Regression) model. The
    output includes the gamma, nu, alpha, and beta parameters, where nu and
    beta are enforced to be positive using the softplus function.

    Methods:
        forward(x):
            Defines the forward pass through the layer, producing the SDER
            parameters.

    Attributes:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass to compute SDER parameters.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, n_features),
                              where n_features must be at least 4.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 4) containing the
                          parameters:
                          - gamma
                          - nu (softplus applied)
                          - alpha (nu + 1)
                          - beta (softplus applied)
        """
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nu + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class ConvLayers(nn.Module):
    """A series of convolutional layers followed by average pooling and
    flattening.

    This class implements a convolutional neural network architecture designed
    for processing 2D input data, such as images. It consists of multiple
    convolutional layers with ReLU activation functions and average pooling
    layers to reduce spatial dimensions.

    Methods:
        forward(x):
            Defines the forward pass through the convolutional layers, applying
            ReLU activations and average pooling.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        pool1 (nn.AvgPool2d): First average pooling layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        pool2 (nn.AvgPool2d): Second average pooling layer.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        conv5 (nn.Conv2d): Fifth convolutional layer.
        flatten (nn.Flatten): Layer to flatten the output from the
            convolutions.
    """

    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=7, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=7, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(5, 5, kernel_size=5, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(5, 10, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Forward pass through the convolutional layers.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels,
                              height, width). If the input is 3D
                              (batch_size, height, width), a channel
                              dimension will be added.

        Returns:
            torch.Tensor: The flattened output tensor after passing through the
                          convolutional and pooling layers.
        """
        assert x.dim() != 2, (
            f"should enter here with a dimension of at least 3, "
            f"{x.dim()}, {x.shape}"
        )
        if x.dim() == 3:  # Check if the input is of shape (batchsize, 32, 32)
            # Add channel dimension, becomes (batchsize, 1, 32, 32)
            x = x.unsqueeze(1)
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


class MuVarLayer(nn.Module):
    """A layer that extracts and transforms mean and variance from input
    tensors.

    This layer takes an input tensor, where the first column represents the
    mean (mu) and the second column represents the variance (var). The
    variance is passed through a softplus activation function to ensure it
    remains positive.

    Methods:
        forward(x):
            Defines the forward pass of the layer, returning the mean and
            positive variance.

    Attributes:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass through the MuVarLayer.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 2), where:
                - x[:, 0] corresponds to the mean (mu).
                - x[:, 1] corresponds to the variance (var).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, 2) containing the
                          mean and positive variance.
        """
        mu = x[:, 0]
        # softplus enforces positivity
        var = nn.functional.softplus(x[:, 1])
        # var = x[:, 1]
        return torch.stack((mu, var), dim=1)


# This following is from PasteurLabs -
# https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py


class Model(nn.Module):
    """A simple feedforward neural network model.

    This class defines a neural network with two hidden layers, each followed
    by a ReLU activation function. The network takes input features, processes
    them through the hidden layers, and produces output predictions.

    Parameters:
        n_input (int, optional): The number of input features. Defaults to 3.
        n_hidden (int, optional): The number of hidden units in each hidden
            layer. Defaults to 64.
        n_output (int, optional): The number of output features or predictions.
            Defaults to 4.
    """

    def __init__(self, n_input=3, n_hidden=64, n_output=4):
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


def model_setup_DER(loss_type, DEVICE, n_hidden=64, data_type="0D"):
    """Set up the Deep Evidential Regression (DER) model based on the
    specified loss type and data type.

    This function initializes a model and a loss function for DER. Depending
    on the specified loss type, it configures different layers and loss
    functions to handle both 0D and 2D data.

    Parameters:
        loss_type (str): The type of loss to be used. Options include:
                         - "SDER": Simplified Deep Evidential Regression loss.
                         - "DER": Deep Evidential Regression loss.
        DEVICE (torch.device): The device on which the model will be allocated
                               (CPU or GPU).
        n_hidden (int, optional): The number of hidden units in the model.
                                  Defaults to 64.
        data_type (str, optional): The type of data being processed.
                                   Options include:
                                    - "0D": for 0D data.
                                    - "2D": for 2D data. Defaults to "0D".

    Returns:
        tuple: A tuple containing:
            - torch.nn.Module: The initialized model.
            - callable: The loss function to be used during training.
    """
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
            Model(
                n_hidden=n_hidden,
                n_input=360,
                n_output=4,  # 405
            ),  # Adjust input size according to the flattened output size
            Layer(),
        )
    elif data_type == "0D":
        # from https://github.com/pasteurlabs/unreasonable_effective_der
        # /blob/main/x3_indepth.ipynb
        model = torch.nn.Sequential(
            Model(n_hidden=n_hidden, n_input=3, n_output=4),
            Layer(),
        )
    model = model.to(DEVICE)
    return model, lossFn


def model_setup_DE(loss_type, DEVICE, n_hidden=64, data_type="0D"):
    """Set up the Deep Evidential Regression model based on the specified loss
    type and data type.

    This function initializes a model and a loss function for Deep Evidential
    Regression (DER). Depending on the specified loss type, it configures
    different layers and loss functions.

    Parameters:
        loss_type (str): The type of loss to be used. Options include:
                         - "bnll_loss": Beta Negative Log-Likelihood loss.
                         - it is possible to retrieve other loss functions,
                         namely NLL and MSE by setting beta = 0 and 1,
                         respectively
        DEVICE (torch.device): The device on which the model will be allocated
                               (CPU or GPU).
        n_hidden (int, optional): The number of hidden units in the model.
                                  Defaults to 64.
        data_type (str, optional): The type of data being processed.
                                   Options include:
                                    - "0D": for 0D data.
                                    - "2D": for 2D data. Defaults to "0D".

    Returns:
        tuple: A tuple containing:
            - torch.nn.Module: The initialized model.
            - callable: The loss function to be used during training.
    """
    if loss_type == "bnll_loss":
        Layer = MuVarLayer
        lossFn = loss_bnll
    if data_type == "2D":
        # Define the convolutional layers
        conv_layers = ConvLayers()
        # Initialize the rest of the model
        model = torch.nn.Sequential(
            conv_layers,
            Model(
                n_input=360, n_hidden=n_hidden, n_output=2
            ),  # Adjust input size according to the flattened output size
            Layer(),
        )
    elif data_type == "0D":
        # from https://github.com/pasteurlabs/unreasonable_effective_der
        # /blob/main/x3_indepth.ipynb
        model = torch.nn.Sequential(
            Model(n_input=3, n_hidden=n_hidden, n_output=2),
            Layer(),
        )
    model = model.to(DEVICE)
    return model, lossFn


def loss_der(y, y_pred, coeff):
    """Compute the Deep Evidential Regression (DER) loss.

    This function calculates the DER loss using predicted values and target
    values, accounting for aleatoric and epistemic uncertainties. The loss is
    formulated based on the parameters provided in the input tensor `y`.

    Parameters:
        y (torch.Tensor): A tensor containing the target values, with the
                          following structure: [gamma, nu, alpha, beta].
                          - gamma: target value
                          - nu: scaling factor for the variance
                          - alpha: shape parameter for the distribution
                          - beta: scale parameter for the distribution
        y_pred (torch.Tensor): A tensor containing the predicted values
                               corresponding to the target values.
        coeff (float): A coefficient that modifies the impact of the
                       uncertainty term in the loss calculation.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The computed DER loss averaged over the samples.
            - numpy.ndarray: The aleatoric uncertainty derived from the input
              parameters.
            - numpy.ndarray: The epistemic uncertainty derived from the input
              parameters.
    """
    gamma, nu, alpha, beta = (
        y[:, 0],
        y[:, 1],
        y[:, 2],
        y[:, 3],
    )
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


def loss_sder(y, y_pred, coeff):
    """Compute the simplified Deep Evidential Regression (DER) loss,
    from Meinert+2023 "The Unreasonable Effectiveness of Deep Evidential
    Regression".

        This function calculates the loss based on the predictions and true
        values, incorporating both aleatoric and epistemic uncertainties.
        It uses the parameters derived from the true values to evaluate the
        error and the variance of the predictions.

        Parameters:
            y (torch.Tensor): A tensor containing the true values, where each
                              row corresponds to a sample and the columns
                              represent the parameters
                              (gamma, nu, alpha, beta).
            y_pred (torch.Tensor): A tensor of predicted values corresponding
                                   to the true values.
            coeff (float): A coefficient used to weigh the contribution of the
                           epistemic uncertainty.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss value, incorporating
                                       the variances and errors.
                - u_al (np.ndarray): The computed aleatoric uncertainty based
                                     on the inputs.
                - u_ep (np.ndarray): The computed epistemic uncertainty based
                                     on the inputs.
    """
    gamma, nu, alpha, beta = (
        y[:, 0],
        y[:, 1],
        y[:, 2],
        y[:, 3],
    )
    error = gamma - y_pred
    var = beta / nu

    # define aleatoric and epistemic uncert
    u_al = np.sqrt(
        (beta.detach().numpy() * (1 + nu.detach().numpy()))
        / (alpha.detach().numpy() * nu.detach().numpy())
    )
    u_ep = 1 / np.sqrt(nu.detach().numpy())

    return (
        torch.mean(torch.log(var) + (1.0 + coeff * nu) * error**2 / var),
        u_al,
        u_ep,
    )


def loss_bnll(mean, variance, target, beta):
    """Compute the Beta Negative Log-Likelihood (BNLL) loss.

    From Martius lab (https://github.com/martius-lab/beta-nll) and
    Seitzer+2020.

    This function calculates the loss based on the mean and variance
    predictions relative to the target values, incorporating a weighting
    factor for the variance.

    Parameters:
        mean (torch.Tensor): A tensor containing the predicted means for each
                             sample.
        variance (torch.Tensor): A tensor containing the predicted variances
                                 for each sample.
        target (torch.Tensor): A tensor containing the true target values
                               corresponding to the predictions.
        beta (float): A weighting parameter that modifies the influence of the
                      variance on the loss.

    Returns:
        torch.Tensor: The computed BNLL loss averaged over the number of
        samples.
    """
    loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())
    if beta > 0:
        loss = loss * (variance.detach() ** beta)
    return loss.sum(axis=-1) / len(mean)
