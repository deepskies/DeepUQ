# Contains modules to analyze the output checkpoints
# from a trained model and make plots for the paper
import torch


class AggregateCheckpoints:
    """A class for loading and processing model checkpoints for deep ensemble
    (DE) and deep evidential regression (DER) models.

    This class provides methods for loading model checkpoints and extracting
    validation metrics, including uncertainty metrics for DER models.

    Methods:
        load_checkpoint(model_name, inject_type, data_dim, noise,
                        epoch, device, path="models/", BETA=0.5, nmodel=1,
                        COEFF=0.5, loss="SDER", load_rs_chk=False, rs=42,
                        load_nh_chk=False, nh=64):
            Loads a PyTorch model checkpoint based on the specified model type
            (DE or DER) and configuration parameters. Supports customizations
            for random seeds, hidden layers, and loss type.

        ep_al_checkpoint_DE(checkpoint):
            Extracts mean and variance validation metrics from a loaded DE
            model checkpoint.

        ep_al_checkpoint_DER(checkpoint):
            Extracts epistemic and aleatoric uncertainty metrics (mean and
            standard deviation) from a loaded DER model checkpoint.
    """

    def load_checkpoint(
        self,
        model_name,
        inject_type,
        data_dim,
        noise,
        epoch,
        device,
        path="models/",
        BETA=0.5,
        nmodel=1,
        COEFF=0.5,
        loss="SDER",
        load_rs_chk=False,
        rs=42,
        load_nh_chk=False,
        nh=64,
    ):
        """Loads a PyTorch model checkpoint from a .pt file, constructing the
        file name based on model type (DE or DER) and other configuration
        parameters.

        Parameters:
            model_name (str): Name of the model to load (e.g., 'DER', 'DE').
            inject_type (str): Type of data injection used in the model.
            data_dim (str): Dimensionality of the data (e.g., '0D', '2D').
            noise (float): Level of noise applied to the model.
            epoch (int): The epoch number to load the checkpoint for.
            device (str): The device ('cuda' or 'cpu') to load the model onto.
            path (str): Directory path to the model checkpoints
            (default: "models/").
            BETA (float): Beta value for DE models (default: 0.5).
            nmodel (int): Number of models in the ensemble (default: 1).
            COEFF (float): Coefficient for the DER model (default: 0.5).
            loss (str): Type of loss used for DER models (e.g., 'SDER', 'DER').
            load_rs_chk (bool): Flag to indicate if random seed checkpoint is
            used (default: False).
            rs (int): Random seed value if load_rs_chk is True (default: 42).
            load_nh_chk (bool): Flag to indicate if a specific hidden layer
            configuration is used (default: False).
            nh (int): Number of hidden units if load_nh_chk is True
            (default: 64).

        Returns:
            dict: The loaded checkpoint containing model weights and
            additional data.
        """
        print(model_name)
        if model_name[0:3] == "DER":
            file_name = (
                str(path)
                + f"{model_name}_{inject_type}_{data_dim}"
                + f"_noise_{noise}_loss_{loss}_COEFF_{COEFF}_epoch_{epoch}"
            )
            if load_rs_chk:
                file_name += f"_rs_{rs}"
            if load_nh_chk:
                file_name += f"_n_hidden_{nh}"
            file_name += ".pt"
        elif model_name[0:2] == "DE":
            file_name = (
                str(path)
                + f"{model_name}_{inject_type}_{data_dim}"
                f"_noise_{noise}_beta_{BETA}_nmodel_{nmodel}_epoch_{epoch}.pt"
            )
        print("loading this chk", file_name)
        checkpoint = torch.load(file_name, map_location=device)
        return checkpoint

    def ep_al_checkpoint_DE(self, checkpoint):
        """Extracts mean and variance validation metrics from a loaded DE model
        checkpoint.

        Parameters:
            checkpoint (dict): The loaded DE model checkpoint.

        Returns:
            tuple: A tuple containing mean validation and variance validation
            metrics.
        """
        # Extract additional information
        # loaded_epoch = checkpoint.get("epoch", None)
        mean_validation = checkpoint.get("valid_mean", None).detach().numpy()
        var_validation = checkpoint.get("valid_var", None).detach().numpy()
        return mean_validation, var_validation

    def ep_al_checkpoint_DER(self, checkpoint):
        """Extracts mean and variance validation metrics from a loaded DER
        model checkpoint.

        Parameters:
            checkpoint (dict): The loaded DER model checkpoint.

        Returns:
            tuple: A tuple containing mean validation and variance validation
            metrics.
        """
        mean_u_ep = checkpoint.get("mean_u_ep_validation", None)
        mean_u_al = checkpoint.get("mean_u_al_validation", None)
        std_u_ep = checkpoint.get("std_u_ep_validation", None)
        std_u_al = checkpoint.get("std_u_al_validation", None)

        return mean_u_ep, mean_u_al, std_u_ep, std_u_al
