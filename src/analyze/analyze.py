# Contains modules to analyze the output checkpoints
# from a trained model and make plots for the paper
import torch


class AggregateCheckpoints:
    """A class to load and manage model checkpoints in PyTorch.

    This class provides functionality to load specific model checkpoints,
    as well as to extract additional information from these checkpoints.

    Methods
    -------
    load_checkpoint(model_name, noise, epoch, device, path="models/",
                    BETA=0.5, nmodel=None, COEFF=0.5, loss="SDER",
                    load_rs_chk=False, rs=42, load_nh_chk=False, nh=64)
        Loads a PyTorch model checkpoint from a .pt file based on the
        given parameters.

    ep_al_checkpoint_DE(checkpoint)
        Extracts additional information (mean and variance of validation) from
        a DE model checkpoint.

    ep_al_checkpoint_DER(checkpoint)
        Extracts additional information (mean and std of validation) from
        a DER model checkpoint.
    """

    def load_checkpoint(
        self,
        model_name,
        noise,
        epoch,
        device,
        path="models/",
        BETA=0.5,
        nmodel=None,
        COEFF=0.5,
        loss="SDER",
        load_rs_chk=False,
        rs=42,
        load_nh_chk=False,
        nh=64,
    ):
        """
        Load a PyTorch model checkpoint from a .pt file.

        Parameters
        ----------
        model_name : str
            The name of the model to load.
        noise : float
            The noise level of the model.
        epoch : int
            The epoch number of the checkpoint to load.
        device : str
            The device to load the model onto ('cuda' or 'cpu').
        path : str, optional
            The path where the model checkpoint is stored
            (default is "models/").
        BETA : float, optional
            The beta parameter for the DE model (default is 0.5).
        nmodel : int, optional
            The model number for the DE model (default is None).
        COEFF : float, optional
            The coefficient for the DER model (default is 0.5).
        loss : str, optional
            The loss type for the DER model (default is "SDER").
        load_rs_chk : bool, optional
            Whether to load a checkpoint with a specific random seed
            (default is False).
        rs : int, optional
            The random seed to use if load_rs_chk is True (default is 42).
        load_nh_chk : bool, optional
            Whether to load a checkpoint with a specific number of
            hidden units (default is False).
        nh : int, optional
            The number of hidden units to use if load_nh_chk is True
            (default is 64).

        Returns
        -------
        dict
            The loaded checkpoint as a dictionary.
        """
        if model_name[0:3] == "DER":
            file_name = (
                str(path)
                + f"{model_name}_noise_{noise}_loss_{loss}"
                + f"_COEFF_{COEFF}_epoch_{epoch}"
            )
            if load_rs_chk:
                file_name += f"_rs_{rs}"
            if load_nh_chk:
                file_name += f"_n_hidden_{nh}"
            file_name += ".pt"
        elif model_name[0:2] == "DE":
            file_name = (
                str(path) + f"{model_name}_noise_{noise}_beta_{BETA}_"
                f"nmodel_{nmodel}_epoch_{epoch}.pt"
            )
        checkpoint = torch.load(file_name, map_location=device)
        return checkpoint

    def ep_al_checkpoint_DE(self, checkpoint):
        """
        Extract additional information from a DE model checkpoint.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary from which to extract information.

        Returns
        -------
        tuple
            A tuple containing:
            - mean_validation (numpy.ndarray):
              The predicted mean values for the validation set.
              Again, this is the mu value for all of the data points
              in the entire validation set.
            - var_validation (numpy.ndarray):
              The predicted variance values for the validation set.
              This is the sigma^2 value for all of the data points
              in the entire validation set.
        """
        mean_validation = checkpoint.get("valid_mean", None).detach().numpy()
        var_validation = checkpoint.get("valid_var", None).detach().numpy()
        return mean_validation, var_validation

    def ep_al_checkpoint_DER(self, checkpoint):
        """
        Extract additional information from a DER model checkpoint.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary from which to extract information.

        Returns
        -------
        tuple
            A tuple containing:
            - mean_u_ep (Any): The mean epistemic uncertainty for the epoch
              for the validation set.
            - mean_u_al (Any): The mean aleatoric uncertainty for the epoch
              for the validation set.
            - std_u_ep (Any): The standard deviation of the epistemic
              uncertainty for the epoch for the validation set.
            - std_u_al (Any): The standard deviation of the aleatoric
              uncertainty for the epoch for the validation set.
        """
        mean_u_ep = checkpoint.get("mean_u_ep_validation", None)
        mean_u_al = checkpoint.get("mean_u_al_validation", None)
        std_u_ep = checkpoint.get("std_u_ep_validation", None)
        std_u_al = checkpoint.get("std_u_al_validation", None)

        return mean_u_ep, mean_u_al, std_u_ep, std_u_al
