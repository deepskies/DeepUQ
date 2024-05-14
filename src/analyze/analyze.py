# Contains modules to analyze the output checkpoints
# from a trained model and make plots for the paper
import numpy as np
import torch


class AggregateCheckpoints:
    # def load_final_checkpoints():
    # def load_all_checkpoints():
    # functions for loading model checkpoints
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
    ):
        """
        Load PyTorch model checkpoint from a .pt file.

        :param path: Location to load the model from
        :param DER_type: Type of your model
        :param epoch: Epoch to load
        :param device: Device to load the model onto ('cuda' or 'cpu')
        :param model: PyTorch model to load the checkpoint into
        :return: Loaded model
        """
        if model_name[0:3] == "DER":
            file_name = (
                str(path)
                + f"{model_name}_noise_{noise}_loss_{loss}"
                + f"_COEFF_{COEFF}_epoch_{epoch}.pt"
            )
            checkpoint = torch.load(file_name, map_location=device)
        elif model_name[0:2] == "DE":
            file_name = (
                str(path)
                + f"{model_name}_noise_{noise}_beta_{BETA}_"
                  f"nmodel_{nmodel}_epoch_{epoch}.pt"
            )
            checkpoint = torch.load(file_name, map_location=device)
        return checkpoint

    def ep_al_checkpoint_DE(self, checkpoint):
        # Extract additional information
        # loaded_epoch = checkpoint.get("epoch", None)
        mean_validation = checkpoint.get("valid_mean", None).detach().numpy()
        # this valid_sigma is actually the variance so you'll need to take
        # the sqrt of this
        sigma_validation = np.sqrt(
            checkpoint.get("valid_sigma", None).detach().numpy())
        return mean_validation, sigma_validation

    def ep_al_checkpoint_DER(self, checkpoint):
        """
        # Handle the case where extra information is present in the state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        """

        # Extract additional information
        med_u_ep = checkpoint.get("med_u_ep_validation", None)
        med_u_al = checkpoint.get("med_u_al_validation", None)
        std_u_ep = checkpoint.get("std_u_ep_validation", None)
        std_u_al = checkpoint.get("std_u_al_validation", None)

        return med_u_ep, med_u_al, std_u_ep, std_u_al
