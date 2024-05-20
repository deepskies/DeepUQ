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
        load_rs_chk=False,
        rs=42,
        load_nh_chk=False,
        nh=64,
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
                    + f"_COEFF_{COEFF}_epoch_{epoch}"
                    )
            if load_rs_chk:
                file_name += (f"_rs_{rs}")
            if load_nh_chk:
                file_name += (f"_n_hidden_{nh}")
            file_name += ".pt"
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
        # valid_sigma is technically the variance
        var_validation = checkpoint.get("valid_sigma", None).detach().numpy()
        return mean_validation, var_validation

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
