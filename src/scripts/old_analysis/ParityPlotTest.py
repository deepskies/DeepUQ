import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.config import Config
from utils.defaults import DefaultsAnalysis
from data.data import DataPreparation
from analyze.analyze import AggregateCheckpoints
from torch.utils.data import TensorDataset
from models import models

# from plots import Plots


def parse_args():
    parser = argparse.ArgumentParser(description="data handling module")
    # there are three options with the parser:
    # 1) Read from a yaml
    # 2) Reads from the command line and default file
    # and dumps to yaml

    # option to pass name of config
    parser.add_argument("--config", "-c", default=None)

    # model
    # we need some info about the model to run this analysis
    # path to save the model results
    parser.add_argument("--dir", default=DefaultsAnalysis["common"]["dir"])
    # now args for model
    parser.add_argument(
        "--n_models",
        type=int,
        default=DefaultsAnalysis["model"]["n_models"],
        help="Number of MVEs in the ensemble",
    )
    parser.add_argument(
        "--BETA",
        type=beta_type,
        required=False,
        default=DefaultsAnalysis["model"]["BETA"],
        help="If loss_type is bnn_loss, specify a beta as a float or \
            there are string options: linear_decrease, \
            step_decrease_to_0.5, and step_decrease_to_1.0",
    )
    parser.add_argument(
        "--COEFF",
        type=float,
        required=False,
        default=DefaultsAnalysis["model"]["COEFF"],
        help="COEFF for DER",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        required=False,
        default=DefaultsAnalysis["model"]["loss_type"],
        help="loss_type for DER, either SDER or DER",
    )
    parser.add_argument(
        "--noise_level_list",
        type=list,
        required=False,
        default=DefaultsAnalysis["analysis"]["noise_level_list"],
        help="Noise levels to compare",
    )
    parser.add_argument(
        "--model_names_list",
        type=list,
        required=False,
        default=DefaultsAnalysis["analysis"]["model_names_list"],
        help="Beginning of name for saved checkpoints and figures",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=DefaultsAnalysis["model"]["n_epochs"],
        help="number of epochs",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=DefaultsAnalysis["analysis"]["plot"],
        help="option to plot in notebook",
    )
    parser.add_argument(
        "--color_list",
        type=list,
        default=DefaultsAnalysis["plots"]["color_list"],
        help="list of named or hexcode colors to use for the noise levels",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        default=DefaultsAnalysis["analysis"]["savefig"],
        help="option to save a figure of the true and predicted values",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DefaultsAnalysis["analysis"]["verbose"],
        help="verbose option for train",
    )
    args = parser.parse_args()
    args = parser.parse_args()
    if args.config is not None:
        print("Reading settings from config file", args.config)
        config = Config(args.config)

    else:
        temp_config = DefaultsAnalysis["common"]["temp_config"]
        print(
            "Reading settings from cli and default, \
              dumping to temp config: ",
            temp_config,
        )
        os.makedirs(os.path.dirname(temp_config), exist_ok=True)

        # check if args were specified in cli
        input_yaml = {
            "common": {"dir": args.dir},
            "model": {
                "n_models": args.n_models,
                "n_epochs": args.n_epochs,
                "BETA": args.BETA,
                "COEFF": args.COEFF,
                "loss_type": args.loss_type,
            },
            "analysis": {
                "noise_level_list": args.noise_level_list,
                "model_names_list": args.model_names_list,
                "plot": args.plot,
                "savefig": args.savefig,
                "verbose": args.verbose,
            },
            "plots": {"color_list": args.color_list},
            # "metrics": {key: {} for key in args.metrics},
        }

        yaml.dump(input_yaml, open(temp_config, "w"))
        config = Config(temp_config)

    return config
    # return parser.parse_args()


def beta_type(value):
    if isinstance(value, float):
        return value
    elif value.lower() == "linear_decrease":
        return value
    elif value.lower() == "step_decrease_to_0.5":
        return value
    elif value.lower() == "step_decrease_to_1.0":
        return value
    else:
        raise argparse.ArgumentTypeError(
            "BETA must be a float or one of 'linear_decrease', \
            'step_decrease_to_0.5', 'step_decrease_to_1.0'"
        )


if __name__ == "__main__":
    config = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_list = config.get_item("analysis", "noise_level_list", "Analysis")
    color_list = config.get_item("plots", "color_list", "Analysis")
    BETA = config.get_item("model", "BETA", "Analysis")
    COEFF = config.get_item("model", "COEFF", "Analysis")
    loss_type = config.get_item("model", "loss_type", "Analysis")
    sigma_list = []
    for noise in noise_list:
        sigma_list.append(DataPreparation.get_sigma(noise))
    root_dir = config.get_item("common", "dir", "Analysis")
    path_to_chk = root_dir + "checkpoints/"
    path_to_out = root_dir + "analysis/"
    # check that this exists and if not make it
    if not os.path.isdir(path_to_out):
        print("does not exist, making dir", path_to_out)
        os.mkdir(path_to_out)
    else:
        print("already exists", path_to_out)
    model_name_list = config.get_item(
        "analysis", "model_names_list", "Analysis"
    )
    print("model list", model_name_list)
    print("noise list", noise_list)
    chk_module = AggregateCheckpoints()

    # now create a test set
    data = DataPreparation()
    data.sample_params_from_prior(1000)
    data.simulate_data(data.params, 1, "linear_homogeneous", seed=41)
    df_array = data.get_dict()
    # Convert non-tensor entries to tensors
    df = {}
    for key, value in df_array.items():

        if isinstance(value, TensorDataset):
            # Keep tensors as they are
            df[key] = value
        else:
            # Convert lists to tensors
            df[key] = torch.tensor(value)
    len_df = len(df["params"][:, 0].numpy())
    len_x = len(df["inputs"].numpy())
    ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
    bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
    xs_array = np.tile(df["inputs"].numpy(), len_df)
    ys_array = np.reshape(df["output"].numpy(), (len_df * len_x))

    inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs = DataPreparation.normalize(
        inputs, ys_array, False
    )
    _, x_test, _, y_test = DataPreparation.train_val_split(
        model_inputs,
        model_outputs,
        val_proportion=0.1,
        random_state=41,
    )

    # load up the checkpoints for DER
    # and run it on the test data, make a parity plot
    DERmodel, lossFn = models.model_setup_DER("DER", DEVICE, 64)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    axr = fig.add_subplot(212)

    for i, noise in enumerate(noise_list):
        chk = chk_module.load_checkpoint(
            "DER",
            noise,
            99,
            DEVICE,
            path=path_to_chk,
            COEFF=COEFF,
            loss=loss_type,
        )
        # first, define the model at this epoch
        DERmodel.load_state_dict(chk.get("model_state_dict"))
        # checkpoint['model_state_dict'])
        DERmodel.eval()
        # now run on the x_test
        y_pred = DERmodel(torch.Tensor(x_test)).detach().numpy()

        ax.scatter(
            y_test,
            y_pred[:, 0],
            color=color_list[i],
            label=r"$\sigma = $" + str(sigma_list[i]),
            s=3,
        )
        # ax.set_xlabel('True y')
        ax.set_ylabel(r"Predicted $y$")
        ax.plot(
            range(-100, 1100),
            range(-100, 1100),
            ls="--",
            color="grey",
        )

        axr.scatter(
            y_test,
            y_pred[:, 0] - y_test,
            color=color_list[i],
            label=r"$\sigma = $" + str(sigma_list[i]),
            s=3,
            zorder=-100,
        )

        axr.set_xlabel(r"True $y^*$")
        axr.set_ylabel("Residual (predicted - true)")
        axr.axhline(y=0, ls="--", color="grey")
    plt.legend()
    plt.show()
