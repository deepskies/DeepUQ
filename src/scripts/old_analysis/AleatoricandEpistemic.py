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
        help="loss_type for DER",
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
    # make an empty nested dictionary with keys for
    # model names followed by noise levels
    ep_dict = {
        model_name: {noise: [] for noise in noise_list}
        for model_name in model_name_list
    }
    al_dict = {
        model_name: {noise: [] for noise in noise_list}
        for model_name in model_name_list
    }

    ep_std_dict = {
        model_name: {noise: [] for noise in noise_list}
        for model_name in model_name_list
    }
    al_std_dict = {
        model_name: {noise: [] for noise in noise_list}
        for model_name in model_name_list
    }
    n_epochs = config.get_item("model", "n_epochs", "Analysis")
    for model in model_name_list:
        for noise in noise_list:
            # append a noise key
            # now run the analysis on the resulting checkpoints
            if model[0:3] == "DER":
                for epoch in range(n_epochs):
                    chk = chk_module.load_checkpoint(
                        model,
                        noise,
                        epoch,
                        DEVICE,
                        path=path_to_chk,
                        COEFF=COEFF,
                        loss=loss_type,
                    )
                    # path=path_to_chk)
                    # things to grab: 'valid_mse' and 'valid_bnll'
                    (
                        epistemic_m,
                        aleatoric_m,
                        e_std,
                        a_std,
                    ) = chk_module.ep_al_checkpoint_DER(chk)
                    ep_dict[model][noise].append(epistemic_m)
                    al_dict[model][noise].append(aleatoric_m)
                    ep_std_dict[model][noise].append(e_std)
                    al_std_dict[model][noise].append(a_std)

            elif model[0:2] == "DE":
                n_models = config.get_item("model", "n_models", "DE")
                for epoch in range(n_epochs):
                    list_mus = []
                    list_vars = []
                    for nmodels in range(n_models):
                        chk = chk_module.load_checkpoint(
                            model,
                            noise,
                            epoch,
                            DEVICE,
                            path=path_to_chk,
                            BETA=BETA,
                            nmodel=nmodels,
                        )
                        mu_vals, var_vals = chk_module.ep_al_checkpoint_DE(chk)
                        list_mus.append(mu_vals)
                        list_vars.append(var_vals)
                    ep_dict[model][noise].append(
                        np.mean(np.std(list_mus, axis=0))
                    )
                    ep_std_dict[model][noise].append(
                        np.std(np.std(list_mus, axis=0))
                    )
                    al_dict[model][noise].append(
                        np.mean(np.mean(list_vars, axis=0))
                    )
                    al_std_dict[model][noise].append(
                        np.std(np.mean(list_vars, axis=0))
                    )
    # make a two-paneled plot for the different noise levels
    # make one panel per model
    # for the noise levels:
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    # try this instead with a fill_between method
    for i, model in enumerate(model_name_list):
        ax = fig.add_subplot(1, len(model_name_list), i + 1)
        # Your plotting code for each model here
        ax.set_title(model)  # Set title for each subplot
        for i, noise in enumerate(noise_list):
            if model[0:3] == "DER":
                al = np.array(al_dict[model][noise])
                al_std = np.array(al_std_dict[model][noise])
            elif model[0:2] == "DE":
                al = np.array(np.sqrt(al_dict[model][noise]))
                al_std = np.array(np.sqrt(al_std_dict[model][noise]))
            ep = np.array(ep_dict[model][noise])
            ep_std = np.array(ep_std_dict[model][noise])
            total = np.sqrt(al**2 + ep**2)
            total_std = np.sqrt(al_std**2 + ep_std**2)
            ax.fill_between(
                range(n_epochs),
                total - total_std,
                total + total_std,
                color=color_list[i],
                alpha=0.25,
                edgecolor=None,
            )
            ax.plot(
                range(n_epochs),
                total,
                color=color_list[i],
                label=r"$\sigma = $" + str(sigma_list[i]),
            )
            ax.axhline(
                y=sigma_list[i],
                color=color_list[i],
                ls="--",
            )
        ax.set_ylabel("Total Uncertainty")
        ax.set_xlabel("Epoch")
        if model[0:3] == "DER":
            ax.set_title("Deep Evidential Regression")
        elif model[0:2] == "DE":
            ax.set_title("Deep Ensemble (100 models)")
        ax.set_ylim([0, 15])
    plt.legend()
    if config.get_item("analysis", "savefig", "Analysis"):
        plt.savefig(
            str(path_to_out)
            + "aleatoric_and_epistemic_uncertainty_n_epochs_"
            + str(n_epochs)
            + "_n_models_DE_"
            + str(n_models)
            + ".png"
        )
    if config.get_item("analysis", "plot", "Analysis"):
        plt.show()

    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    # try this instead with a fill_between method
    for i, model in enumerate(model_name_list):
        ax = fig.add_subplot(1, len(model_name_list), i + 1)
        # Your plotting code for each model here
        ax.set_title(model)  # Set title for each subplot
        for i, noise in enumerate(noise_list):
            if model[0:3] == "DER":
                al = np.array(al_dict[model][noise])
                al_std = np.array(al_std_dict[model][noise])
            elif model[0:2] == "DE":
                al = np.array(np.sqrt(al_dict[model][noise]))
                al_std = np.array(np.sqrt(al_std_dict[model][noise]))
            ep = np.array(ep_dict[model][noise])
            ep_std = np.array(ep_std_dict[model][noise])
            total = np.sqrt(al**2 + ep**2)
            total_std = np.sqrt(al_std**2 + ep_std**2)
            ax.fill_between(
                range(n_epochs),
                total - total_std,
                total + total_std,
                color=color_list[i],
                alpha=0.25,
                edgecolor=None,
            )
            ax.plot(
                range(n_epochs),
                al,
                color=color_list[i],
                ls="dashed",
                label=r"Aleatoric, $\sigma = $" + str(sigma_list[i]),
            )
            ax.plot(
                range(n_epochs),
                ep,
                color=color_list[i],
                ls="dotted",
                label=r"Epistemic, $\sigma = $" + str(sigma_list[i]),
            )
            ax.plot(
                range(n_epochs),
                total,
                color=color_list[i],
                label=r"Total, $\sigma = $" + str(sigma_list[i]),
            )
            ax.axhline(
                y=sigma_list[i],
                color=color_list[i],
                ls="--",
            )
        ax.set_ylabel("Total Uncertainty")
        ax.set_xlabel("Epoch")
        if model[0:3] == "DER":
            ax.set_title("Deep Evidential Regression")
        elif model[0:2] == "DE":
            ax.set_title("Deep Ensemble (100 models)")
        ax.set_ylim([0, 15])
    plt.legend()
    if config.get_item("analysis", "savefig", "Analysis"):
        plt.savefig(
            str(path_to_out)
            + "aleatoric_and_epistemic_and_total_uncertainty_n_epochs_"
            + str(n_epochs)
            + "_n_models_DE_"
            + str(n_models)
            + ".png"
        )
    if config.get_item("analysis", "plot", "Analysis"):
        plt.show()
