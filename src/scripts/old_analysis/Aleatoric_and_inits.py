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
        "--data_prescription",
        type=str,
        default=DefaultsAnalysis["model"]["data_prescription"],
        help="Current only case is linear homoskedastic",
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
        "--inject_type_list",
        type=list,
        required=False,
        default=DefaultsAnalysis["analysis"]["inject_type_list"],
        help="Options are predictive and feature",
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
                "data_prescription": args.data_prescription,
                "BETA": args.BETA,
                "COEFF": args.COEFF,
                "loss_type": args.loss_type,
            },
            "analysis": {
                "noise_level_list": args.noise_level_list,
                "model_names_list": args.model_names_list,
                "inject_type_list": args.inject_type_list,
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
    prescription = config.get_item("model", "data_prescription", "Analysis")
    inject_type_list = config.get_item(
        "analysis", "inject_type_list", "Analysis"
    )
    dim = config.get_item("model", "data_dimension", "Analysis")
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
    model = model_name_list[0]
    print("one model at a time", model)
    print("noise list", noise_list)
    chk_module = AggregateCheckpoints()
    # make an empty nested dictionary with keys for
    # model names followed by noise levels
    al_dict = {
        typei: {
            model_name: {noise: [] for noise in noise_list}
            for model_name in model_name_list
        }
        for typei in inject_type_list
    }
    al_std_dict = {
        typei: {
            model_name: {noise: [] for noise in noise_list}
            for model_name in model_name_list
        }
        for typei in inject_type_list
    }
    n_epochs = config.get_item("model", "n_epochs", "Analysis")
    # switching from two panels for different models to
    # two panels for different injection types
    # could eventually make this into a four panel plot
    # for model in model_name_list:
    for inject_type in inject_type_list:
        for model_name in model_name_list:
            for noise in noise_list:
                # append a noise key
                # now run the analysis on the resulting checkpoints
                if model[0:3] == "DER":
                    for epoch in range(n_epochs):
                        chk = chk_module.load_checkpoint(
                            model_name,
                            prescription,
                            inject_type,
                            dim,
                            noise,
                            epoch,
                            DEVICE,
                            path=path_to_chk,
                            COEFF=COEFF,
                            loss=loss_type,
                            load_nh_chk=False,
                        )
                        # path=path_to_chk)
                        # things to grab: 'valid_mse' and 'valid_bnll'
                        (
                            epistemic_m,
                            aleatoric_m,
                            e_std,
                            a_std,
                        ) = chk_module.ep_al_checkpoint_DER(chk)
                        al_dict[inject_type][model][noise].append(aleatoric_m)
                        al_std_dict[inject_type][model][noise].append(a_std)

                elif model[0:2] == "DE":
                    n_models = config.get_item("model", "n_models", "DE")
                    for epoch in range(n_epochs):
                        list_mus = []
                        list_vars = []
                        for nmodels in range(n_models):
                            chk = chk_module.load_checkpoint(
                                model,
                                prescription,
                                inject_type,
                                dim,
                                noise,
                                epoch,
                                DEVICE,
                                path=path_to_chk,
                                BETA=BETA,
                                nmodel=nmodels,
                            )
                            mu_vals, var_vals = chk_module.ep_al_checkpoint_DE(
                                chk
                            )
                            list_mus.append(mu_vals)
                            list_vars.append(var_vals)
                        # first taking the mean across the validation data
                        # then looking at the mean and standard deviation
                        # across all of the nmodels
                        al_dict[inject_type][model][noise].append(
                            np.mean(np.mean(list_vars, axis=0))
                        )
                        al_std_dict[inject_type][model][noise].append(
                            np.std(np.mean(list_vars, axis=0))
                        )
    # make a two-paneled plot for the different noise levels
    # make one panel per model
    # for the noise levels:
    # this needs to be redone
    rs_list = [10, 11, 12, 13, 14]
    # make an empty nested dictionary with keys for
    # model names followed by noise levels
    al_rs_dict = {
        model_name: {noise: {rs: [] for rs in rs_list} for noise in noise_list}
        for model_name in model_name_list
    }
    """
    al_rs_std_dict = {
        model_name: {noise: {rs: [] for rs in rs_list} for noise in noise_list}
        for model_name in model_name_list
    }
    """
    n_epochs = config.get_item("model", "n_epochs", "Analysis")
    # for model in model_name_list:
    for inject_type in inject_type_list:
        for noise in noise_list:
            for rs in rs_list:

                # append a noise key
                # now run the analysis on the resulting checkpoints
                if model[0:3] == "DER":
                    for epoch in range(n_epochs):
                        chk = chk_module.load_checkpoint(
                            model,
                            prescription,
                            inject_type,
                            noise,
                            epoch,
                            DEVICE,
                            path=path_to_chk,
                            COEFF=COEFF,
                            loss=loss_type,
                            load_rs_chk=True,
                            rs=rs,
                            load_nh_chk=False,
                        )
                        # path=path_to_chk)
                        # things to grab: 'valid_mse' and 'valid_bnll'
                        _, aleatoric_m, _, a_std = (
                            chk_module.ep_al_checkpoint_DER(chk)
                        )
                        al_rs_dict[model][noise][rs].append(aleatoric_m)
                        # al_std_dict[model][noise][rs].append(a_std)
            if model[0:2] == "DE" and model[0:3] != "DER":
                n_models = config.get_item("model", "n_models", "DE")
                for epoch in range(n_epochs):
                    list_mus = []
                    list_vars = []
                    for nmodels in range(n_models):
                        chk = chk_module.load_checkpoint(
                            model,
                            prescription,
                            inject_type,
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
                        try:
                            al_rs_dict[model][noise][nmodels + 1].append(
                                np.mean(list_vars)
                            )
                        except KeyError:
                            continue
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
        for n, noise in enumerate(noise_list):
            if model[0:3] == "DER":
                al = np.array(al_dict[model][noise])
                al_std = np.array(al_std_dict[model][noise])
            elif model[0:2] == "DE":
                # only take the sqrt for the case of DE,
                # which is the variance
                al = np.array(np.sqrt(al_dict[model][noise]))
                al_std = np.array(np.sqrt(al_std_dict[model][noise]))
            ax.fill_between(
                range(n_epochs),
                al - al_std,
                al + al_std,
                color=color_list[n],
                alpha=0.25,
                edgecolor=None,
            )
            ax.plot(
                range(n_epochs),
                al,
                color=color_list[n],
                label=r"$\sigma = $" + str(sigma_list[n]),
                lw=3,
            )
            for r, rs in enumerate(rs_list):
                if model[0:3] == "DER":
                    al = np.array(al_rs_dict[model][noise][rs])
                elif model[0:2] == "DE":
                    al = np.array(np.sqrt(al_rs_dict[model][noise][rs]))
                """
                # it doesn't really make sense to plot the std for the
                # case of the DE because each individual model
                # makes up one in the ensemble
                """
                """
                if model[0:3] == "DER":
                    al_std = np.array(al_std_dict[model][noise][rs])
                    ax.fill_between(
                        range(n_epochs),
                        al - al_std,
                        al + al_std,
                        color=color_list[n],
                        alpha=0.1,
                        edgecolor=None,
                    )
                """
                if r == 0:
                    ax.plot(
                        range(n_epochs),
                        al,
                        color=color_list[n],
                    )
                else:
                    ax.plot(
                        range(n_epochs),
                        al,
                        color=color_list[n],
                    )
            ax.axhline(
                y=sigma_list[n],
                color=color_list[n],
                ls="--",
            )
        ax.set_ylabel("Aleatoric Uncertainty")
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
            + "aleatoric_uncertainty_and_inits_n_epochs_"
            + str(n_epochs)
            + "_n_models_DE_"
            + str(n_models)
            + ".png"
        )
    if config.get_item("analysis", "plot", "Analysis"):
        plt.show()
