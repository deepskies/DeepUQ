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
        "--data_prescription",
        type=str,
        default=DefaultsAnalysis["model"]["data_prescription"],
        help="Current only case is linear homoskedastic",
    )
    parser.add_argument(
        "--data_dimension",
        type=str,
        default=DefaultsAnalysis["model"]["data_dimension"],
        help="0D or 2D",
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
                "data_dimension": args.data_dimension,
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
    inject_type_list = config.get_item("analysis",
                                       "inject_type_list",
                                       "Analysis")
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
        "analysis", "model_names_list", "Analysis")
    print("model list", model_name_list)
    print("noise list", noise_list)
    chk_module = AggregateCheckpoints()
    # make an empty nested dictionary with keys for
    # model names followed by noise levels
    al_dict = {
        typei: {model_name: {noise: [] for noise in noise_list}
                for model_name in model_name_list}
        for typei in inject_type_list
    }
    al_std_dict = {
        typei: {model_name: {noise: [] for noise in noise_list}
                for model_name in model_name_list}
        for typei in inject_type_list
    }
    n_epochs = config.get_item("model", "n_epochs", "Analysis")
    noise_to_sigma = {"low": 1, "medium": 5, "high": 10, "vhigh": 100}
    for inject_type in inject_type_list:
        for model in model_name_list:
            for i, noise in enumerate(noise_list):
                # now create a test set
                size_df = 1000
                data = DataPreparation()
                if dim == "0D":
                    data.sample_params_from_prior(size_df)
                    data.simulate_data(
                        data.params,
                        noise_to_sigma[noise],
                        "linear_homoskedastic",
                        inject_type=inject_type,
                        seed=41,
                    )
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
                    len_x = np.shape(df["output"])[1]
                    ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
                    bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
                    xs_array = np.reshape(df["inputs"].numpy(), (len_df * len_x))
                    ys_array = np.reshape(df["output"].numpy(), (len_df * len_x))

                    inputs = np.array([xs_array, ms_array, bs_array]).T
                elif dim == "2D":
                    data.sample_params_from_prior(
                        size_df,
                        low=[1, 1, -1.5],
                        high=[10, 10, 1.5],
                        n_params=3,
                        seed=41)
                    model_inputs, model_outputs = data.simulate_data_2d(
                        size_df,
                        data.params,
                        image_size=32,
                        inject_type=inject_type)
                model_inputs, model_outputs = DataPreparation.normalize(
                    model_inputs, model_outputs, False
                )
                _, x_test, _, y_test = DataPreparation.train_val_split(
                    model_inputs, model_outputs, val_proportion=0.1,
                    random_state=41
                )
                # append a noise key
                # now run the analysis on the resulting checkpoints
                if model[0:3] == "DER":
                    # first, define the model
                    DERmodel, lossFn = models.model_setup_DER(
                        loss_type, DEVICE, n_hidden=64, data_type=dim
                    )
                    for epoch in range(n_epochs):
                        chk = chk_module.load_checkpoint(
                            model,
                            prescription,
                            inject_type,
                            dim,
                            noise,
                            epoch,
                            DEVICE,
                            path=path_to_chk,
                            COEFF=COEFF,
                            loss=loss_type,
                        )
                        # first, define the model at this epoch
                        DERmodel.load_state_dict(chk.get("model_state_dict"))
                        # checkpoint['model_state_dict'])
                        # print(chk.get("model_state_dict"))
                        # now
                        DERmodel.eval()
                        # now run on the x_test
                        y_pred = DERmodel(torch.Tensor(x_test))

                        loss = lossFn(y_pred, torch.Tensor(y_test), 0.01)
                        mean_u_al_test = np.mean(loss[1])
                        mean_u_ep_test = np.mean(loss[2])
                        std_u_al_test = np.std(loss[1])
                        std_u_ep_test = np.std(loss[2])
                        al_dict[inject_type][model][noise].append(mean_u_al_test)
                        al_std_dict[inject_type][model][noise].append(std_u_al_test)

                elif model[0:2] == "DE":
                    DEmodel, lossFn = models.model_setup_DE(
                        "bnll_loss", DEVICE, n_hidden=64, data_type=dim)
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
                            DEmodel.load_state_dict(chk.get("model_state_dict"))
                            DEmodel.eval()
                            y_pred = DEmodel(torch.Tensor(x_test)).detach().numpy()
                            list_mus.append(y_pred[:, 0].flatten())
                            list_vars.append(y_pred[:, 1].flatten())
                        al_dict[inject_type][model][noise].append(
                            np.mean(np.mean(list_vars, axis=0)))
                        al_std_dict[inject_type][model][noise].append(
                            np.std(np.mean(list_vars, axis=0)))
    # make a two-paneled plot for the different noise levels
    # make one panel per model
    # for the noise levels:
    plt.clf()
    fig = plt.figure(figsize=(10, 4))
    # ax = fig.add_subplot(111)
    # try this instead with a fill_between method
    sym_list = ["^", "*"]

    #for m, model in enumerate(model_name_list):
    for j, inject_type in enumerate(inject_type_list):
        if inject_type == "predictive":
            true_sigma = {"low": 1, "medium": 5, "high": 10}
        elif inject_type == "feature":
            if dim == "0D":
                true_sigma = {
                    "low": 1 * np.mean(x_test[:, 1]),
                    "medium": 5 * np.mean(x_test[:, 1]),
                    "high": 10 * np.mean(x_test[:, 1])}
            elif dim == "2D":
                true_sigma = {
                    "low": 1 * np.mean(x_test[:, 1]),
                    "medium": np.sqrt(5) * 32,
                    "high": np.sqrt(10) * 32}
        
        ax = fig.add_subplot(1, len(inject_type_list), j + 1)
        # Your plotting code for each model here
        for i, noise in enumerate(noise_list):
            if model[0:3] == "DER":
                al = np.array(al_dict[inject_type][model][noise])
                al_std = np.array(al_std_dict[inject_type][model][noise])
            elif model[0:2] == "DE":
                # only take the sqrt for the case of DE,
                # which is the variance
                al = np.array(np.sqrt(al_dict[inject_type][model][noise]))
                al_std = np.array(np.sqrt(al_std_dict[inject_type][model][noise]))
            # summarize the aleatoric
            ax.errorbar(
                #sigma_list[i],
                true_sigma[noise],
                al[-1],
                yerr=al_std[-1],
                color=color_list[i],
                capsize=5
            )
            ax.scatter(
                true_sigma[noise],
                al[-1],
                color=color_list[i],
                label=r"$\sigma = $" + str(sigma_list[i]),
            )
        ax.set_ylabel("Aleatoric Uncertainty")
        ax.set_xlabel("True (Injected) Uncertainty")
        ax.plot(range(0, 15), range(0, 15), ls="--", color="black")
        #ax.set_ylim([0, 14])
        #ax.set_xlim([0, 14])
        if model[0:3] == "DER":
            title = "Deep Evidential Regression"
        elif model[0:2] == "DE":
            title = "Deep Ensemble (100 models)"
        title += inject_type
        ax.set_title(title)
    plt.legend()
    if config.get_item("analysis", "savefig", "Analysis"):
        plt.savefig(
            str(path_to_out)
            + "parity_plot_uncertainty_test_"
            + str(n_epochs)
            + "_n_models_DE_"
            + str(n_models)
            + ".png"
        )
    if config.get_item("analysis", "plot", "Analysis"):
        plt.show()
