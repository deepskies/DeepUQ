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
        "--prescription",
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
        "--inject_type_list",
        type=str,
        default=DefaultsAnalysis["analysis"]["inject_type_list"],
        help="Options are predictive or feature",
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
        "--noise_level",
        type=str,
        required=False,
        default=DefaultsAnalysis["analysis"]["noise_level"],
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
                "prescription": args.prescription,
                "data_dimension": args.data_dimension,
                "BETA": args.BETA,
                "COEFF": args.COEFF,
                "loss_type": args.loss_type,
            },
            "analysis": {
                "noise_level": args.noise_level,
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
    noise = config.get_item("analysis", "noise_level", "Analysis")
    inject_type_list = config.get_item(
        "analysis", "inject_type_list", "Analysis")
    color_list = config.get_item("plots", "color_list", "Analysis")
    BETA = config.get_item("model", "BETA", "Analysis")
    COEFF = config.get_item("model", "COEFF", "Analysis")
    model_type = config.get_item("analysis", "model_names_list", "Analysis")[0]
    loss_type = config.get_item("model", "loss_type", "Analysis")
    prescription = config.get_item("model", "prescription", "Analysis")
    dim = config.get_item("model", "data_dimension", "Analysis")

    print("noise", noise)
    print("inject type list", inject_type_list)

    noise_to_sigma = {"low": 1, "medium": 5, "high": 10, "vhigh": 100}
    sigma_inject = noise_to_sigma[noise]
    root_dir = config.get_item("common", "dir", "Analysis")
    path_to_chk = root_dir + "checkpoints/"
    path_to_out = root_dir + "analysis/"
    # check that this exists and if not make it
    if not os.path.isdir(path_to_out):
        print("does not exist, making dir", path_to_out)
        os.mkdir(path_to_out)
    else:
        print("already exists", path_to_out)
    chk_module = AggregateCheckpoints()

    
    print("model_type", model_type)
    if model_type == "DER":
        # load up the checkpoints for DER
        # and run it on the test data, make a parity plot
        model, lossFn = models.model_setup_DER(
            "DER", DEVICE, n_hidden=64, data_type=dim)
    elif model_type == "DE":
        print('loss_type', loss_type)
        model, lossFn = models.model_setup_DE(
            loss_type, DEVICE, n_hidden=64, data_type=dim)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    axr = fig.add_subplot(212)

    #for i, noise in enumerate(noise_list):
    for typei in inject_type_list:
        # now create a test set
        size_df = 1000
        data = DataPreparation()
        if dim == "0D":
            data.sample_params_from_prior(size_df)
            data.simulate_data(
                data.params,
                sigma_inject,
                "linear_homoskedastic",
                inject_type=typei,
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
                inject_type=typei)
        model_inputs, model_outputs = DataPreparation.normalize(
            model_inputs, model_outputs, False
        )
        _, x_test, _, y_test = DataPreparation.train_val_split(
            model_inputs, model_outputs, val_proportion=0.1,
            random_state=41
        )
        if model_type == "DER":
            chk = chk_module.load_checkpoint(
                model_type,
                prescription,
                typei,
                dim,
                noise,
                99,
                DEVICE,
                path=path_to_chk,
                COEFF=COEFF,
                loss=loss_type,
            )
        elif model_type == "DE":
            chk = chk_module.load_checkpoint(
                model_type,
                prescription,
                typei,
                dim,
                noise,
                99,
                DEVICE,
                path=path_to_chk,
                nmodel=0,
                COEFF=COEFF,
                loss=loss_type,
            )
        

        # first, define the model at this epoch
        model.load_state_dict(chk.get("model_state_dict"))
        # checkpoint['model_state_dict'])
        model.eval()
        # now run on the x_test
        y_pred = model(torch.Tensor(x_test))
        print(y_pred)
        y_pred = model(torch.Tensor(x_test)).detach().numpy()
        if model_type == "DER":
            assert model_type == "DER", "stop haven't written this yet"
        elif model_type == "DE":
            sigma = np.sqrt(y_pred[:, 1])


        plt.clf()
        plt.scatter(y_test, y_pred[:, 0])
        plt.errorbar(y_test, y_pred[:, 0], yerr=sigma, fmt='o', linestyle='None')
        plt.xlabel('true')
        plt.ylabel('predicted')
        plt.show()

        plt.clf()
        plt.hist(sigma, alpha=0.5)
        plt.axvline(x=np.mean(sigma))
        plt.title(str(round(np.mean(sigma), 2)))
        plt.xlabel('output sigma')
        plt.show()

        STOP

        print(np.shape(x_test))
        for i in range(10):
            plt.clf()
            plt.imshow(x_test[i,:,:])
            plt.title(f'y_true = {y_test[i]}, y_pred = {y_pred[i, 0]} +/- {sigma[i]}')
            plt.show()
        
        

        print(x_test[:, 1])
        print('mean of predicted sigmas', np.mean(sigma))
        if dim == "0D":
            if typei == "predictive":
                y_noisy = y_test
                y_noiseless = x_test[:, 1] * x_test[:, 0] + x_test[:, 2]
                sub = y_noisy - y_noiseless
                label = r"$y_{noisy} - y_{noiseless}$"
            elif typei == "feature":
                y_noisy = x_test[:, 1] * x_test[:, 0] + x_test[:, 2]
                y_noiseless = y_test
                sub = y_noisy - y_noiseless  # / x_test[:, 1]
                label = r"$(y_{noisy} - y_{noiseless})$"  # / m$'
                # finally, analytically propagate
                if dim == "0D":
                    print('mean of ms', np.mean(x_test[:, 1]))
                    true_analytic = sigma_inject * np.mean(x_test[:, 1])
        elif dim == "2D":
            if typei == "predictive":
                assert "haven't done this yet"
            elif typei == "feature":
                assert "haven't done this yet"

        plt.clf()
        if noise == "high":
            heights1, bins = np.histogram(sub, bins=50, range=[-200, 200])
        elif noise == "medium":
            heights1, bins = np.histogram(sub, bins=50, range=[-100, 100])
        elif noise == "low":
            heights1, bins = np.histogram(sub, bins=50, range=[-20, 20])
        heights2, _ = np.histogram(sigma, bins=bins)
        plt.hist(
            sub,
            bins=bins,
            alpha=0.5,
            label=label,
            color="#610345",
            )
        plt.hist(
            sigma,
            bins=bins,
            alpha=0.5,
            label="predicted sigma",
            color="#9EB25D",
            )
        plt.annotate(str(round(true_analytic, 2)),
                     xy=(np.mean(true_analytic),
                         np.max([np.max(heights1), np.max(heights2)])/2 + 200),
                     xycoords='data',
                     color="black",
                     )
        plt.axvline(x=np.mean(sigma), color="#9EB25D", ls='-')
        plt.annotate(str(round(np.mean(sigma), 2)),
                     xy=(np.mean(sigma),
                         np.max([np.max(heights1), np.max(heights2)])/2 + 100),
                     xycoords='data',
                     color="#9EB25D",
                     )
        # plt.axvline(x=np.mean(sub), color="black", ls="--")
        plt.axvline(x=np.std(sub), color="black", ls="--")
        plt.annotate(str(round(np.std(sub), 2)),
                     xy=(np.std(sub),
                         np.max([np.max(heights1), np.max(heights2)])/2),
                     xycoords='data',
                     color="grey",
                     )
        plt.axvline(
            x=np.percentile(sub, 50) - np.percentile(sub, 16),
            color="red", ls="--"
        )
        
        plt.axvline(
            x=np.percentile(sub, 84) - np.percentile(sub, 50),
            color="red", ls="--"
        )
        # plt.axvline(x=noise_to_sigma[noise], color="black")
        plt.legend()
        plt.title(str(noise) + " noise " + str(model_type))
        plt.show()