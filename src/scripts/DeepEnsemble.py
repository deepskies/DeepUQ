import time
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

# from scripts import train, models, io
from train import train
from models import models
from data import DataModules
from models import ModelModules
from utils.config import Config
from utils.defaults import DefaultsDE
from data.data import DataPreparation, MyDataLoader

# from analyze.analyze import AggregateCheckpoints

# from plots import Plots


def parse_args():
    """Parses command-line (or config) arguments for the DeepEnsemble script.

    This function creates an argument parser that supports:
    1) Reading from a YAML configuration file (via --config).
    2) Reading arguments from the command line and using default values 
       if not provided, with the option to dump arguments to a temporary 
       YAML configuration file.
    3) Specifying data-related parameters such as the data path, dimension, 
       and injection method, as well as model parameters like the number 
       of models in the ensemble, learning rate, and loss type.

    The parser supports the following argument categories:
    - Data-related arguments:
        --data_path, --data_dimension, --data_prescription, --data_injection,
        --data_engine, --size_df, --noise_level, --val_proportion, --randomseed, 
        --generatedata, --batchsize, --normalize, --uniform
    - Model-related arguments:
        --model_engine, --n_models, --init_lr, --loss_type, --BETA, --model_type,
        --n_epochs, --save_all_checkpoints, --save_final_checkpoint,
        --overwrite_final_checkpoint, --plot, --savefig, --save_chk_random_seed_init,
        --rs_list, --n_hidden, --save_n_hidden, --save_data_size, --verbose
    - General arguments:
        --config, --out_dir

    Returns:
        Config: Configuration object that combines parsed arguments, 
        either from the command line or a YAML configuration file.
    """
    parser = argparse.ArgumentParser(description="data handling module")
    # there are three options with the parser:
    # 1) Read from a yaml
    # 2) Reads from the command line and default file
    # and dumps to yaml

    # option to pass name of config
    parser.add_argument("--config", "-c", default=None)

    # data info
    parser.add_argument(
        "--data_path",
        "-d",
        default=DefaultsDE["data"]["data_path"],
    )
    parser.add_argument(
        "--data_dimension", "-dd", default=DefaultsDE["data"]["data_dimension"]
    )
    parser.add_argument(
        "--data_prescription",
        "-dp",
        default=DefaultsDE["data"]["data_prescription"],
    )
    parser.add_argument(
        "--data_injection", "-di", default=DefaultsDE["data"]["data_injection"]
    )
    parser.add_argument(
        "--data_engine",
        "-dl",
        default=DefaultsDE["data"]["data_engine"],
        choices=DataModules.keys(),
    )

    # model
    # path to save the model results
    parser.add_argument("--out_dir", default=DefaultsDE["common"]["out_dir"])
    parser.add_argument(
        "--model_engine",
        "-e",
        default=DefaultsDE["model"]["model_engine"],
        choices=ModelModules.keys(),
    )
    parser.add_argument(
        "--size_df",
        type=float,
        required=False,
        default=DefaultsDE["data"]["size_df"],
        help="Used to load the associated .h5 data file",
    )
    parser.add_argument(
        "--noise_level",
        type=str,
        default=DefaultsDE["data"]["noise_level"],
        choices=["low", "medium", "high", "vhigh"],
        help="low, medium, high or vhigh, \
            used to look up associated sigma value",
    )
    parser.add_argument(
        "--normalize",
        required=False,
        action="store_true",
        default=DefaultsDE["data"]["normalize"],
        help="If true theres an option to normalize the dataset",
    )
    parser.add_argument(
        "--uniform",
        required=False,
        action="store_true",
        default=DefaultsDE["data"]["uniform"],
        help="If true theres an option to uniformize the dataset",
    )
    parser.add_argument(
        "--val_proportion",
        type=float,
        required=False,
        default=DefaultsDE["data"]["val_proportion"],
        help="Proportion of the dataset to use as validation",
    )
    parser.add_argument(
        "--randomseed",
        type=int,
        required=False,
        default=DefaultsDE["data"]["randomseed"],
        help="Random seed used for shuffling the training and validation set",
    )
    parser.add_argument(
        "--generatedata",
        action="store_true",
        default=DefaultsDE["data"]["generatedata"],
        help="option to generate df, if not specified \
            default behavior is to load from file",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        required=False,
        default=DefaultsDE["data"]["batchsize"],
        help="Size of batched used in the traindataloader",
    )
    # now args for model
    parser.add_argument(
        "--n_models",
        type=int,
        default=DefaultsDE["model"]["n_models"],
        help="Number of MVEs in the ensemble",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        required=False,
        default=DefaultsDE["model"]["init_lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        required=False,
        default=DefaultsDE["model"]["loss_type"],
        help="Loss types for MVE, options are no_var_loss, var_loss, \
            and bnn_loss",
    )
    parser.add_argument(
        "--BETA",
        type=beta_type,
        required=False,
        default=DefaultsDE["model"]["BETA"],
        help="If loss_type is bnn_loss, specify a beta as a float or \
            there are string options: linear_decrease, \
            step_decrease_to_0.5, and step_decrease_to_1.0",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default=DefaultsDE["model"]["model_type"],
        help="Beginning of name for saved checkpoints and figures",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=DefaultsDE["model"]["n_epochs"],
        help="number of epochs for each MVE",
    )
    parser.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        default=DefaultsDE["model"]["save_all_checkpoints"],
        help="option to save all checkpoints",
    )
    parser.add_argument(
        "--save_final_checkpoint",
        action="store_true",  # Set to True if argument is present
        default=DefaultsDE["model"]["save_final_checkpoint"],
        help="option to save the final epoch checkpoint for each ensemble",
    )
    parser.add_argument(
        "--overwrite_final_checkpoint",
        action="store_true",
        default=DefaultsDE["model"]["overwrite_final_checkpoint"],
        help="option to overwite already saved checkpoints",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=DefaultsDE["model"]["plot"],
        help="option to plot in notebook",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        default=DefaultsDE["model"]["savefig"],
        help="option to save a figure of the true and predicted values",
    )
    parser.add_argument(
        "--save_chk_random_seed_init",
        action="store_true",
        default=DefaultsDE["model"]["save_chk_random_seed_init"],
        help="option to save the chk with a random seed",
    )
    parser.add_argument(
        "--rs_list",
        type=list[int],
        default=DefaultsDE["model"]["rs_list"],
        help="random seed list for the pytorch model initialization, \
            should be the same length as n_models",
    )
    parser.add_argument(
        "--save_n_hidden",
        action="store_true",
        default=DefaultsDE["model"]["save_n_hidden"],
        help="save chk with the number of neurons in the hidden layer",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        required=False,
        default=DefaultsDE["model"]["n_hidden"],
        help="Number of hidden neurons in the hidden layer, default 64",
    )
    parser.add_argument(
        "--save_data_size",
        action="store_true",
        default=DefaultsDE["model"]["save_data_size"],
        help="save chk with the number of examples in the dataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DefaultsDE["model"]["verbose"],
        help="verbose option for train",
    )
    args = parser.parse_args()
    args = parser.parse_args()
    if args.config is not None:
        print("Reading settings from config file", args.config)
        config = Config(args.config)

    else:
        temp_config_prefix = DefaultsDE["common"]["temp_config"]
        # modify this to also have a timestamp
        # Get current timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")

        # Modify name with timestamp
        temp_config = temp_config_prefix.replace(".yml", f"_{timestamp}.yml")

        print(
            "Reading settings from cli and default, \
              dumping to temp config: ",
            temp_config,
        )
        os.makedirs(os.path.dirname(temp_config), exist_ok=True)

        # check if args were specified in cli
        # if not, default is from DefaultsDE dictionary
        input_yaml = {
            "common": {"out_dir": args.out_dir},
            "model": {
                "model_engine": args.model_engine,
                "model_type": args.model_type,
                "loss_type": args.loss_type,
                "n_models": args.n_models,
                "init_lr": args.init_lr,
                "BETA": args.BETA,
                "n_epochs": args.n_epochs,
                "save_all_checkpoints": args.save_all_checkpoints,
                "save_final_checkpoint": args.save_final_checkpoint,
                "overwrite_final_checkpoint": args.overwrite_final_checkpoint,
                "plot": args.plot,
                "savefig": args.savefig,
                "save_chk_random_seed_init": args.save_chk_random_seed_init,
                "rs_list": args.rs_list,
                "save_n_hidden": args.save_n_hidden,
                "n_hidden": args.n_hidden,
                "save_data_size": args.save_data_size,
                "verbose": args.verbose,
            },
            "data": {
                "data_path": args.data_path,
                "data_engine": args.data_engine,
                "data_dimension": args.data_dimension,
                "data_prescription": args.data_prescription,
                "data_injection": args.data_injection,
                "size_df": args.size_df,
                "noise_level": args.noise_level,
                "val_proportion": args.val_proportion,
                "randomseed": args.randomseed,
                "batchsize": args.batchsize,
                "generatedata": args.generatedata,
                "normalize": args.normalize,
                "uniform": args.uniform,
            },
            # "plots": {key: {} for key in args.plots},
            # "metrics": {key: {} for key in args.metrics},
        }

        yaml.dump(input_yaml, open(temp_config, "w"))
        config = Config(temp_config)

    return config
    # return parser.parse_args()


def beta_type(value):
    """Validates and returns the BETA argument for the DeepEnsemble script.

    The BETA argument can either be:
    1) A float value.
    2) One of the following strings: 'linear_decrease', 'step_decrease_to_0.5', 
       or 'step_decrease_to_1.0'.

    Args:
        value (str or float): The BETA value passed as a command-line argument.

    Returns:
        float or str: The valid BETA value (either a float or one of the allowed strings).

    Raises:
        argparse.ArgumentTypeError: If the value is neither a float nor one of the 
        allowed string values.
    """
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
    """Main execution script for the DeepEnsemble pipeline.

    This script performs the following steps:
    1. Parses command-line arguments and configuration settings using
    `parse_args`.
    2. Prepares and/or loads data based on the specified data dimensionality
    (0D or 2D).
    3. Processes and normalizes the data, applying optional noise injection.
    4. Visualizes the data distribution if verbose mode is enabled.
    5. Splits the data into training and validation sets.
    6. Trains an ensemble of models using the DeepEnsemble framework and
    specified configurations.
    7. Optionally saves checkpoints and plots during the training process.

    Execution starts if the script is run as a standalone module.
    """
    config = parse_args()
    verbose = config.get_item("model", "verbose", "DE")
    size_df = int(config.get_item("data", "size_df", "DE"))
    noise = config.get_item("data", "noise_level", "DE")
    norm = config.get_item("data", "normalize", "DE", raise_exception=False)
    uniform = config.get_item("data", "uniform", "DE", raise_exception=False)
    val_prop = config.get_item("data", "val_proportion", "DE")
    # this is the data rs
    rs = config.get_item("data", "randomseed", "DE")
    BATCH_SIZE = config.get_item("data", "batchsize", "DE")
    path_to_data = config.get_item("data", "data_path", "DE")
    prescription = config.get_item("data", "data_prescription", "DE")
    injection = config.get_item("data", "data_injection", "DE")
    dim = config.get_item("data", "data_dimension", "DE")
    if config.get_item("data", "generatedata", "DE", raise_exception=False):
        # generate the df
        print("generating the data")
        data = DataPreparation()
        if dim == "0D":
            data.sample_params_from_prior(size_df)
            print("injecting this noise", noise)
            if injection == "feature":
                vary_sigma = True
                print("are we varying sigma", vary_sigma)
                data.simulate_data(
                    data.params,
                    noise,
                    prescription,
                    x=np.linspace(0, 10, 100),
                    inject_type=injection,
                    vary_sigma=vary_sigma,
                )
            elif injection == "predictive":
                sigma = DataPreparation.get_sigma(
                    noise, inject_type=injection, data_dimension=dim
                )
                print(
                    f"inject type is {injection},"
                    f"dim is {dim}, sigma is {sigma}"
                )
                data.simulate_data(
                    data.params,
                    sigma,
                    prescription,
                    x=np.linspace(0, 10, 100),
                    inject_type=injection,
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
        elif dim == "2D":
            print("2D data")
            sigma = DataPreparation.get_sigma(
                noise, inject_type=injection, data_dimension=dim
            )
            print(
                f"inject type is {injection}, dim is {dim}, sigma is {sigma}"
            )
            data.sample_params_from_prior(
                size_df,
                low=[0, 1, -1.5],
                high=[0.01, 10, 1.5],
                n_params=3,
                seed=42,
            )
            model_inputs, model_outputs = data.simulate_data_2d(
                size_df,
                data.params,
                sigma,
                image_size=32,
                inject_type=injection,
            )
    else:
        loader = MyDataLoader()
        if dim == "0D":
            filename = (
                str(prescription)
                + "_"
                + str(injection)
                + "_sigma_"
                + str(sigma)
                + "_size_"
                + str(size_df)
            )
            df = loader.load_data_h5(filename, path=path_to_data)
            print("loaded this file: ", filename)
    if dim == "0D":
        len_df = len(df["params"][:, 0].numpy())
        len_x = np.shape(df["output"])[1]
        ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
        bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
        xs_array = np.reshape(df["inputs"].numpy(), (len_df * len_x))
        model_outputs = np.reshape(df["output"].numpy(), (len_df * len_x))
        model_inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs, norm_params = DataPreparation.normalize(
        model_inputs, model_outputs, norm
    )
    if uniform:
        model_inputs, model_outputs = DataPreparation.select_uniform(
            model_inputs, model_outputs, dim=dim, verbose=verbose, rs=40
        )
    if verbose:
        plt.clf()
        plt.hist(model_outputs)
        plt.axvline(x=np.mean(model_outputs), color="yellow")
        plt.annotate(
            str(np.mean(model_outputs)),
            xy=(0.02, 0.9),
            xycoords="axes fraction",
        )
        plt.show()
        if dim == "2D":
            print(model_outputs)
            counter = 0
            for p in range(len(model_outputs)):
                if counter > 5:
                    break
                if model_outputs[p] > 0.75 and model_outputs[p] < 1.25:
                    plt.clf()
                    plt.imshow(model_inputs[p])
                    plt.annotate(
                        "Pixel sum = " + str(round(model_outputs[p], 2)),
                        xy=(0.02, 0.9),
                        xycoords="axes fraction",
                        color="white",
                        size=10,
                    )
                    plt.colorbar()
                    plt.show()
                    counter += 1
        elif dim == "0D":
            plt.clf()
            plt.scatter(
                model_inputs[0:1000, 0],
                model_outputs[0:1000],
                c=model_inputs[0:1000, 1],
                cmap="viridis",
            )
            plt.colorbar()
            # plt.plot(model_inputs[0:100, 0], model_outputs[0:100])
            plt.title("x and y, colorbar is m value")
            plt.show()

            # select an m value
            print("m value", model_inputs[0, 1], model_inputs[500, 1])
            # grab everything that corresponds to it
            plt.clf()
            plt.scatter(
                model_inputs[:, 0][model_inputs[:, 1] == model_inputs[0, 1]],
                model_outputs[model_inputs[:, 1] == model_inputs[0, 1]],
                color="yellow",
            )
            plt.scatter(
                model_inputs[:, 0][model_inputs[:, 1] == model_inputs[500, 1]],
                model_outputs[model_inputs[:, 1] == model_inputs[500, 1]],
                color="orange",
            )
            plt.scatter(
                model_inputs[:, 0][model_inputs[:, 1] == model_inputs[550, 1]],
                model_outputs[model_inputs[:, 1] == model_inputs[550, 1]],
                color="green",
            )
            plt.scatter(
                model_inputs[:, 0][model_inputs[:, 1] == model_inputs[350, 1]],
                model_outputs[model_inputs[:, 1] == model_inputs[350, 1]],
                color="blue",
            )
            plt.colorbar()
            # plt.plot(model_inputs[0:100, 0], model_outputs[0:100])
            plt.title("plotting just a couple of m values")
            plt.show()

            # look at how m is distributed
            plt.hist(model_inputs[:, 1], bins=100)
            plt.show()

            # now look at how noise/m is distributed
            plt.hist(0.01 / model_inputs[:, 1], bins=100)
            plt.xlabel("injected uncertainty in x")
            plt.show()
    x_train, x_val, y_train, y_val = DataPreparation.train_val_split(
        model_inputs, model_outputs, val_proportion=val_prop, random_state=rs
    )
    trainData = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    trainDataLoader = DataLoader(
        trainData, batch_size=BATCH_SIZE, shuffle=True
    )
    # set the device we will be using to train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get_item("model", "model_type", "DE")
    model, lossFn = models.model_setup_DE(
        config.get_item("model", "loss_type", "DE"),
        DEVICE,
        n_hidden=config.get_item("model", "n_hidden", "DE"),
        data_type=dim,
    )
    print(
        "save final checkpoint has this value",
        config.get_item("model", "save_final_checkpoint", "DE"),
    )
    print("model name is ", model_name)
    print("dim is ", dim)
    print("norm params", norm_params)
    model_ensemble = train.train_DE(
        trainDataLoader,
        x_val,
        y_val,
        config.get_item("model", "init_lr", "DE"),
        DEVICE,
        config.get_item("model", "loss_type", "DE"),
        config.get_item("model", "n_models", "DE"),
        norm_params,
        model_name=model_name,
        BETA=config.get_item("model", "BETA", "DE"),
        EPOCHS=config.get_item("model", "n_epochs", "DE"),
        path_to_model=config.get_item("common", "out_dir", "DE"),
        data_prescription=prescription,
        inject_type=injection,
        data_dim=dim,
        noise_level=noise,
        save_all_checkpoints=config.get_item(
            "model", "save_all_checkpoints", "DE"
        ),
        save_final_checkpoint=config.get_item(
            "model", "save_final_checkpoint", "DE"
        ),
        overwrite_final_checkpoint=config.get_item(
            "model", "overwrite_final_checkpoint", "DE"
        ),
        plot=config.get_item("model", "plot", "DE"),
        savefig=config.get_item("model", "savefig", "DE"),
        set_and_save_rs=config.get_item(
            "model", "save_chk_random_seed_init", "DE"
        ),
        rs_list=config.get_item("model", "rs_list", "DE"),
        save_n_hidden=config.get_item("model", "save_n_hidden", "DE"),
        n_hidden=config.get_item("model", "n_hidden", "DE"),
        save_size_df=config.get_item("model", "save_data_size", "DE"),
        size_df=size_df,
        verbose=verbose,
    )
