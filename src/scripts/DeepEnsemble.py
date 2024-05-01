import os
import yaml
import argparse
import numpy as np
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

# from plots import Plots


def parse_args():
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
        temp_config = DefaultsDE["common"]["temp_config"]
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
                "verbose": args.verbose,
            },
            "data": {
                "data_path": args.data_path,
                "data_engine": args.data_engine,
                "size_df": args.size_df,
                "noise_level": args.noise_level,
                "val_proportion": args.val_proportion,
                "randomseed": args.randomseed,
                "batchsize": args.batchsize,
            },
            # "plots": {key: {} for key in args.plots},
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
    size_df = int(config.get_item("data", "size_df", "DE"))
    noise = config.get_item("data", "noise_level", "DE")
    norm = config.get_item("data", "normalize", "DE", raise_exception=False)
    val_prop = config.get_item("data", "val_proportion", "DE")
    rs = config.get_item("data", "randomseed", "DE")
    BATCH_SIZE = config.get_item("data", "batchsize", "DE")
    sigma = DataPreparation.get_sigma(noise)
    path_to_data = config.get_item("data", "data_path", "DE")
    if config.get_item("data", "generatedata", "DE", raise_exception=False):
        # generate the df
        data = DataPreparation()
        data.sample_params_from_prior(size_df)
        data.simulate_data(data.params, sigma, "linear_homogeneous")
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
    else:
        loader = MyDataLoader()
        df = loader.load_data_h5(
            "linear_sigma_" + str(sigma) + "_size_" + str(size_df),
            path=path_to_data,
        )
    len_df = len(df["params"][:, 0].numpy())
    len_x = len(df["inputs"].numpy())
    ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
    bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
    xs_array = np.tile(df["inputs"].numpy(), len_df)
    ys_array = np.reshape(df["output"].numpy(), (len_df * len_x))

    inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs = DataPreparation.normalize(inputs,
                                                            ys_array,
                                                            norm)
    x_train, x_val, y_train, y_val = DataPreparation.train_val_split(
        model_inputs, model_outputs, val_proportion=val_prop, random_state=rs
    )
    trainData = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    trainDataLoader = DataLoader(trainData,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
    # set the device we will be using to train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get_item("model",
                                 "model_type",
                                 "DE") + "_noise_" + noise
    model, lossFn = models.model_setup_DE(
        config.get_item("model", "loss_type", "DE"), DEVICE
    )
    print(
        "save final checkpoint has this value",
        config.get_item("model", "save_final_checkpoint", "DE"),
    )
    model_ensemble = train.train_DE(
        trainDataLoader,
        x_val,
        y_val,
        config.get_item("model", "init_lr", "DE"),
        DEVICE,
        config.get_item("model", "loss_type", "DE"),
        config.get_item("model", "n_models", "DE"),
        model_name,
        BETA=config.get_item("model", "BETA", "DE"),
        EPOCHS=config.get_item("model", "n_epochs", "DE"),
        path_to_model=config.get_item("common", "out_dir", "DE"),
        save_all_checkpoints=config.get_item("model",
                                             "save_all_checkpoints",
                                             "DE"),
        save_final_checkpoint=config.get_item("model",
                                              "save_final_checkpoint",
                                              "DE"),
        overwrite_final_checkpoint=config.get_item(
            "model", "overwrite_final_checkpoint", "DE"
        ),
        plot=config.get_item("model", "plot", "DE"),
        savefig=config.get_item("model", "savefig", "DE"),
        verbose=config.get_item("model", "verbose", "DE"),
    )
