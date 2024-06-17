import time
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
from utils.defaults import DefaultsDER
from data.data import DataPreparation, MyDataLoader

# from plots import Plots


def parse_args():
    parser = argparse.ArgumentParser(description="Runs DER")
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
        default=DefaultsDER["data"]["data_path"]
    )
    parser.add_argument(
        "--data_prescription",
        "-dp",
        default=DefaultsDER["data"]["data_prescription"]
    )
    parser.add_argument(
        "--data_engine",
        "-dl",
        default=DefaultsDER["data"]["data_engine"],
        choices=DataModules.keys(),
    )

    # model
    parser.add_argument("--out_dir", default=DefaultsDER["common"]["out_dir"])
    parser.add_argument(
        "--model_engine",
        "-e",
        default=DefaultsDER["model"]["model_engine"],
        choices=ModelModules.keys(),
    )
    parser.add_argument(
        "--size_df",
        type=float,
        required=False,
        default=DefaultsDER["data"]["size_df"],
        help="Used to load the associated .h5 data file,\
            number of lines generated",
    )
    parser.add_argument(
        "--noise_level",
        type=str,
        default=DefaultsDER["data"]["noise_level"],
        choices=["low", "medium", "high", "vhigh"],
        help="low, medium, high or vhigh, \
            used to look up associated sigma value",
    )
    parser.add_argument(
        "--normalize",
        required=False,
        action="store_true",
        default=DefaultsDER["data"]["normalize"],
        help="If true theres an option to normalize the dataset",
    )
    parser.add_argument(
        "--val_proportion",
        type=float,
        required=False,
        default=DefaultsDER["data"]["val_proportion"],
        help="Proportion of the dataset to use as validation",
    )
    parser.add_argument(
        "--randomseed",
        type=int,
        required=False,
        default=DefaultsDER["data"]["randomseed"],
        help="Random seed used for shuffling the training and validation set",
    )
    parser.add_argument(
        "--generatedata",
        action="store_true",
        default=DefaultsDER["data"]["generatedata"],
        help="option to generate df, if not specified \
            default behavior is to load from file",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        required=False,
        default=DefaultsDER["data"]["batchsize"],
        help="Size of batched used in the traindataloader",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        required=False,
        default=DefaultsDER["model"]["init_lr"],
        help="Learning rate",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        required=False,
        default=DefaultsDER["model"]["loss_type"],
        help="Loss types for DER",
    )
    parser.add_argument(
        "--COEFF",
        type=float,
        required=False,
        default=DefaultsDER["model"]["COEFF"],
        help="Coefficient for DER",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default=DefaultsDER["model"]["model_type"],
        help="Beginning of name for saved checkpoints and figures",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=DefaultsDER["model"]["n_epochs"],
        help="number of epochs for each MVE",
    )
    parser.add_argument(
        "--save_all_checkpoints",
        action="store_true",
        default=DefaultsDER["model"]["save_all_checkpoints"],
        help="option to save all checkpoints",
    )
    parser.add_argument(
        "--save_final_checkpoint",
        action="store_true",  # Set to True if argument is present
        default=DefaultsDER["model"]["save_final_checkpoint"],
        help="option to save the final epoch checkpoint for each ensemble",
    )
    parser.add_argument(
        "--overwrite_final_checkpoint",
        action="store_true",
        default=DefaultsDER["model"]["overwrite_final_checkpoint"],
        help="option to overwite already saved checkpoints",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=DefaultsDER["model"]["plot"],
        help="option to plot in notebook",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        default=DefaultsDER["model"]["savefig"],
        help="option to save a figure of the true and predicted values",
    )
    parser.add_argument(
        "--save_chk_random_seed_init",
        action="store_true",
        default=DefaultsDER["model"]["save_chk_random_seed_init"],
        help="option to save the chk with a random seed",
    )
    parser.add_argument(
        "--rs",
        type=int,
        default=DefaultsDER["model"]["rs"],
        help="random seed for the pytorch model initialization",
    )
    parser.add_argument(
        "--save_n_hidden",
        action="store_true",
        default=DefaultsDER["model"]["save_n_hidden"],
        help="save chk with the number of neurons in the hidden layer",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        required=False,
        default=DefaultsDER["model"]["n_hidden"],
        help="Number of hidden neurons in the hidden layer, default 64",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DefaultsDER["model"]["verbose"],
        help="verbose option for train",
    )
    args = parser.parse_args()
    if args.config is not None:
        config = Config(args.config)

    else:
        temp_config_prefix = DefaultsDER["common"]["temp_config"]
        # modify this to also have a timestamp
        # Get current timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")

        # Modify name with timestamp
        temp_config = temp_config_prefix.replace(".yml", f"_{timestamp}.yml")

        os.makedirs(os.path.dirname(temp_config), exist_ok=True)

        input_yaml = {
            "common": {"out_dir": args.out_dir},
            "model": {
                "model_path": args.out_dir,
                "model_engine": args.model_engine,
                "model_type": args.model_type,
                "loss_type": args.loss_type,
                "init_lr": args.init_lr,
                "COEFF": args.COEFF,
                "n_epochs": args.n_epochs,
                "save_all_checkpoints": args.save_all_checkpoints,
                "save_final_checkpoint": args.save_final_checkpoint,
                "overwrite_final_checkpoint": args.overwrite_final_checkpoint,
                "plot": args.plot,
                "savefig": args.savefig,
                "save_chk_random_seed_init": args.save_chk_random_seed_init,
                "rs": args.rs,
                "save_n_hidden": args.save_n_hidden,
                "n_hidden": args.n_hidden,
                "verbose": args.verbose,
            },
            "data": {
                "data_path": args.data_path,
                "data_engine": args.data_engine,
                "prescription": args.data_prescription,
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


if __name__ == "__main__":
    config = parse_args()
    size_df = int(config.get_item("data", "size_df", "DER"))
    noise = config.get_item("data", "noise_level", "DER")
    norm = config.get_item("data", "normalize", "DER", raise_exception=False)
    val_prop = config.get_item("data", "val_proportion", "DER")
    rs = config.get_item("data", "randomseed", "DER")
    BATCH_SIZE = config.get_item("data", "batchsize", "DER")
    sigma = DataPreparation.get_sigma(noise)
    path_to_data = config.get_item("data", "data_path", "DER")
    prescription = config.get_item("data", "prescription", "DER")
    if config.get_item("data", "generatedata", "DER", raise_exception=False):
        # generate the df
        data = DataPreparation()
        data.sample_params_from_prior(size_df)
        data.simulate_data(data.params, sigma, prescription)
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
            str(prescription) + "_sigma_" + str(sigma) + "_size_" + str(size_df),
            path=path_to_data,
        )
    len_df = len(df["params"][:, 0].numpy())
    len_x = len(df["inputs"].numpy())
    ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
    bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
    xs_array = np.tile(df["inputs"].numpy(), len_df)
    ys_array = np.reshape(df["output"].numpy(), (len_df * len_x))
    inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs = DataPreparation.normalize(
        inputs,
        ys_array,
        norm)
    x_train, x_val, y_train, y_val = DataPreparation.train_val_split(
        model_inputs, model_outputs, val_proportion=val_prop, random_state=rs
    )
    trainData = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    trainDataLoader = DataLoader(
        trainData,
        batch_size=BATCH_SIZE,
        shuffle=True)
    print("[INFO] initializing the gal model...")
    # set the device we will be using to train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get_item(
        "model",
        "model_type",
        "DER") + "_noise_" + noise
    model, lossFn = models.model_setup_DER(
        config.get_item("model", "loss_type", "DER"),
        DEVICE,
        n_hidden=config.get_item("model", "n_hidden", "DER")
    )
    model_ensemble = train.train_DER(
        trainDataLoader,
        x_val,
        y_val,
        config.get_item("model", "init_lr", "DER"),
        DEVICE,
        config.get_item("model", "COEFF", "DER"),
        config.get_item("model", "loss_type", "DER"),
        model_name,
        EPOCHS=config.get_item("model", "n_epochs", "DER"),
        path_to_model=config.get_item("common", "out_dir", "DER"),
        save_all_checkpoints=config.get_item(
            "model",
            "save_all_checkpoints",
            "DER"),
        save_final_checkpoint=config.get_item(
            "model",
            "save_final_checkpoint",
            "DER"),
        overwrite_final_checkpoint=config.get_item(
            "model", "overwrite_final_checkpoint", "DER"
        ),
        plot=config.get_item("model", "plot", "DER"),
        savefig=config.get_item("model", "savefig", "DER"),
        set_and_save_rs=config.get_item("model",
                                        "save_chk_random_seed_init",
                                        "DER"),
        rs=config.get_item("model", "rs", "DER"),
        save_n_hidden=config.get_item("model",
                                      "save_n_hidden",
                                      "DER"),
        n_hidden=config.get_item("model", "n_hidden", "DER"),
        verbose=config.get_item("model", "verbose", "DER"),
    )
