import time
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

# from scripts import train, models, io
from deepuq.train import train
from deepuq.models import models
from deepuq.data import DataModules
from deepuq.models import ModelModules
from deepuq.utils.config import Config
from deepuq.utils.defaults import DefaultsDER
from deepuq.data.data import DataPreparation, MyDataLoader

# from plots import Plots


def parse_args():
    """Parses command-line (or config) arguments for the
    DeepEvidentialRegression script.

    This function creates an argument parser that supports:
    1) Reading from a YAML configuration file (via --config).
    2) Reading arguments from the command line and using default values
       if not provided, with the option to dump arguments to a temporary
       YAML configuration file.
    3) Specifying data-related parameters such as the data path, dimension,
       and injection method, as well as model parameters like the DER
       coefficient, learning rate, and loss type.

    The parser supports the following argument categories:
    - Data-related arguments:
        --data_path, --data_dimension, --data_injection,
        --data_engine, --size_df, --noise_level, --val_proportion,
        --randomseed, --generatedata, --batchsize, --normalize, --uniform
    - Model-related arguments:
        --model_engine, --init_lr, --loss_type, --COEFF, --model_type,
        --n_epochs, --save_all_checkpoints, --save_final_checkpoint,
        --overwrite_final_checkpoint, --plot, --savefig,
        --save_chk_random_seed_init, --rs_list, --n_hidden, --save_n_hidden,
        --save_data_size, --verbose
    - General arguments:
        --config, --out_dir

    Returns:
        Config: Configuration object that combines parsed arguments,
        either from the command line or a YAML configuration file.
    """
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
        default=DefaultsDER["data"]["data_path"],
    )
    parser.add_argument(
        "--data_dimension",
        "-dd",
        default=DefaultsDER["data"]["data_dimension"],
    )
    parser.add_argument(
        "--data_injection",
        "-di",
        default=DefaultsDER["data"]["data_injection"],
    )
    parser.add_argument(
        "--data_engine",
        "-dl",
        default=DefaultsDER["data"]["data_engine"],
        choices=DataModules.keys(),
    )

    # model
    parser.add_argument(
        "--out_dir",
        default=DefaultsDER["common"]["out_dir"],
    )
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
        "--uniform",
        required=False,
        action="store_true",
        default=DefaultsDER["data"]["uniform"],
        help="If true theres an option to uniformize the dataset",
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
        help="option to generate data, if not specified \
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
        "--save_data_size",
        action="store_true",
        default=DefaultsDER["model"]["save_data_size"],
        help="save chk with the number of examples in the dataset",
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

        print(
            "Reading settings from cli and default, \
              dumping to temp config: ",
            temp_config,
        )

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
                "save_data_size": args.save_data_size,
                "verbose": args.verbose,
            },
            "data": {
                "data_path": args.data_path,
                "data_engine": args.data_engine,
                "data_dimension": args.data_dimension,
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


def main():
    """Main execution script for the DeepEvidentialRegression pipeline.

    This script performs the following steps:
    1. Parses command-line arguments and configuration settings using
    `parse_args`.
    2. Prepares and/or loads data based on the specified data dimensionality
    (0D or 2D).
    3. Processes and normalizes the data, applying optional noise injection.
    4. Visualizes the data distribution if verbose mode is enabled.
    5. Splits the data into training and validation sets.
    6. Trains a model using the Deep Evidential Regression framework and
    specified configurations.
    7. Optionally saves checkpoints and plots during the training process.

    Execution starts if the script is run as a standalone module.
    """
    config = parse_args()
    verbose = config.get_item("model", "verbose", "DER")
    size_df = int(config.get_item("data", "size_df", "DER"))
    noise = config.get_item("data", "noise_level", "DER")
    norm = config.get_item("data", "normalize", "DER", raise_exception=False)
    uniform = config.get_item("data", "uniform", "DER", raise_exception=False)
    val_prop = config.get_item("data", "val_proportion", "DER")
    rs = config.get_item("data", "randomseed", "DER")
    BATCH_SIZE = config.get_item("data", "batchsize", "DER")
    path_to_data = config.get_item("data", "data_path", "DER")
    injection = config.get_item("data", "data_injection", "DER")
    assert (
        injection == "input" or injection == "output"
    ), f"data injection type must be 'input' or 'output' and is {injection}"
    dim = config.get_item("data", "data_dimension", "DER")
    assert (
        dim == "0D" or dim == "2D"
    ), f"data dimension must be '0D' or '2D' and is {dim}"
    data = DataPreparation()
    if config.get_item("data", "generatedata", "DER", raise_exception=False):
        print("generating the data")
        model_inputs, model_outputs = data.generate_df(
            size_df, noise, dim, injection, uniform, verbose
        )
    else:
        print("loading data from file")
        loader = MyDataLoader()
        filename = (
            str(dim)
            + "_"
            + str(injection)
            + "_noise_"
            + noise
            + "_size_"
            + str(size_df)
        )
        df = loader.load_data_h5(filename, path=path_to_data)
        if dim == "2D":
            model_inputs = df["input"].numpy()
            model_outputs = df["output"].numpy()
        if dim == "0D":
            len_df = len(df["params"][:, 0].numpy())
            len_x = np.shape(df["output"])[1]
            ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
            bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
            xs_array = np.reshape(df["input"].numpy(), (len_df * len_x))
            model_inputs = np.array([xs_array, ms_array, bs_array]).T
            model_outputs = np.reshape(df["output"].numpy(), (len_df * len_x))
    print(
        "shape of input",
        np.shape(model_inputs),
        "shape of output",
        np.shape(model_outputs),
        "type of input",
        type(model_inputs),
    )
    model_inputs, model_outputs, norm_params = data.normalize(
        model_inputs, model_outputs, norm
    )
    if verbose:
        plt.clf()
        plt.hist(model_outputs)
        plt.axvline(x=np.mean(model_outputs), color="yellow")
        plt.xlabel("output variable")
        plt.annotate(
            "mean output variable (should be ~1) = "
            + str(np.mean(model_outputs)),
            xy=(0.02, 0.9),
            xycoords="axes fraction",
        )
        plt.show()
        if dim == "2D":
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
            plt.xlabel("model input")
            plt.ylabel("model output")
            plt.colorbar()
            plt.title("a selection of x and y, colorbar is m value")
            plt.show()

    x_train, x_val, y_train, y_val = data.train_val_split(
        model_inputs,
        model_outputs,
        val_proportion=val_prop,
        random_state=rs,
    )
    trainData = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    trainDataLoader = DataLoader(
        trainData, batch_size=BATCH_SIZE, shuffle=True
    )
    # set the device we will be using to train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get_item("model", "model_type", "DER")
    model, lossFn = models.model_setup_DER(
        config.get_item("model", "loss_type", "DER"),
        DEVICE,
        n_hidden=config.get_item("model", "n_hidden", "DER"),
        data_type=dim,
    )
    print("model name is ", model_name)
    train.train_DER(
        trainDataLoader,
        x_val,
        y_val,
        config.get_item("model", "init_lr", "DER"),
        DEVICE,
        config.get_item("model", "COEFF", "DER"),
        config.get_item("model", "loss_type", "DER"),
        norm_params,
        model_name=model_name,
        EPOCHS=config.get_item("model", "n_epochs", "DER"),
        path_to_model=config.get_item("common", "out_dir", "DER"),
        inject_type=injection,
        data_dim=dim,
        noise_level=noise,
        save_all_checkpoints=config.get_item(
            "model", "save_all_checkpoints", "DER"
        ),
        save_final_checkpoint=config.get_item(
            "model", "save_final_checkpoint", "DER"
        ),
        overwrite_final_checkpoint=config.get_item(
            "model", "overwrite_final_checkpoint", "DER"
        ),
        plot=config.get_item("model", "plot", "DER"),
        savefig=config.get_item("model", "savefig", "DER"),
        set_and_save_rs=config.get_item(
            "model", "save_chk_random_seed_init", "DER"
        ),
        rs=config.get_item("model", "rs", "DER"),
        save_n_hidden=config.get_item("model", "save_n_hidden", "DER"),
        n_hidden=config.get_item("model", "n_hidden", "DER"),
        save_size_df=config.get_item("model", "save_data_size", "DER"),
        size_df=size_df,
        verbose=config.get_item("model", "verbose", "DER"),
    )


if __name__ == "__main__":
    main()
