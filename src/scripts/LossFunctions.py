import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


from data import DataModules
from models import ModelModules
from utils.config import Config
from utils.defaults import DefaultsAnalysis, DefaultsDE
from data.data import DataPreparation, MyDataLoader
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

    # data info
    parser.add_argument(
        "--data_path",
        "-d",
        default=DefaultsAnalysis["data"]["data_path"],
    )
    parser.add_argument(
        "--data_engine",
        "-dl",
        default=DefaultsAnalysis["data"]["data_engine"],
        choices=DataModules.keys(),
    )

    # model
    # path to save the model results
    parser.add_argument("--out_dir",
                        default=DefaultsAnalysis["common"]["out_dir"])

    # now args for model
    parser.add_argument(
        "--n_models",
        type=int,
        default=DefaultsDE["model"]["n_models"],
        help="Number of MVEs in the ensemble",
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
        "--noise_level_list",
        type=str,
        required=False,
        default=DefaultsAnalysis["analysis"]["noise_level_list"],
        help="Noise levels to compare",
    )
    parser.add_argument(
        "--model_names_list",
        type=str,
        required=False,
        default=DefaultsAnalysis["analysis"]["model_names_list"],
        help="Beginning of name for saved checkpoints and figures",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=DefaultsAnalysis["analysis"]["n_epochs"],
        help="number of epochs",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=DefaultsAnalysis["analysis"]["plot"],
        help="option to plot in notebook",
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
        # if not, default is from DefaultsDE dictionary
        input_yaml = {
            "common": {"out_dir": args.out_dir},
            "data": {
                "data_path": args.data_path,
                "data_engine": args.data_engine,
            },
            "model": {"n_models": args.n_models,
                      "BETA": args.BETA},
            "analysis": {"noise_level_list": args.noise_level_list,
                         "model_names_list": args.model_names_list,
                         "n_epochs": args.n_epochs,
                         "plot": args.plot,
                         "savefig": args.savefig,
                         "verbose": args.verbose,}
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_list = config.get_item("analysis", "noise_level_list", "Analysis")
    sigma_list = []
    for noise in noise_list:
        sigma_list.append(DataPreparation.get_sigma(noise))
    print('noise list', noise_list)
    print('sigma list', sigma_list)
    path_to_chk = config.get_item("common", "out_dir", "Analysis")
    model_name_list = config.get_item("analysis",
                                      "model_names_list",
                                      "Analysis")
    print('model list', model_name_list)
    for noise in noise_list:
        for model in model_name_list:
            # now run the analysis on the resulting checkpoints
            chk_module = AggregateCheckpoints()
            if model == "DER":
                for epoch in range(config.get_item("analysis",
                                                   "n_epochs",
                                                   "Analysis")):
                    chk = chk_module.load_checkpoint(
                                    model,
                                    noise,
                                    epoch,
                                    config.get_item("model", "BETA", "DE"),
                                    DEVICE,
                    )
                                    #path=path_to_chk)
                    # things to grab: 'valid_mse' and 'valid_bnll'
                    print(chk)
                    mse_loss.append(chk['valid_mse'])
                    loss.append(chk['valid_loss'])
            elif model[0:2] == "DE":
                loss = []
                mse_loss = []
                n_epochs = config.get_item("analysis",
                                                    "n_epochs",
                                                    "Analysis")
                n_models = config.get_item("model", "n_models", "DE")
                print('n_models', n_models)
                print('n_epochs', n_epochs)
                for nmodel in range(n_models):
                    mse_loss_one_model = []
                    loss_one_model = []
                    for epoch in range(n_epochs):
                        try:
                            chk = chk_module.load_checkpoint(
                                            model,
                                            noise,
                                            epoch,
                                            config.get_item("model", "BETA", "DE"),
                                            DEVICE,
                                            nmodel=nmodel,
                            )
                        except FileNotFoundError:
                            continue
                                        #path=path_to_chk)
                        # things to grab: 'valid_mse' and 'valid_bnll'
                        # print(chk)
                        mse_loss_one_model.append(chk['valid_mse'])
                        loss_one_model.append(chk['valid_loss'])
                    mse_loss.append(mse_loss_one_model)
                    loss.append(loss_one_model)
                print('loss', loss, loss[0], len(loss[0]))
                print('shape of loss', np.shape(loss))
                plt.clf()
                for i in range(n_models):
                    plt.scatter(range(n_epochs), mse_loss[i], label = 'Model '+str(i))
                plt.ylabel('MSE')
                plt.xlabel('epochs')
                plt.show()
                STOP
    
