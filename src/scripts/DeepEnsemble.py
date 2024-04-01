import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scripts import train, models, analysis, io


def beta_type(value):
    if isinstance(value, float):
        return value
    elif value.lower() == 'linear_decrease':
        return value
    elif value.lower() == 'step_decrease_to_0.5':
        return value
    elif value.lower() == 'step_decrease_to_1.0':
        return value
    else:
        raise argparse.ArgumentTypeError("BETA must be a float or one of 'linear_decrease', 'step_decrease_to_0.5', 'step_decrease_to_1.0'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="data handling module"
    )
    parser.add_argument(
        "--size_df",
        type=float,
        required=False,
        default=1000,
        help="Used to load the associated .h5 data file",
    )
    parser.add_argument(
        "noise_level",
        type=str,
        default='low',
        help="low, medium, high or vhigh, used to look up associated sigma value",
    )
    '''
    parser.add_argument(
        "size_df",
        type=str,
        nargs="?",
        default="/repo/embargo",
        help="Butler Repository path from which data is transferred. \
            Input str. Default = '/repo/embargo'",
    )
    '''
    parser.add_argument(
        "--normalize",
        required=False,
        action="store_true",
        help="If true theres an option to normalize the dataset",
    )
    parser.add_argument(
        "--val_proportion",
        type=float,
        required=False,
        default=0.1,
        help="Proportion of the dataset to use as validation",
    )
    parser.add_argument(
        "--randomseed",
        type=int,
        required=False,
        default=42,
        help="Random seed used for shuffling the training and validation set",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        required=False,
        default=100,
        help="Size of batched used in the traindataloader",
    )
    # now args for model
    parser.add_argument(
        "n_models",
        type=int,
        default=100,
        help="Number of MVEs in the ensemble",
    )
    parser.add_argument(
        "--init_lr",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        required=False,
        default="bnll_loss",
        help="Loss types for MVE, options are no_var_loss, var_loss, and bnn_loss",
    )
    parser.add_argument(
        "--BETA",
        type=beta_type,
        required=False,
        default=0.5,
        help="If loss_type is bnn_loss, specify a beta as a float or there are string options: linear_decrease, step_decrease_to_0.5, and step_decrease_to_1.0",
    )
    parser.add_argument(
        "wd",
        type=str,
        help="Top level of directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="DE",
        help="Beginning of name for saved checkpoints and figures",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=100,
        help="number of epochs for each MVE",
    )
    parser.add_argument(
        "--path_to_models",
        type=str,
        required=False,
        default="models/",
        help="path to where the checkpoints are saved",
    )
    parser.add_argument(
        "--save_all_checkpoints",
        type=bool,
        required=False,
        default=False,
        help="option to save all checkpoints",
    )
    parser.add_argument(
        "--save_final_checkpoints",
        type=bool,
        required=False,
        default=False,
        help="option to save the final epoch checkpoint for each ensemble",
    )
    parser.add_argument(
        "--overwrite_final_checkpoints",
        type=bool,
        required=False,
        default=False,
        help="option to overwite already saved checkpoints",
    )
    parser.add_argument(
        "--plot",
        type=bool,
        required=False,
        default=False,
        help="option to plot in notebook",
    )
    parser.add_argument(
        "--savefig",
        type=bool,
        required=False,
        default=True,
        help="option to save a figure of the true and predicted values",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=False,
        help="verbose option for train",
    )
    return parser.parse_args()


if __name__ == "__main__":
    namespace = parse_args()
    size_df = namespace.size_df
    noise = namespace.noise_level
    norm = namespace.normalize
    val_prop = namespace.val_proportion
    rs = namespace.randomseed
    BATCH_SIZE = namespace.batchsize
    sigma = io.DataPreparation.get_sigma(noise)
    loader = io.DataLoader()
    data = loader.load_data_h5('linear_sigma_'+str(sigma)+'_size_'+str(size_df),
                               path='/Users/rnevin/Documents/DeepUQ/data/')
    len_df = len(data['params'][:, 0].numpy())
    len_x = len(data['inputs'].numpy())
    ms_array = np.repeat(data['params'][:, 0].numpy(), len_x)
    bs_array = np.repeat(data['params'][:, 1].numpy(), len_x)
    xs_array = np.tile(data['inputs'].numpy(), len_df)
    ys_array = np.reshape(data['output'].numpy(), (len_df * len_x))
    inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs = io.DataPreparation.normalize(inputs,
                                                               ys_array,
                                                               norm)
    x_train, x_val, y_train, y_val = io.DataPreparation.train_val_split(model_inputs,
                                                                        model_outputs,
                                                                        val_proportion=val_prop,
                                                                        random_state=rs)
    trainData = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    trainDataLoader = DataLoader(trainData,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
    '''
    valData = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    valDataLoader = DataLoader(valData,
                               batch_size=BATCH_SIZE)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE
    
    return trainDataLoader, x_val, y_val
    '''
    print("[INFO] initializing the gal model...")
    # set the device we will be using to train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = namespace.model_type + '_noise_' + noise
    model, lossFn = models.model_setup_DE(namespace.loss_type, DEVICE)
    model_ensemble = train.train_DE(trainDataLoader,
                                    x_val,
                                    y_val,
                                    namespace.init_lr,
                                    DEVICE,
                                    namespace.loss_type,
                                    namespace.n_models,
                                    namespace.wd,
                                    model_name,
                                    BETA=namespace.BETA,
                                    EPOCHS=namespace.n_epochs,
                                    path_to_model=namespace.path_to_models,
                                    save_all_checkpoints=namespace.save_all_checkpoints,
                                    save_final_checkpoint=namespace.save_final_checkpoints,
                                    overwrite_final_checkpoint=namespace.overwrite_final_checkpoints,          
                                    plot=namespace.plot,
                                    savefig=namespace.savefig,
                                    verbose=namespace.verbose
                                    )


