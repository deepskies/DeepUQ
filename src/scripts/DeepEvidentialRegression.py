import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scripts import train, models, io


def parse_args():
    parser = argparse.ArgumentParser(description="data handling module")
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
        default="low",
        help="low, medium, high or vhigh, \
            used to look up associated sigma value",
    )
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
        "--generatedata",
        action="store_true",
        default=False,
        help="option to generate df, if not specified \
            default behavior is to load from file",
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
        "--init_lr",
        type=float,
        required=False,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--coeff",
        type=float,
        required=False,
        default=0.5,
        help="Coeff, see DER lit",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        required=False,
        default="SDER",
        help="Loss types. \
              For MVE, options are no_var_loss, var_loss, \
              and bnn_loss. \
              For DER, options are DER or SDER",
    )
    parser.add_argument(
        "wd",
        type=str,
        help="Top level of directory, required arg",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="DER",
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
        action="store_true",
        default=False,
        help="option to save all checkpoints",
    )
    parser.add_argument(
        "--save_final_checkpoint",
        action="store_true",  # Set to True if argument is present
        default=False,  # Set default value to False if argument is not present
        help="option to save the final epoch checkpoint for each ensemble",
    )
    parser.add_argument(
        "--overwrite_final_checkpoint",
        action="store_true",
        default=False,
        help="option to overwite already saved checkpoints",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="option to plot in notebook",
    )
    parser.add_argument(
        "--savefig",
        action="store_true",
        default=False,
        help="option to save a figure of the true and predicted values",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
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
    if namespace.generatedata:
        # generate the df
        data = io.DataPreparation()
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
        loader = io.DataLoader()
        df = loader.load_data_h5(
            "linear_sigma_" + str(sigma) + "_size_" + str(size_df),
            path="/Users/rnevin/Documents/DeepUQ/data/",
        )
    print('df', df)
    len_df = len(df["params"][:, 0].numpy())
    len_x = len(df["inputs"].numpy())
    ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
    bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
    xs_array = np.tile(df["inputs"].numpy(), len_df)
    ys_array = np.reshape(df["output"].numpy(), (len_df * len_x))
    inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs = io.DataPreparation.normalize(inputs,
                                                               ys_array,
                                                               norm)
    x_train, x_val, y_train, y_val = io.DataPreparation.train_val_split(
        model_inputs, model_outputs, val_proportion=val_prop, random_state=rs
    )
    trainData = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    trainDataLoader = DataLoader(trainData,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
    print("[INFO] initializing the gal model...")
    # set the device we will be using to train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = namespace.model_type + "_noise_" + noise
    model, lossFn = models.model_setup_DER(namespace.loss_type, DEVICE)
    model_ensemble = train.train_DER(
        trainDataLoader,
        x_val,
        y_val,
        namespace.init_lr,
        DEVICE,
        namespace.coeff,
        namespace.loss_type,
        namespace.wd,
        model_name,
        EPOCHS=namespace.n_epochs,
        path_to_model=namespace.path_to_models,
        save_all_checkpoints=namespace.save_all_checkpoints,
        save_final_checkpoint=namespace.save_final_checkpoint,
        overwrite_final_checkpoint=namespace.overwrite_final_checkpoint,
        plot=namespace.plot,
        savefig=namespace.savefig,
        verbose=namespace.verbose,
    )
    '''
    trainDataLoader,
    x_val,
    y_val,
    INIT_LR,
    DEVICE,
    COEFF,
    loss_type,
    wd,
    model_name="DER",
    EPOCHS=100,
    path_to_model="models/",
    save_all_checkpoints=False,
    save_final_checkpoint=False,
    overwrite_final_checkpoint=False,
    plot=True,
    savefig=True,
    verbose=True
    '''
