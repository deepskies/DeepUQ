# Contains modules used to prepare a dataset
# with varying noise properties
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from torch.distributions import Uniform
from torch.utils.data import DataLoader, TensorDataset
import torch
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description="data handling module"
    )
    parser.add_argument(
        "size_df",
        type=float,
        required=False,
        default=1000,
        help="Used to load the associated .h5 data file",
    )
    parser.add_argument(
        "noise_level",
        type=str,
        required=False,
        default='low',
        help="low, medium, high or vhigh, used to look up associated sigma value",
    )
    parser.add_argument(
        "size_df",
        type=str,
        nargs="?",
        default="/repo/embargo",
        help="Butler Repository path from which data is transferred. \
            Input str. Default = '/repo/embargo'",
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
        type=float,
        required=False,
        default=42,
        help="Random seed used for shuffling the training and validation set",
    )
    parser.add_argument(
        "--batchsize",
        type=float,
        required=False,
        default=100,
        help="Size of batched used in the traindataloader",
    )
    return parser.parse_args()


class ModelLoader:
    def save_model_pkl(self, path, model_name, posterior):
        """
        Save the pkl'ed saved posterior model

        :param path: Location to save the model
        :param model_name: Name of the model
        :param posterior: Model object to be saved
        """
        file_name = path + model_name + ".pkl"
        with open(file_name, "wb") as file:
            pickle.dump(posterior, file)

    def load_model_pkl(self, path, model_name):
        """
        Load the pkl'ed saved posterior model

        :param path: Location to load the model from
        :param model_name: Name of the model
        :return: Loaded model object that can be used with the predict function
        """
        print(path)
        with open(path + model_name + ".pkl", "rb") as file:
            posterior = pickle.load(file)
        return posterior

    def predict(input, model):
        """

        :param input: loaded object used for inference
        :param model: loaded model
        :return: Prediction
        """
        return 0


class DataLoader:
    def save_data_pkl(self, data_name, data, path="../data/"):
        """
        Save and load the pkl'ed training/test set

        :param path: Location to save the model
        :param model_name: Name of the model
        :param posterior: Model object to be saved
        """
        file_name = path + data_name + ".pkl"
        with open(file_name, "wb") as file:
            pickle.dump(data, file)

    def load_data_pkl(self, data_name, path="../data/"):
        """
        Load the pkl'ed saved posterior model

        :param path: Location to load the model from
        :param model_name: Name of the model
        :return: Loaded model object that can be used with the predict function
        """
        print(path)
        with open(path + data_name + ".pkl", "rb") as file:
            data = pickle.load(file)
        return data

    def save_data_h5(self, data_name, data, path="../data/"):
        """
        Save data to an h5 file.

        :param path: Location to save the data
        :param data_name: Name of the data
        :param data: Data to be saved
        """
        data_arrays = {key: np.asarray(value) for key, value in data.items()}

        file_name = path + data_name + ".h5"
        with h5py.File(file_name, "w") as file:
            # Save each array as a dataset in the HDF5 file
            for key, value in data_arrays.items():
                file.create_dataset(key, data=value)

    def load_data_h5(self, data_name, path="../data/"):
        """
        Load data from an h5 file.

        :param path: Location to load the data from
        :param data_name: Name of the data
        :return: Loaded data
        """
        file_name = path + data_name + ".h5"
        loaded_data = {}
        with h5py.File(file_name, "r") as file:
            for key in file.keys():
                loaded_data[key] = torch.Tensor(file[key][...])
        return loaded_data


class DataPreparation:
    """
    A class for loading, preprocessing, and simulating datasets.

    Parameters:
    - file_path (str): The path to the dataset file.

    Methods:
    - load_data(): Load data from the specified file path.
    - preprocess_data(): Preprocess the loaded data.
    - simulate_data(simulation_name, num_samples=1000):
      Simulate data based on the specified simulation.
    - save_data(output_file='output_data.csv'): Save the current dataset to
      a CSV file.
    - get_data(): Retrieve the current dataset.

    Example Usage:
    ```
    dataset_manager = DatasetPreparation('your_dataset.csv')
    dataset_manager.load_data()
    dataset_manager.preprocess_data()
    dataset_manager.simulate_data('linear')
    dataset_manager.save_data('simulated_data.csv')
    simulated_data = dataset_manager.get_data()
    ```

    Note: Replace 'your_dataset.csv' with the actual dataset file path.
    """

    def __init__(self):
        self.data = None

    def simulate_data(
        self,
        thetas,
        sigma,
        simulation_name,
        x=np.linspace(0, 100, 101),
        seed=13
    ):
        if simulation_name == "linear_homogeneous":
            # convert to numpy array (if tensor):
            thetas = np.atleast_2d(thetas)
            # Check if the input has the correct shape
            if thetas.shape[1] != 2:
                raise ValueError(
                    "Input tensor must have shape (n, 2) where n is \
                        the number of parameter sets."
                )

            # Unpack the parameters
            if thetas.shape[0] == 1:
                # If there's only one set of parameters, extract them directly
                m, b = thetas[0, 0], thetas[0, 1]
            else:
                # If there are multiple sets of parameters,
                # extract them for each row
                m, b = thetas[:, 0], thetas[:, 1]
            rs = np.random.RandomState(seed)  # 2147483648)#
            # I'm thinking sigma could actually be a function of x
            # if we want to get fancy down the road
            # Generate random noise (epsilon) based
            # on a normal distribution with mean 0 and standard deviation sigma
            ε = rs.normal(loc=0, scale=sigma, size=(len(x), thetas.shape[0]))

            # Initialize an empty array to store the results
            # for each set of parameters
            y = np.zeros((len(x), thetas.shape[0]))
            for i in range(thetas.shape[0]):
                m, b = thetas[i, 0], thetas[i, 1]
                y[:, i] = m * x + b + ε[:, i]
            # simulated_data = pd.DataFrame({'Feature': x, 'Target': y})
            print("Linear simulation data generated.")
        elif simulation_name == "quadratic":
            # Example quadratic simulation
            y = 3 * x**2 + 2 * x + 1 + np.random.normal(0, 1, len(x))
        else:
            print(
                f"Error: Unknown simulation name '{simulation_name}'. \
                    No data generated."
            )
            return
        self.input = x
        self.output = torch.Tensor(y.T)
        self.output_err = ε[:, i]
        # self.data = simulated_data

    def sample_params_from_prior(self, n_samples):
        low_bounds = torch.tensor([0, -10], dtype=torch.float32)
        high_bounds = torch.tensor([10, 10], dtype=torch.float32)
        prior = Uniform(low=low_bounds, high=high_bounds)
        params = prior.sample((n_samples,))
        self.params = params

    def get_dict(self):
        data_dict = {
            "params": self.params,
            "inputs": self.input,
            "output": self.output,
            "output_err": self.output_err,
        }
        return data_dict

    def get_data(self):
        return self.data

    def get_sigma(noise):
        if noise == 'low':
            sigma = 1
        if noise == 'medium':
            sigma = 5
        if noise == 'high':
            sigma = 10
        if noise == 'vhigh':
            sigma = 100
        return sigma
    
    def normalize(inputs,
                  ys_array,
                  norm=False):
        if norm:
            # normalize everything before it goes into a network
            inputmin = np.min(inputs, axis=0)
            inputmax = np.max(inputs, axis=0)
            outputmin = np.min(ys_array)
            outputmax = np.max(ys_array)
            model_inputs = (inputs - inputmin) / (inputmax - inputmin)
            model_outputs = (ys_array - outputmin) / (outputmax - outputmin)
        else:
            model_inputs = inputs
            model_outputs = ys_array
        return model_inputs, model_outputs
    
    def train_val_split(model_inputs,
                        model_outputs,
                        val_proportion=0.1,
                        random_state=42):
        x_train, x_val, y_train, y_val = train_test_split(model_inputs,
                                                          model_outputs,
                                                          test_size=val_proportion,
                                                          random_state=random_state)
        return x_train, x_val, y_train, y_val


# Example usage:
if __name__ == "__main__":
    namespace = parse_args()
    size_df = namespace.size_df
    noise = namespace.noise_level
    norm = namespace.normalize
    val_prop = namespace.val_proportion
    rs = namespace.randomseed
    BATCH_SIZE = namespace.batchsize
    sigma = DataPreparation.get_sigma(noise)
    loader = DataLoader()
    data = loader.load_data_h5('linear_sigma_'+str(sigma)+'_size_'+str(size_df))
    len_df = len(data['params'][:, 0].numpy())
    len_x = len(data['inputs'].numpy())
    ms_array = np.repeat(data['params'][:, 0].numpy(), len_x)
    bs_array = np.repeat(data['params'][:, 1].numpy(), len_x)
    xs_array = np.tile(data['inputs'].numpy(), len_df)
    ys_array = np.reshape(data['output'].numpy(), (len_df * len_x))
    inputs = np.array([xs_array, ms_array, bs_array]).T
    model_inputs, model_outputs = DataPreparation.normalize(inputs,
                              ys_array,
                              norm)
    x_train, x_val, y_train, y_val = DataPreparation.train_val_split(model_inputs,
                                    model_outputs,
                                    test_size=val_prop,
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
    '''
    #return trainDataLoader, x_val, y_val
