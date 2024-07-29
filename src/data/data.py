# Contains modules used to prepare a dataset
# with varying noise properties
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import torch
import h5py
from deepbench.astro_object import GalaxyObject


class MyDataLoader:
    def __init__(self):
        self.data = None

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

    def image_gen(
        self,
        image_size=100,
        amplitude=10,
        radius=10,
        center_x=50,
        center_y=50,
        theta=0,
        noise_level=0.0,
        simulation_name="linear_homoskedastic",
        inject_type="predictive",
        seed=42,
    ):
        image = GalaxyObject(
            image_dimensions=(image_size, image_size),
            amplitude=amplitude,
            noise_level=noise_level,
            ellipse=0.5,
            theta=theta,
            radius=radius,
        ).create_object(center_x=center_x, center_y=center_y)
        return image

    def simulate_data_2d(
        self,
        size_df,
        params,
        sigma,
        image_size=32,
        inject_type="predictive",
    ):
        image_size = 32
        image_array = np.zeros((size_df, image_size, image_size))
        total_brightness = []
        for i in range(size_df):
            image = self.image_gen(
                image_size=image_size,
                amplitude=params[i, 0],
                radius=params[i, 1],
                center_x=16,
                center_y=16,
                theta=params[i, 2],
                noise_level=0,
            )
            if inject_type == "predictive":
                image_array[i, :, :] = image
                total_brightness.append(
                    np.sum(image) + np.random.normal(loc=0, scale=sigma)
                )
            elif inject_type == "feature":
                noisy_image = image + np.random.normal(
                    loc=0, scale=sigma, size=(image_size, image_size)
                )
                image_array[i, :, :] = noisy_image
                total_brightness.append(np.sum(image))
            # we'll need the noisy image summed if we want to
            # do a comparison of y - y':
            # total_brightness_prop_noisy.append(np.sum(noisy_image))
        return image_array, total_brightness

    def simulate_data(
        self,
        thetas,
        sigma,
        simulation_name="linear_homoskedastic",
        x=np.linspace(0, 100, 101),
        inject_type="predictive",
        seed=42,
    ):
        if simulation_name == "linear_homoskedastic":
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
            x_noisy = np.zeros((len(x), thetas.shape[0]))
            y_noisy = np.zeros((len(x), thetas.shape[0]))
            y = np.zeros((len(x), thetas.shape[0]))
            for i in range(thetas.shape[0]):
                m, b = thetas[i, 0], thetas[i, 1]
                if inject_type == "predictive":
                    y_noisy[:, i] = m * x + b + ε[:, i]
                elif inject_type == "feature":
                    # y_prime[:, i] = m * (x + ε[:, i]) + b
                    y[:, i] = m * x + b
                    x_noisy[:, i] = x + ε[:, i]

        else:
            print(
                f"Error: Unknown simulation name '{simulation_name}'. \
                    No data generated."
            )
            return
        if inject_type == "predictive":
            # self.input = x
            self.input = torch.Tensor(np.tile(x, thetas.shape[0]).T)
            self.output = torch.Tensor(y_noisy.T)
            self.output_err = ε[:, i].T
        elif inject_type == "feature":
            self.input = torch.Tensor(x_noisy.T)
            self.output = torch.Tensor(y.T)
            self.output_err = ε[:, i].T
        print(
            f"{simulation_name} simulation data generated, \
                with noise injected type: {inject_type}."
        )

    def sample_params_from_prior(
        self, n_samples, low=[0, -10], high=[10, 10], n_params=2, seed=42
    ):
        assert (
            len(low) == len(high) == n_params
        ), "the length of the bounds must match that of the n_params"
        low_bounds = torch.tensor(low, dtype=torch.float32)
        high_bounds = torch.tensor(high, dtype=torch.float32)
        rs = np.random.RandomState(seed)  # 2147483648)#
        prior = rs.uniform(
            low=low_bounds, high=high_bounds, size=(n_samples, n_params)
        )
        """
        the prior way of doing this (lol)
        #print(np.shape(prior), prior)
        #prior = Uniform(low=low_bounds,
        #                high=high_bounds,
        #                seed=seed)
        # not random_seed, rs, or seed
        #params = prior.sample((n_samples,))
        """
        self.params = prior

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

    def get_sigma(noise, inject_type="predictive", data_dimension="0D"):
        """_summary_

        Args:
            noise (_type_): _description_
            inject_type (str, optional): _description_.
            Defaults to "predictive".
            data_dimension (str, optional): _description_.
            Defaults to "0D".

        Returns:
            _type_: the value of injected sigma, for the feature injection this
            is sigma_x, for the predictive injection, this is sigma_y
        """
        if inject_type == "predictive":
            if noise == "low":
                sigma = 1
            elif noise == "medium":
                sigma = 5
            elif noise == "high":
                sigma = 10
            elif noise == "vhigh":
                sigma = 100
            else:
                print("cannot find a match for this noise", noise)
        elif inject_type == "feature" and data_dimension == "0D":
            if noise == "low":
                sigma = 1 / 5
            elif noise == "medium":
                sigma = 5 / 5
            elif noise == "high":
                sigma = 10 / 5
        elif inject_type == "feature" and data_dimension == "2D":
            if noise == "low":
                sigma = 1 / 32
            elif noise == "medium":
                sigma = 5 / 32
            elif noise == "high":
                sigma = 10 / 32
        return sigma

    def normalize(inputs, ys_array, norm=False):
        if norm:
            # normalize everything before it goes into a network
            inputmin = np.min(inputs)  # , axis=0)
            inputmax = np.max(inputs)  # , axis=0)
            outputmin = np.min(ys_array)
            outputmax = np.max(ys_array)
            model_inputs = (inputs - inputmin) / (inputmax - inputmin)
            model_outputs = (ys_array - outputmin) / (outputmax - outputmin)
            # save the normalization parameters
            normalization_params = {
                "inputmin": inputmin,
                "inputmax": inputmax,
                "outputmin": outputmin,
                "outputmax": outputmax,
            }
        else:
            normalization_params = None
            model_inputs = inputs
            model_outputs = ys_array
        return model_inputs, model_outputs, normalization_params

    def train_val_split(
        model_inputs, model_outputs, val_proportion=0.1, random_state=42
    ):
        x_train, x_val, y_train, y_val = train_test_split(
            model_inputs,
            model_outputs,
            test_size=val_proportion,
            random_state=random_state,
        )
        return x_train, x_val, y_train, y_val
