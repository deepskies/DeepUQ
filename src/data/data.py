# Contains modules used to prepare a dataset
# with varying noise properties
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import h5py
from deepbench.astro_object import GalaxyObject
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset


class MyDataLoader:
    """A class for loading and saving data in HDF5 format.

    This class provides methods to save data as HDF5 files and load data from
    HDF5 files. It simplifies the process of managing datasets, allowing for
    efficient storage and retrieval.

    Attributes:
        data (Any): Placeholder for the data being loaded or saved.

    Methods:
        save_data_h5(data_name, data, path="../data/"):
            Saves the provided data as an HDF5 file with the specified name
            at the given path.

        load_data_h5(data_name, path="../data/"):
            Loads data from an HDF5 file with the specified name from the
            given path. Returns the loaded data as a dictionary, where each
            key corresponds to a dataset in the HDF5 file.
    """

    def __init__(self):
        self.data = None

    def save_data_h5(self, data_name, data, path="../data/"):
        """Save the provided data as an HDF5 file.

        This method converts the input data dictionary into NumPy arrays and
        stores each array as a dataset in an HDF5 file. The file is saved
        with the specified name at the given path.

        Parameters:
            data_name (str): The name of the HDF5 file (without extension) to
                save the data.
            data (dict): A dictionary containing the data to be saved, where
                keys are the dataset names and values are the corresponding
                data arrays.
            path (str, optional): The directory path where the HDF5 file will
                be saved. Defaults to "../data/".
        """
        data_arrays = {key: np.asarray(value) for key, value in data.items()}

        file_name = path + data_name + ".h5"
        with h5py.File(file_name, "w") as file:
            # Save each array as a dataset in the HDF5 file
            for key, value in data_arrays.items():
                file.create_dataset(key, data=value)

    def load_data_h5(self, data_name, path="../data/"):
        """Load data from an HDF5 file.

        This method reads the specified HDF5 file and loads its contents into a
        dictionary, where each key corresponds to a dataset in the file and the
        values are stored as PyTorch tensors.

        Parameters:
            data_name (str): The name of the HDF5 file (without extension)
                from which to load the data.
            path (str, optional): The directory path where the HDF5 file is
                located. Defaults to "../data/".

        Returns:
            dict: A dictionary containing the loaded data, with keys
                representing the dataset names and values as PyTorch tensors.
        """
        file_name = path + data_name + ".h5"
        loaded_data = {}
        with h5py.File(file_name, "r") as file:
            for key in file.keys():
                loaded_data[key] = torch.Tensor(file[key][...])
        return loaded_data


class DataPreparation:
    """A class for preparing and simulating data for modeling purposes.

    This class provides methods to normalize data, sample parameters from
    priors, simulate data with added noise, and select uniform samples from
    datasets. It supports both one-dimensional and two-dimensional data
    simulations, allowing for the generation of synthetic datasets for testing
    and training models.

    Attributes:
        data (Any): Placeholder for the data being prepared or simulated.

    Methods:
        select_uniform(model_inputs, model_outputs, dim, verbose=False, rs=40):
            Selects a uniform subset of inputs and outputs from the given data
            based on specified bins.

        simulate_data(thetas, sigma, x=np.linspace(0, 10, 100),
                      inject_type="output", seed=42, vary_sigma=False,
                      verbose=False):
            Simulates data based on parameter sets, adding noise as specified.

        simulate_data_2d(size_df, params, sigma, image_size=32,
                         inject_type="output", rs=40):
            Simulates two-dimensional data using specified parameters and
            noise levels.

        image_gen(image_size=100, amplitude=10, radius=10, center_x=50,
                  center_y=50, theta=0, noise_level=0.0, inject_type="output",
                  seed=42):
            Generates a 2D image based on specified parameters.

        sample_params_from_prior(n_samples, low=[0.1, 0], high=[0.4, 0],
                                 n_params=2, seed=42):
            Samples parameter sets uniformly from a specified prior
            distribution.

        normalize(inputs, ys_array, norm=False):
            Normalizes the inputs and outputs based on specified normalization
            parameters.

        get_sigma(noise, inject_type="output", data_dimension="0D"):
            Determines the appropriate sigma value based on noise level and
            injection type.

        get_sigma_m(noise, m):
            Computes sigma for a given noise level and parameter m.

        get_dict():
            Returns a dictionary representation of the current parameters and
            data.

        get_data():
            Returns the raw data stored in the instance.
    """

    def __init__(self):
        self.data = None

    def generate_df(self, size_df, noise, dim, injection, uniform, verbose):
        if verbose:
            print("generating dataframe")
        if uniform:
            if verbose:
                print(
                    "inflating starting size because sub-selecting \
                      uniform"
                )
            size_df_gen = 5 * size_df
        else:
            size_df_gen = size_df
        if dim == "0D":
            self.sample_params_from_prior(size_df_gen)
            if verbose:
                print("injecting this noise", noise)
                print(
                    f"inject type is {injection}, \
                      dim is {dim}, noise is {noise}"
                )
            if injection == "input":
                vary_sigma = True
                if verbose:
                    print("are we varying sigma", vary_sigma)
                self.simulate_data(
                    self.params,
                    noise,
                    x=np.linspace(0, 10, 100),
                    inject_type=injection,
                    vary_sigma=vary_sigma,
                )
            elif injection == "output":
                sigma = self.get_sigma(
                    noise,
                    inject_type=injection,
                    data_dimension=dim,
                )
                self.simulate_data(
                    self.params,
                    sigma,
                    x=np.linspace(0, 10, 100),
                    inject_type=injection,
                )
            df_array = self.get_dict()
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
            sigma = self.get_sigma(
                noise,
                inject_type=injection,
                data_dimension=dim,
            )
            self.sample_params_from_prior(
                size_df_gen,
                low=[0, 1, -1.5],
                high=[0.01, 10, 1.5],
                n_params=3,
                seed=42,
            )
            model_inputs, model_outputs = self.simulate_data_2d(
                size_df_gen,
                self.params,
                sigma,
                image_size=32,
                inject_type=injection,
            )
        if dim == "0D":
            len_df = len(df["params"][:, 0].numpy())
            len_x = np.shape(df["output"])[1]
            ms_array = np.repeat(df["params"][:, 0].numpy(), len_x)
            bs_array = np.repeat(df["params"][:, 1].numpy(), len_x)
            xs_array = np.reshape(df["input"].numpy(), (len_df * len_x))
            model_inputs = np.array([xs_array, ms_array, bs_array]).T
            model_outputs = np.reshape(df["output"].numpy(), (len_df * len_x))
        if uniform:
            model_inputs, model_outputs = self.select_uniform(
                model_inputs,
                model_outputs,
                size_df,
                verbose=verbose,
                rs=40,
            )
            if verbose:
                print("size after uniform", np.shape(model_inputs))
        return model_inputs, model_outputs

    def select_uniform(
        self,
        model_inputs,
        model_outputs,
        size_df,
        num_bins=10,
        verbose=False,
        rs=40,
    ):
        """Selects a uniform subset of data for the output variable by
        sampling from the output distribution across defined bins.

        This function divides the `model_outputs` into uniform bins and
        randomly samples from each bin to ensure a balanced representation of
        output values. The number of samples per bin depends on the desired
        size of the dataframe and the number of bins, and the subset
        is returned for both `model_inputs` and `model_outputs`.

        Args:
            model_inputs (np.ndarray): Input data from which the subset will
                be selected.
            model_outputs (np.ndarray): Output data used for binning and
                sampling.
            size_df (float): The desired total size of the returned dataframe.
            num_bins (int): Number of bins to use to select the uniform
                distribution.
            verbose (bool): If True, the function prints debug information and
                plots the distribution of the selected subset. Default is
                False.
            rs (int): Random seed for reproducibility. Default is 40.

        Returns:
            np.ndarray: A subset of `model_inputs` selected uniformly based on
                `model_outputs`.
            np.ndarray: A subset of `model_outputs` with uniform distribution
                across the defined bins.

        Example:
            ```
            input_subset, output_subset = select_uniform(
                model_inputs, model_outputs, dim="2D", verbose=True)
            ```

        Notes:
            - The function divides the output range into `num_bins` bins
            (default is 10), and uniformly samples `sample_size` points from
            each bin.
            - The sample size is determined by the dimensionality (`dim`),
            with 2D data having a smaller sample size.
            - The output subset is plotted if `verbose` is True.
        """
        # number of bins (adjust based on desired granularity)
        lower_bound = 0
        upper_bound = 2

        # Create bins and sample uniformly from each bin
        bins = np.linspace(lower_bound, upper_bound, num_bins + 1)
        if verbose:
            print("bins for uniformity in y", bins)
        n_bin_values = []

        # First go through and calculate how many are in each bin
        for i in range(num_bins):
            # Select values in the current bin
            bin_indices = np.where(
                (model_outputs >= bins[i]) & (model_outputs < bins[i + 1])
            )[0]
            n_bin_values.append(len(bin_indices))

        if verbose:
            print("starting n_bin_values", n_bin_values)

        # Setting a random seed
        np.random.seed(rs)
        selected_indices = []

        # sample size is the number of samples to pull
        # from each bin in order to achieve the desired
        # uniform distribution across num_bins
        # that add to the total desired size_df
        sample_size = int(size_df / num_bins)
        for i in range(num_bins):
            # Get indices in the current bin
            bin_indices = np.where(
                (model_outputs >= bins[i]) & (model_outputs < bins[i + 1])
            )[0]
            # Take and randomly sample from each bin
            sampled_indices = np.random.choice(
                bin_indices, sample_size, replace=False
            )
            selected_indices.extend(sampled_indices)
        selected_indices = np.array(selected_indices)
        input_subset = model_inputs[selected_indices]
        output_subset = np.array(model_outputs)[selected_indices]

        if verbose:
            plt.hist(output_subset.flatten())
            plt.xlabel("output variable")
            plt.show()
            print("shape before cut", np.shape(model_outputs))
            print(
                "shape once uniform",
                np.shape(output_subset),
            )

        return input_subset, output_subset

    def image_gen(
        self,
        image_size=100,
        amplitude=10,
        radius=10,
        center_x=50,
        center_y=50,
        theta=0,
        noise_level=0.0,
        inject_type="output",
        seed=42,
    ):
        """Generates a synthetic 2D image of a galaxy-like object based on
        specified parameters.

        This function creates an image of a galaxy object using parameters
        such as amplitude, radius, center coordinates, and orientation (theta).
        Optionally, noise can be added to the image. The object is generated
        based on a galaxy model defined in the `GalaxyObject` class.

        This function is from the DeepBench software:
        https://github.com/deepskies/DeepBench/tree/main

        Args:
            image_size (int): Size of the image (width and height in pixels).
                Default is 100.
            amplitude (float): Peak brightness of the object. Default is 10.
            radius (float): Radius of the galaxy object. Default is 10.
            center_x (int): X-coordinate of the galaxy center. Default is 50.
            center_y (int): Y-coordinate of the galaxy center. Default is 50.
            theta (float): Rotation angle of the galaxy object in radians.
                Default is 0.
            noise_level (float): Standard deviation of noise to be added to
                the image. Default is 0.0 (no noise).
            inject_type (str): Determines where noise is injected:
                - "output": Noise is added after the object is generated
                (currently unused). Default is "output".
            seed (int): Random seed for reproducibility. Default is 42.

        Returns:
            np.ndarray: 2D array representing the generated image.

        Example:
            To generate an image of a galaxy with specific amplitude and
            radius:
            ```
            image = image_gen(
                image_size=64, amplitude=15, radius=8,
                center_x=32, center_y=32)
            ```

        Notes:
            - The galaxy object is modeled using the `GalaxyObject` class,
            which simulates an elliptical galaxy.
            - Noise can be added to the image based on the `noise_level`
            argument.
        """
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
        inject_type="output",
        rs=40,
    ):
        """Simulates 2D image data based on provided parameters and noise
        levels.

        This function generates synthetic 2D image data by simulating images
        with specified parameters such as amplitude, radius, and rotation.
        Noise can be injected either in the output (the total brightness) or
        directly into the image pixels. The resulting images and brightness
        values are returned.

        Args:
            size_df (int): Number of images to simulate.
            params (np.ndarray): Array of shape (n_samples, 3) containing
                parameters for image generation:
                - Amplitude (param[:, 0])
                - Radius (param[:, 1])
                - Theta (param[:, 2])
            sigma (float): Standard deviation of the noise to be injected.
            image_size (int): Size of the images to generate
                (image_size x image_size). Default is 32.
            inject_type (str): Determines where the noise is injected:
                - "output": Noise is added to the total brightness.
                - "input": Noise is added to the image pixels directly.
                Default is "output".
            rs (int): Random seed for reproducibility. Default is 40.

        Returns:
            tuple:
            - image_array (np.ndarray): Array of shape
                (size_df, image_size, image_size) containing the generated
                images (with or without noise).
            - total_brightness (list): List of total brightness values for
                each image, either with noise added ("output" mode) or without
                ("input" mode).

        Example:
            After calling this function, the generated images and brightness
            values can be used for further analysis or training:
            ```
            image_array, total_brightness = simulate_data_2d(
                size_df=100, params=params, sigma=0.1)
            ```

        Notes:
            - The `image_gen` method is used to generate individual images
                based on the amplitude, radius, and theta values.
            - Noise is added to the total brightness when
                `inject_type="output"`.
            - Noise is added directly to the image pixels when
                `inject_type="input"`.
        """
        # set the random seed
        np.random.seed(rs)
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
            if inject_type == "output":
                image_array[i, :, :] = image
                total_brightness.append(
                    np.sum(image) + np.random.normal(loc=0, scale=sigma)
                )
            elif inject_type == "input":
                noisy_image = image + np.random.normal(
                    loc=0,
                    scale=sigma,
                    size=(image_size, image_size),
                )
                image_array[i, :, :] = noisy_image
                total_brightness.append(np.sum(image))
            # we'll need the noisy image summed if we want to
            # do a comparison of y - y':
            # total_brightness_prop_noisy.append(np.sum(noisy_image))
        self.input = image_array
        self.output = total_brightness
        print(
            f"2D data generated, \
                with noise injected type: {inject_type}."
        )
        return image_array, total_brightness

    def simulate_data(
        self,
        thetas,
        sigma,
        x=np.linspace(0, 10, 100),
        inject_type="output",
        seed=42,
        vary_sigma=False,
        verbose=False,
    ):
        """Simulates linear data based on provided parameters and noise levels.

        This function generates synthetic data by simulating linear
        relationships with noise injected into either the inputs or outputs.
        The parameters are sampled from a prior distribution, and random
        Gaussian noise is added to simulate uncertainty. The simulated data is
        stored in the `input`, `output`, and `output_err` attributes.

        Args:
            thetas (np.ndarray): Array of shape (n_samples, 2) containing the
                slope (m) and intercept (b) parameters for each simulation.
            sigma (float): Standard deviation of the noise to be injected.
                Can be varied per simulation if `vary_sigma` is set to True.
            x (np.ndarray): Array of x-values for generating the data. Default
                is 100 values between 0 and 10.
            inject_type (str): Indicates whether noise is injected into
                the "input" or "output". Default is "output".
            seed (int): Random seed for reproducibility. Default is 42.
            vary_sigma (bool): If True, varies the noise level based on the
                slope (m) of each simulation. Default is False.
            verbose (bool): If True, displays a histogram of the noise scale
                and other visualizations for debugging. Default is False.

        Raises:
            ValueError: If `thetas` does not have shape (n, 2).

        Returns:
            None: Simulated data is stored in the `input`, `output`, and
            `output_err` attributes of the object.

        Example:
            After calling this function, the simulated input and output data
            can be accessed via the `input`, `output`, and `output_err`
            attributes:
            ```
            self.input   -> Tensor of simulated input values
            self.output  -> Tensor of simulated output values
            self.output_err -> Tensor of noise values applied to the output
            ```
        """
        # convert to numpy array (if tensor):
        thetas = np.atleast_2d(thetas)
        n_sim = thetas.shape[0]
        print("number of sims", n_sim)
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
        if vary_sigma:
            print("YES WERE VARYING SIGMA")
            new_sig = self.get_sigma_m(sigma, m)
            ε = rs.normal(loc=0, scale=new_sig, size=(len(x), n_sim))
            scale = new_sig
        else:
            print("NO WERE NOT VARYING SIGMA")
            ε = rs.normal(loc=0, scale=sigma, size=(len(x), n_sim))
            scale = sigma
        if verbose:
            plt.clf()
            plt.hist(scale)
            plt.annotate(
                "mean = " + str(np.mean(scale)),
                xy=(0.02, 0.9),
                xycoords="axes fraction",
            )
            plt.title("scale param, injection " + str(inject_type))
            plt.show()
        # Initialize an empty array to store the results
        # for each set of parameters
        x_noisy = np.zeros((len(x), thetas.shape[0]))
        y_noisy = np.zeros((len(x), thetas.shape[0]))
        y = np.zeros((len(x), thetas.shape[0]))
        for i in range(thetas.shape[0]):
            m, b = thetas[i, 0], thetas[i, 1]
            if inject_type == "output":
                y_noisy[:, i] = m * x + b + ε[:, i]
                y[:, i] = m * x + b
            elif inject_type == "input":
                # y_prime[:, i] = m * (x + ε[:, i]) + b
                y[:, i] = m * x + b
                x_noisy[:, i] = x + ε[:, i]

        if inject_type == "output":
            # self.input = x
            self.input = torch.Tensor(np.tile(x, thetas.shape[0]).T)
            self.output = torch.Tensor(y_noisy.T)
            # self.output_err = ε[:, i].T
        elif inject_type == "input":
            self.input = torch.Tensor(x_noisy.T)
            self.output = torch.Tensor(y.T)
            # self.output_err = ε[:, i].T
        print(
            f"0D data generated, \
                with noise injected type: {inject_type}."
        )
        return

    def sample_params_from_prior(
        self,
        n_samples,
        low=[0.1, 0],
        high=[0.4, 0],
        n_params=2,
        seed=42,
    ):
        """Sample parameters from a uniform prior distribution within
        specified bounds.

        This method generates a set of random parameter values for a model by
        sampling uniformly from a prior distribution defined by `low` and
        `high` bounds for each parameter. The resulting samples are stored in
        the `params` attribute.

        Args:
            n_samples (int): The number of parameter samples to generate.
            low (list of float): The lower bounds for each parameter
                (default is [0.1, 0]).
            high (list of float): The upper bounds for each parameter
                (default is [0.4, 0]).
            n_params (int): The number of parameters to sample (default is 2).
            seed (int): The random seed for reproducibility (default is 42).

        Raises:
            AssertionError: If the size of `low` and `high` does not match
            `n_params`.

        Returns:
            None: The sampled parameters are stored in the `params` attribute
            as a NumPy array of shape `(n_samples, n_params)`.
        """
        assert (
            len(low) == len(high) == n_params
        ), "the length of the bounds must match that of the n_params"
        low_bounds = torch.tensor(low, dtype=torch.float32)
        high_bounds = torch.tensor(high, dtype=torch.float32)
        rs = np.random.RandomState(seed)  # 2147483648)#
        prior = rs.uniform(
            low=low_bounds,
            high=high_bounds,
            size=(n_samples, n_params),
        )
        self.params = prior

    def get_dict(self):
        """Retrieve a dictionary containing key data attributes.

        This method returns a dictionary that includes parameters, input data,
        output data, and output errors from the object. The keys in the
        dictionary are:
        - 'params': Model parameters or other relevant settings.
        - 'input': Input data used for the model or simulation.
        - 'output': Output data generated by the model or simulation.

        Returns:
            dict: A dictionary containing 'params', 'input', and 'output'.
        """
        data_dict = {
            "params": self.params,
            "input": self.input,
            "output": self.output,
        }
        return data_dict

    def get_data(self):
        """Retrieve the main data object.

        This method returns the data stored in the 'data' attribute of the
        object. The data could represent various forms, such as raw input data,
        processed datasets, or simulation outputs.

        Returns:
            any: The data stored in the 'data' attribute.
        """
        return self.data

    def get_sigma_m(self, noise, m):
        """Get the sigma value based on the noise level and slope `m`.

        This function computes the sigma value for uncertainty quantification,
        where sigma is scaled by the absolute value of `m`. The noise level
        can be specified as 'low', 'medium', or 'high', and determines the
        output value of sigma before scaling.

        Args:
            noise (str): The noise level, expected to be one of the following:
                        - 'low': Low noise level, output sigma is 0.01.
                        - 'medium': Medium noise level, output sigma is 0.05.
                        - 'high': High noise level, output sigma is 0.10.
            m (float): A scaling factor that sigma is divided by. The absolute
                value of `m` is used to prevent negative scaling.

        Returns:
            float: The scaled sigma value based on the noise level and `m`.

        Raises:
            ValueError: If the noise level is not 'low', 'medium', or 'high'.
        """
        if noise == "low":
            sigma = 0.01 / abs(m)
        elif noise == "medium":
            sigma = 0.05 / abs(m)
        elif noise == "high":
            sigma = 0.10 / abs(m)
        return sigma

    def get_sigma(self, noise, inject_type="output", data_dimension="0D"):
        """Get the value of sigma (standard deviation) based on noise level
        and injection type.

        This function returns a sigma value that represents the standard
        deviation of noise based on the specified noise level (`low`, `medium`,
        `high`, or `vhigh`), injection type, and data dimension. The sigma is
        used to quantify the level of uncertainty or variability injected into
        the data.

        Args:
            noise (str): The noise level, expected to be one of the following:
                        'low', 'medium', 'high', or 'vhigh'.
            inject_type (str, optional): Specifies where the noise is injected.
                                        Defaults to 'output'. Valid options
                                        are:
                - 'output': Noise is applied to the output.
                - 'input': Noise is applied to the input.
            data_dimension (str, optional): The dimensionality of the data.
                Defaults to "0D". Only relevant when `inject_type` is 'input'.
                Expected to be either '0D' or '2D'.

        Returns:
            float: The sigma value corresponding to the noise level and
                injection type.

        Raises:
            ValueError: If the noise level does not match any of the
                predefined categories.
        """
        if inject_type == "output":
            if noise == "low":
                sigma = 0.01
            elif noise == "medium":
                sigma = 0.05
            elif noise == "high":
                sigma = 0.10
            elif noise == "vhigh":
                sigma = 1.00
            else:
                print(
                    "cannot find a match for this noise",
                    noise,
                )
        elif inject_type == "input" and data_dimension == "2D":
            if noise == "low":
                sigma = 0.01 / 32
            elif noise == "medium":
                sigma = 0.05 / 32
            elif noise == "high":
                sigma = 0.10 / 32
        return sigma

    def normalize(self, inputs, ys_array, norm=False):
        """Normalize input and output arrays, with optional normalization
        based on min-max scaling.

        This function optionally normalizes the input (`inputs`) and output
        (`ys_array`) data using min-max scaling, which rescales the data to
        the range [0, 1]. If normalization is applied (`norm=True`), the
        function also returns the normalization parameters
        (min and max values for both inputs and outputs).

        Args:
            inputs (array-like): The input data to be normalized, typically
                features.
            ys_array (array-like): The output data to be normalized, typically
                labels.
            norm (bool, optional): If True, normalization is applied. If False,
                the data is returned unchanged. Default is False.

        Returns:
            tuple: A tuple containing three elements:
                - model_inputs (array-like): The (normalized) input data.
                - model_outputs (array-like): The (normalized) output data.
                - normalization_params (dict or None): A dictionary with the
                    min and max values used for normalization if `norm=True`,
                    otherwise None.
        """
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
        return (
            model_inputs,
            model_outputs,
            normalization_params,
        )

    def train_val_split(
        self,
        model_inputs,
        model_outputs,
        val_proportion=0.1,
        random_state=42,
    ):
        """Split model inputs and outputs into training and validation sets.

        This function takes input data (`model_inputs`) and corresponding
        output data (`model_outputs`) and splits them into training and
        validation sets based on the specified validation proportion. It uses
        the `train_test_split` method from scikit-learn to perform the split
        and ensures reproducibility through a random state.

        Args:
            model_inputs (array-like): The input data for the model, typically
                features.
            model_outputs (array-like): The corresponding output data for the
                model, typically labels.
            val_proportion (float, optional): The proportion of data to be
                used for validation. Default is 0.1 (10%).
            random_state (int, optional): Seed used by the random number
                generator for reproducibility. Default is 42.

        Returns:
            tuple: A tuple containing four elements:
                - x_train: The training set of inputs.
                - x_val: The validation set of inputs.
                - y_train: The training set of outputs.
                - y_val: The validation set of outputs.
        """
        x_train, x_val, y_train, y_val = train_test_split(
            model_inputs,
            model_outputs,
            test_size=val_proportion,
            random_state=random_state,
        )
        return x_train, x_val, y_train, y_val
