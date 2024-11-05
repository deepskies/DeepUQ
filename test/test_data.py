import pytest
import os
import subprocess
import tempfile
import shutil
import yaml
from deepuq.data.data import MyDataLoader, DataPreparation


@pytest.fixture()
def temp_data():  # noise_level, size_df):
    """Create a temporary directory, generate synthetic data, and save it to
    an HDF5 file for testing purposes.

    This pytest fixture creates a temporary directory with subdirectories for
    storing data. It uses the `DataPreparation` class to sample parameters and
    simulate data with different noise levels. The simulated data is saved as
    an HDF5 file in the temporary directory. After the test runs, the
    directory and its contents are deleted.

    Setup:
        - Create a temporary directory and a 'data' subdirectory.
        - Generate synthetic data using the `DataPreparation` class with
          specified parameters.
        - Simulate data based on a noise level (`low`, `medium`, `high`,
          `vhigh`) which affects the value of `sigma`.
        - Save the generated data as an HDF5 file in the temporary directory.

    Teardown:
        - Delete the temporary directory and all of its contents after the
          test completes.

    Yields:
        str: The path to the temporary directory containing the saved HDF5
             file.

    Example:
        def test_example(temp_data):
            data_dir = temp_data
            # Use the data directory for testing
    """
    # setup: Create a temporary directory with one folder level
    temp_dir = tempfile.mkdtemp()

    # create subdirectories within the temporary directory
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir)

    # now create
    data = DataPreparation()
    noise = "low"
    size_df = 100
    dim = "2D"
    injection = "input"
    uniform = True
    verbose = False
    data.generate_df(size_df, noise, dim, injection, uniform, verbose,
                     rs_uniform=0, rs_simulate_0D=0, rs_simulate_2D=0,
                     rs_prior=0)
    dict = data.get_dict()
    saver = MyDataLoader()
    # save the dataframe
    filename = (
            str(dim)
            + "_"
            + str(injection)
            + "_noise_"
            + noise
            + "_size_"
            + str(size_df)
        )
    saver.save_data_h5(filename, dict, path=data_dir)

    yield data_dir  # provide the temporary directory path to the test function

    # teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_directory():
    """Create a temporary directory with multiple subdirectories for testing
    purposes.

    This pytest fixture sets up a temporary directory structure with
    subdirectories for storing YAML files, model checkpoints, and image
    animations. The fixture yields the path to the root temporary directory,
    which can be used in test functions. After the test completes, the entire
    directory and its contents are deleted to ensure no leftover files remain.

    Setup:
        - Create a temporary root directory.
        - Create subdirectories:
            - 'yamls' for YAML files.
            - 'checkpoints' for model checkpoints.
            - 'images/animations' for animation image files.

    Teardown:
        - Delete the temporary root directory and all of its subdirectories
          and contents after the test completes.

    Yields:
        str: The path to the root temporary directory containing the
             subdirectories.

    Example:
        def test_example(temp_directory):
            temp_dir = temp_directory
            # Use the temp_dir for testing
    """
    # setup: Create a temporary directory with one folder level
    temp_dir_root = tempfile.mkdtemp()

    # create subdirectories within the temporary directory
    yaml_dir = os.path.join(temp_dir_root, "yamls")
    os.makedirs(yaml_dir)

    models_dir = os.path.join(temp_dir_root, "checkpoints")
    os.makedirs(models_dir)

    animations_dir = os.path.join(temp_dir_root, "images", "animations")
    os.makedirs(animations_dir)

    yield temp_dir_root
    # provide the temporary directory path to the test function

    # teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir_root)


def create_test_config(
    temp_directory, temp_data, n_epochs, noise_level="low", size_df=100
):
    """Generate and save a test configuration YAML file for a deep ensemble
    model.

    This function creates a dictionary containing configuration settings for a
    deep ensemble (DE) model, including model parameters, data paths, and
    training settings. The configuration is based on the specified number of
    epochs, data size, and noise level. It saves the configuration as a YAML
    file in a temporary directory provided by the `temp_directory` fixture.

    Args:
        temp_directory (str): Path to the root temporary directory where the
                              YAML file will be saved.
        temp_data (str): Path to the directory containing the generated data
                         for the test.
        n_epochs (int): Number of epochs for model training.
        noise_level (str, optional): Noise level for data generation, default
                                     is "low". Options include "low", "medium",
                                     "high", and "vhigh".
        size_df (int, optional): Size of the dataset used for testing, default
                                 is 100.

    File Output:
        Saves a YAML configuration file named `DE.yaml` in the 'yamls'
        subdirectory of the temporary directory.

    Example:
        create_test_config(temp_directory, temp_data, n_epochs=50,
                           noise_level="medium", size_df=100)

    """
    input_yaml = {
        "common": {"out_dir": str(temp_directory)},
        "model": {
            "model_engine": "DE",
            "model_type": "DE",
            "loss_type": "bnll_loss",
            "init_lr": 0.001,
            "BETA": 0.5,
            "n_models": 2,
            "n_epochs": n_epochs,
            "save_all_checkpoints": False,
            "save_final_checkpoint": True,
            "overwrite_model": True,
            "plot_inline": False,
            "plot_savefig": True,
            "save_chk_random_seed_init": False,
            "rs_list": [41, 42],
            "save_n_hidden": False,
            "n_hidden": 64,
            "save_data_size": False,
            "verbose": False,
        },
        "data": {
            "data_path": temp_data,
            "data_engine": "DataLoader",
            "data_dimension": "2D",
            "data_injection": "input",
            "size_df": size_df,
            "noise_level": noise_level,
            "val_proportion": 0.1,
            "randomseed": 42,
            "batchsize": 100,
        },
        "analysis": {"run_analysis": False},
    }
    print(
        "theoretically dumping here",
        str(temp_directory) + "yamls/DE.yaml",
    )
    yaml.dump(input_yaml, open(str(temp_directory) + "yamls/DE.yaml", "w"))


class TestData:
    """
    Test suite for validating the behavior of the Deep Ensemble (DE) model
    training process, including checkpoint and image generation during
    training.
    """

    def test_DE_from_saved_data(
        self, temp_directory, temp_data, noise_level="low", size_df=100
    ):
        """Test that the correct number of checkpoints and animations are
        saved during DE model training.

        Runs the DE model training as a subprocess and verifies that
        checkpoints and animations corresponding to each epoch are saved in
        the appropriate folders.

        Args:
            temp_directory (str): Path to the temporary directory where
                                  outputs are saved.
            temp_data (str): Path to the generated data for testing.
            noise_level (str, optional): Noise level for data generation,
                                         default is "low".
            size_df (int, optional): Size of the dataset used for testing,
                                     default is 10.
        """
        n_models = 2
        n_epochs = 2
        subprocess_args = [
            "python",
            "deepuq/scripts/DeepEnsemble.py",
            "--data_path",
            str(temp_data),
            "--size_df",
            str(size_df),
            "--noise_level",
            noise_level,
            "--n_models",
            str(n_models),
            "--out_dir",
            str(temp_directory) + "/",
            "--n_epochs",
            str(n_epochs),
            "--save_final_checkpoint",
            "--plot_savefig",
        ]
        # now run the subprocess
        subprocess.run(subprocess_args, check=True)
        # check if the right number of checkpoints are saved
        models_folder = os.path.join(temp_directory, "checkpoints")
        # list all files in the "models" folder
        files_in_models_folder = os.listdir(models_folder)
        # assert that the number of files is equal to 10
        assert (
            len(files_in_models_folder) == n_models
        ), f"Expected {n_models} files in the 'checkpoints' folder"
        f"What is in there {files_in_models_folder}"

        # check if the right number of images were saved
        animations_folder = os.path.join(temp_directory, "images/animations")
        files_in_animations_folder = os.listdir(animations_folder)
        # assert that the number of files is equal to 10
        assert (
            len(files_in_animations_folder) == n_models
        ), f"Expected {n_models} files in the 'images/animations' folder"

        # also check that all files in here have the same name elements
        expected_substring = "epoch_" + str(n_epochs - 1)
        for file_name in files_in_models_folder:
            assert (
                expected_substring in file_name
            ), f"File '{file_name}' does not contain the expected substring"

        # also check that all files in here have the same name elements
        for file_name in files_in_animations_folder:
            assert (
                expected_substring in file_name
            ), f"File '{file_name}' does not contain the expected substring"
