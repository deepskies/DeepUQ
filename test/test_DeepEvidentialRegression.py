import pytest
import os
import subprocess
import tempfile
import shutil
import yaml
from src.data.data import MyDataLoader, DataPreparation


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
    data.generate_df(size_df, noise, dim, injection, uniform, verbose)
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
    """Fixture to create a temporary directory structure for testing.

    This fixture creates a root temporary directory and several subdirectories
    used for saving configuration files, model checkpoints, and images.
    The directory structure includes:

    - 'yamls': Directory for saving configuration files in YAML format.
    - 'checkpoints': Directory for saving model checkpoints.
    - 'images/animations': Directory for saving generated images and
      animations.

    The fixture yields the root directory path for use in tests. After the
    test completes, the entire directory and its contents are deleted.

    Yields:
        str: Path to the root of the temporary directory structure.

    Teardown:
        Removes the temporary directory and its subdirectories after the test.
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
    """Generates and saves a YAML configuration file for testing a model.

    This function creates a YAML configuration file tailored for testing the
    Deep Evidential Regression (DER) model. The configuration includes
    settings for the model, data, and analysis components, and it is saved to
    a temporary directory under the 'yamls' folder.

    Args:
        temp_directory (str): Path to the root temporary directory where the
                              YAML file is saved.
        temp_data (str): Path to the temporary data directory containing the
                         dataset.
        n_epochs (int): Number of epochs for model training.
        noise_level (str, optional): Noise level for the dataset.
                                     Default is 'low'.
        size_df (int, optional): Size of the dataset. Default is 10.

    Configuration Structure:
        - **common**: Directory paths and other general settings.
        - **model**: Configuration for the DER model including engine, type,
                     loss, learning rate, and other training options.
        - **data**: Settings for data loading and injection, including the
                    data path, dimensionality, noise level, and batch size.
        - **analysis**: Flags to control whether analysis is performed after
                        training.

    The configuration file is saved to:
    `{temp_directory}/yamls/DER.yaml`.

    Output:
        - YAML configuration file dumped into the 'yamls' directory of the
          provided `temp_directory`.
    """
    input_yaml = {
        "common": {"out_dir": str(temp_directory)},  # +"results/"},
        "model": {
            "model_engine": "DER",
            "model_type": "DER",
            "loss_type": "DER",
            "init_lr": 0.001,
            "COEFF": 0.5,
            "n_epochs": n_epochs,
            "save_all_checkpoints": False,
            "save_final_checkpoint": True,
            "overwrite_final_checkpoint": True,
            "plot": False,
            "savefig": True,
            "save_chk_random_seed_init": False,
            "rs": 42,
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
            "generatedata": True,
        },
        "analysis": {"run_analysis": False},
    }
    print(
        "theoretically dumping here",
        str(temp_directory) + "yamls/DER.yaml",
    )
    print("this is the yaml", input_yaml)
    yaml.dump(input_yaml, open(str(temp_directory) + "yamls/DER.yaml", "w"))


class TestDER:
    """A class containing unit tests for the Deep Evidential Regression (DER)
    model.

    This class includes tests to verify that checkpoints and images are saved
    correctly during training and when using a YAML configuration file.
    """
    def test_DER_all_chkpts_saved(
        self, temp_directory, temp_data, noise_level="low", size_df=100
    ):
        """Test that checkpoints and images are saved after training with DER.

        This test runs the DER model using subprocess and checks if the
        correct number of checkpoint files and image files are saved in the
        respective directories. It verifies that these files contain the
        expected naming convention based on the number of epochs.

        Args:
            temp_directory (str): Path to the temporary directory where output
                                  files are saved.
            temp_data (str): Path to the temporary dataset used for training.
            noise_level (str, optional): Noise level for the dataset.
                                         Default is 'low'.
            size_df (int, optional): Size of the dataset. Default is 10.

        Asserts:
            - One checkpoint file is saved in the 'checkpoints' folder.
            - One image file is saved in the 'images/animations' folder.
            - All saved files contain the substring 'epoch_{n_epochs-1}'.
        """
        noise_level = "low"
        n_epochs = 10
        subprocess_args = [
            "python",
            "src/scripts/DeepEvidentialRegression.py",
            "--data_path",
            str(temp_data),
            "--noise_level",
            noise_level,
            "--size_df",
            str(size_df),
            "--out_dir",
            str(temp_directory) + "/",
            "--n_epochs",
            str(n_epochs),
            "--save_final_checkpoint",
            "--savefig",
            "--generatedata",
            "--save_all_checkpoints",
            "--save_final_checkpoint"
        ]
        # now run the subprocess
        subprocess.run(subprocess_args, check=True)
        # check if the right number of checkpoints are saved
        models_folder = os.path.join(temp_directory, "checkpoints")
        # list all files in the "models" folder
        files_in_models_folder = os.listdir(models_folder)
        # assert that the number of files is equal to 1
        # because were only saving the final checkpoint
        assert (
            len(files_in_models_folder) == n_epochs
        ), f"Expected {n_epochs} files in the 'models' folder and \
             got {len(files_in_models_folder)}"

    def test_DER_one_chkpt_saved(
        self, temp_directory, temp_data, noise_level="low", size_df=100
    ):
        """Test that checkpoints and images are saved after training with DER.

        This test runs the DER model using subprocess and checks if the
        correct number of checkpoint files and image files are saved in the
        respective directories. It verifies that these files contain the
        expected naming convention based on the number of epochs.

        Args:
            temp_directory (str): Path to the temporary directory where output
                                  files are saved.
            temp_data (str): Path to the temporary dataset used for training.
            noise_level (str, optional): Noise level for the dataset.
                                         Default is 'low'.
            size_df (int, optional): Size of the dataset. Default is 10.

        Asserts:
            - One checkpoint file is saved in the 'checkpoints' folder.
            - One image file is saved in the 'images/animations' folder.
            - All saved files contain the substring 'epoch_{n_epochs-1}'.
        """
        noise_level = "low"
        n_epochs = 2
        subprocess_args = [
            "python",
            "src/scripts/DeepEvidentialRegression.py",
            "--data_path",
            str(temp_data),
            "--noise_level",
            noise_level,
            "--size_df",
            str(size_df),
            "--out_dir",
            str(temp_directory) + "/",
            "--n_epochs",
            str(n_epochs),
            "--save_final_checkpoint",
            "--savefig",
            "--generatedata",
            "--save_final_checkpoint"
        ]
        # now run the subprocess
        subprocess.run(subprocess_args, check=True)
        # check if the right number of checkpoints are saved
        models_folder = os.path.join(temp_directory, "checkpoints")
        # list all files in the "models" folder
        files_in_models_folder = os.listdir(models_folder)
        # assert that the number of files is equal to 1
        # because were only saving the final checkpoint
        assert (
            len(files_in_models_folder) == 1
        ), "Expected 1 file in the 'models' folder"

        # check if the right number of images were saved
        animations_folder = os.path.join(temp_directory, "images/animations")
        files_in_animations_folder = os.listdir(animations_folder)
        # assert that the number of files is equal to 1
        # because only saving the final checkpoint
        assert (
            len(files_in_animations_folder) == 1
        ), "Expected 1 file in the 'images/animations' folder"

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

    def test_DER_from_config(
        self, temp_directory, temp_data, noise_level="low", size_df=100
    ):
        """Test training of DER using a YAML configuration file.

        This test dynamically creates a YAML configuration file, runs the DER
        model using the configuration, and checks if the correct number of
        checkpoint and image files are saved in their respective directories.
        It verifies that the files are named correctly based on the number of
        epochs.

        Args:
            temp_directory (str): Path to the temporary directory where output
                                  files are saved.
            temp_data (str): Path to the temporary dataset used for training.
            noise_level (str, optional): Noise level for the dataset.
                                         Default is 'low'.
            size_df (int, optional): Size of the dataset. Default is 10.

        Asserts:
            - One checkpoint file is saved in the 'checkpoints' folder.
            - One image file is saved in the 'images/animations' folder.
            - All saved files contain the substring 'epoch_{n_epochs-1}'.
        """
        # create the test config dynamically
        # make the temporary config file
        n_epochs = 2
        create_test_config(temp_directory + "/", temp_data, n_epochs)
        subprocess_args = [
            "python",
            "src/scripts/DeepEvidentialRegression.py",
            "--config",
            str(temp_directory) + "/yamls/DER.yaml",
        ]
        # now run the subprocess
        subprocess.run(subprocess_args, check=True)
        # check if the right number of checkpoints are saved
        models_folder = os.path.join(temp_directory, "checkpoints")
        # list all files in the "models" folder
        files_in_models_folder = os.listdir(models_folder)
        # assert that the number of files is equal to 1
        assert (
            len(files_in_models_folder) == 1
        ), "Expected 1 file in the 'models' folder"
        # check if the right number of images were saved
        animations_folder = os.path.join(temp_directory, "images/animations")
        files_in_animations_folder = os.listdir(animations_folder)
        # assert that the number of files is equal to 1
        assert (
            len(files_in_animations_folder) == 1
        ), "Expected 1 file in the 'images/animations' folder"
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
