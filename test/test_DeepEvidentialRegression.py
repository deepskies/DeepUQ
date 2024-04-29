import pytest
import os
import subprocess
import tempfile
import shutil
import yaml


@pytest.fixture
def temp_directory():
    # Setup: Create a temporary directory with one folder level
    temp_dir = tempfile.mkdtemp()

    # Create subdirectories within the temporary directory
    yaml_dir = os.path.join(temp_dir, "yamls")
    os.makedirs(yaml_dir)

    models_dir = os.path.join(temp_dir, "checkpoints")
    os.makedirs(models_dir)

    animations_dir = os.path.join(temp_dir, "images", "animations")
    os.makedirs(animations_dir)

    yield temp_dir  # Provide the temporary directory path to the test function

    # Teardown: Remove the temporary directory and its contents
    """
    for dir_path in [models_dir, animations_dir, temp_dir]:
        os.rmdir(dir_path)
        # Teardown: Remove the temporary directory and its contents
    """
    shutil.rmtree(temp_dir)


def create_test_config(temp_directory, n_epochs):
    print("dumping temp yaml")
    print("temp_dir", temp_directory)
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
            "verbose": False,
        },
        "data": {
            "data_path": "./data",
            "data_engine": "DataLoader",
            "size_df": 1000,
            "noise_level": "low",
            "val_proportion": 0.1,
            "randomseed": 42,
            "batchsize": 100,
        },
    }
    print("theoretically dumping here", str(temp_directory) + "yamls/DER.yaml")
    yaml.dump(input_yaml, open(str(temp_directory) + "yamls/DER.yaml", "w"))


def test_DER_chkpt_saved(temp_directory):
    noise_level = "low"
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEvidentialRegression.py",
        "--noise_level",
        noise_level,
        "--out_dir",
        str(temp_directory) + "/",
        "--n_epochs",
        str(n_epochs),
        "--save_final_checkpoint",
        "--savefig",
        "--generatedata",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
    # check if the right number of checkpoints are saved
    models_folder = os.path.join(temp_directory, "checkpoints")
    # list all files in the "models" folder
    files_in_models_folder = os.listdir(models_folder)
    # assert that the number of files is equal to 10
    assert len(files_in_models_folder) == 1, \
        "Expected 1 file in the 'models' folder"

    # check if the right number of images were saved
    animations_folder = os.path.join(temp_directory, "images/animations")
    files_in_animations_folder = os.listdir(animations_folder)
    # assert that the number of files is equal to 10
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


def test_DER_from_config(temp_directory):
    # create the test config dynamically
    # make the temporary config file
    n_epochs = 2
    create_test_config(temp_directory + "/", n_epochs)
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
    # assert that the number of files is equal to 10
    assert len(files_in_models_folder) == 1, \
        "Expected 1 file in the 'models' folder"
    # check if the right number of images were saved
    animations_folder = os.path.join(temp_directory, "images/animations")
    files_in_animations_folder = os.listdir(animations_folder)
    # assert that the number of files is equal to 10
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
