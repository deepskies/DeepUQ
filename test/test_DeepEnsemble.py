import pytest
import os
import subprocess
import tempfile
import shutil
import yaml
from scripts.io import DataLoader, DataPreparation


@pytest.fixture
def temp_data():
    # setup: Create a temporary directory with one folder level
    temp_dir = tempfile.mkdtemp()

    # create subdirectories within the temporary directory
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir)

    # now create
    data = DataPreparation()
    size_df = 10
    noise = 'low'
    data.sample_params_from_prior(size_df) 
    if noise == 'low':
    sigma = 1
    if noise == 'medium':
        sigma = 5
    if noise == 'high':
        sigma = 10
    if noise == 'vhigh':
        sigma = 100
    data.simulate_data(data.params,
                    sigma, 
                    'linear_homogeneous'
                        )  
    dict = data.get_dict()
    saver = DataLoader()  
    # save the dataframe
    filename = data_dir + 'linear_sigma_'+str(sigma)+'_size_'+str(size_df)
    saver.save_data_h5(filename, dict) 

    yield filename  # provide the temporary directory path to the test function

    # teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_directory():
    # setup: Create a temporary directory with one folder level
    temp_dir = tempfile.mkdtemp()

    # create subdirectories within the temporary directory
    yaml_dir = os.path.join(temp_dir, "yamls")
    os.makedirs(yaml_dir)

    models_dir = os.path.join(temp_dir, "checkpoints")
    os.makedirs(models_dir)

    animations_dir = os.path.join(temp_dir, "images", "animations")
    os.makedirs(animations_dir)

    yield temp_dir  # provide the temporary directory path to the test function

    # teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)


def create_test_config(temp_directory, n_epochs):
    print("dumping temp yaml")
    print("temp_dir", temp_directory)
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
    print("theoretically dumping here", str(temp_directory) + "yamls/DE.yaml")
    yaml.dump(input_yaml, open(str(temp_directory) + "yamls/DE.yaml", "w"))


def test_DE_from_config(temp_directory, temp_data):
    # create the test config dynamically
    # make the temporary config file
    n_epochs = 2
    n_models = 2
    create_test_config(temp_directory + "/", n_epochs)
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        "--config",
        str(temp_directory) + "/yamls/DE.yaml",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
    # check if the right number of checkpoints are saved
    models_folder = os.path.join(temp_directory, "checkpoints")
    print("this is the checkpoints folder", models_folder)
    # list all files in the "models" folder
    files_in_models_folder = os.listdir(models_folder)
    print("files in checkpoints folder", files_in_models_folder)
    # assert that the number of files is equal to 10
    assert (
        len(files_in_models_folder) == n_models
    ), f"Expected {n_models} file in the 'checkpoints' folder"

    # check if the right number of images were saved
    animations_folder = os.path.join(temp_directory, "images/animations")
    files_in_animations_folder = os.listdir(animations_folder)
    assert (
        len(files_in_animations_folder) == n_models
    ), f"Expected {n_models} file in the 'images/animations' folder"

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


def test_DE_chkpt_saved(temp_directory, temp_data):
    noise_level = "low"
    n_models = 2
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        "--noise_level",
        noise_level,
        "--n_models",
        str(n_models),
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
    assert (
        len(files_in_models_folder) == n_models
    ), f"Expected {n_models} files in the 'checkpoints' folder"

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


@pytest.mark.xfail(strict=True)
def test_DE_no_chkpt_saved_xfail(temp_directory, temp_data):
    noise_level = "low"
    n_models = 2
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        "--noise_level",
        noise_level,
        "--n_models",
        str(n_models),
        "--out_dir",
        str(temp_directory) + "/",
        "--n_epochs",
        str(n_epochs),
        "--generatedata",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
    # check if the right number of checkpoints are saved
    models_folder = os.path.join(temp_directory, "models")
    # list all files in the "models" folder
    files_in_models_folder = os.listdir(models_folder)
    # assert that the number of files is equal to 10
    assert (
        len(files_in_models_folder) == n_models
    ), f"Expected {n_models} files in the 'models' folder"


def test_DE_no_chkpt_saved(temp_directory, temp_data):
    noise_level = "low"
    n_models = 2
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        "--noise_level",
        noise_level,
        "--n_models",
        str(n_models),
        "--out_dir",
        str(temp_directory) + "/",
        "--n_epochs",
        str(n_epochs),
        "--generatedata",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
    # check if the right number of checkpoints are saved
    models_folder = os.path.join(temp_directory, "checkpoints")
    # list all files in the "models" folder
    files_in_models_folder = os.listdir(models_folder)
    # assert that the number of files is equal to 10
    assert len(files_in_models_folder) == 0, \
        "Expect 0 files in the 'models' folder"


def test_DE_run_simple_ensemble(temp_directory, temp_data):
    noise_level = "low"
    n_models = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        "--noise_level",
        noise_level,
        "--n_models",
        str(n_models),
        "--out_dir",
        str(temp_directory) + "/",
        "--n_epochs",
        "2",
        "--generatedata",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
