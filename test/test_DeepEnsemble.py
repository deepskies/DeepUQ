import sys
import pytest
import os
import subprocess
import tempfile
import shutil
from scripts import evaluate, models, DeepEnsemble


@pytest.fixture
def temp_directory():
    # Setup: Create a temporary directory with one folder level
    temp_dir = tempfile.mkdtemp()

    # Create subdirectories within the temporary directory
    models_dir = os.path.join(temp_dir, "models")
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

def test_chkpt_saved(temp_directory):
    noise_level = "low"
    n_models = 10
    wd = str(temp_directory) + "/"
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        noise_level,
        str(n_models),
        wd,
        "--n_epochs",
        str(n_epochs),
        "--save_final_checkpoint",
        "--savefig"
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
    ), "Expected 10 files in the 'models' folder"

    # check if the right number of images were saved
    animations_folder = os.path.join(temp_directory, "images/animations")
    files_in_animations_folder = os.listdir(animations_folder)
    # assert that the number of files is equal to 10
    assert (
        len(files_in_animations_folder) == n_models
    ), "Expected 10 files in the 'images/animations' folder"

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
def test_no_chkpt_saved_xfail(temp_directory):
    noise_level = "low"
    n_models = 10
    wd = str(temp_directory) + "/"
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        noise_level,
        str(n_models),
        wd,
        "--n_epochs",
        str(n_epochs),
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
    ), "Expected 10 files in the 'models' folder"


def test_no_chkpt_saved(temp_directory):
    noise_level = "low"
    n_models = 10
    wd = str(temp_directory) + "/"
    n_epochs = 2
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        noise_level,
        str(n_models),
        wd,
        "--n_epochs",
        str(n_epochs),
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
    # check if the right number of checkpoints are saved
    models_folder = os.path.join(temp_directory, "models")
    # list all files in the "models" folder
    files_in_models_folder = os.listdir(models_folder)
    # assert that the number of files is equal to 10
    assert len(files_in_models_folder) == 0, "Expect 0 files in the 'models' folder"



def test_run_simple_ensemble(temp_directory):
    noise_level = "low"
    n_models = "10"
    # here = os.getcwd()
    # wd = self.temp_path
    # os.path.dirname(here) + str(temp_directory) + '/'
    wd = str(temp_directory) + "/"
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        noise_level,
        n_models,
        wd,
        "--n_epochs",
        "2",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)


@pytest.mark.xfail(strict=True)
def test_missing_req_arg(temp_directory):
    noise_level = "low"
    n_models = "10"
    subprocess_args = [
        "python",
        "src/scripts/DeepEnsemble.py",
        noise_level,
        n_models,
        "--n_epochs",
        "2",
    ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
