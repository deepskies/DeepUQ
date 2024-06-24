import pytest
import os
import subprocess
import tempfile
import shutil
import yaml
from data.data import MyDataLoader, DataPreparation


@pytest.fixture()
def temp_data():
    # setup: Create a temporary directory with one folder level
    temp_dir = tempfile.mkdtemp()

    # create subdirectories within the temporary directory
    data_dir = os.path.join(temp_dir, "data")
    os.makedirs(data_dir)

    size_df = 10
    noise_level_list = ["low", "medium", "high"]
    for noise_level in noise_level_list:
        # now create
        data = DataPreparation()
        data.sample_params_from_prior(size_df)
        if noise_level == "low":
            sigma = 1
        if noise_level == "medium":
            sigma = 5
        if noise_level == "high":
            sigma = 10
        if noise_level == "vhigh":
            sigma = 100
        data.simulate_data(data.params, sigma, "linear_homoskedastic")
        dict = data.get_dict()
        saver = MyDataLoader()
        # save the dataframe
        filename = "linear_homoskedastic_sigma_" + str(sigma) + "_size_" + str(size_df)
        saver.save_data_h5(filename, dict, path=data_dir)

    yield data_dir  # provide the temporary directory path to the test function

    # teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_directory():
    # setup: Create a temporary directory with one folder level
    temp_dir_root = tempfile.mkdtemp()

    # create subdirectories within the temporary directory
    yaml_dir = os.path.join(temp_dir_root, "yamls")
    os.makedirs(yaml_dir)

    models_dir = os.path.join(temp_dir_root, "checkpoints")
    os.makedirs(models_dir)

    analysis_dir = os.path.join(temp_dir_root, "analysis")
    os.makedirs(analysis_dir)
    yield temp_dir_root
    # provide the temporary directory path to the test function

    # teardown: Remove the temporary directory and its contents
    shutil.rmtree(temp_dir_root)


def create_test_config_aleatoric(
    temp_directory, n_models, n_epochs,
    noise_level_list=["low", "medium", "high"],
    model_names_list=["DER", "DE"],
):
    input_yaml = {
        "common": {"dir": str(temp_directory)},
        "analysis": {
            "noise_level_list": noise_level_list,
            "model_names_list": model_names_list,
            "plot": False,
            "savefig": True,
            "verbose": False,
        },
        "model": {
            "n_models": n_models,
            "n_epochs": n_epochs,
            "data_prescription": "linear_homoskedastic",
            "BETA": 0.5,
            "COEFF": 0.01,
            "loss_type": "DER",
        },
        "plots": {"color_list": ["#8EA8C3", "#406E8E", "#23395B"]},
    }
    print("theoretically dumping here",
          str(temp_directory) + "yamls/Aleatoric.yaml")
    yaml.dump(input_yaml,
              open(str(temp_directory) + "yamls/Aleatoric.yaml",
                   "w"))


def create_test_config_DE(
    temp_directory, temp_data, n_epochs, noise_level="low", size_df=10
):
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
            "save_all_checkpoints": True,
            "save_final_checkpoint": True,
            "overwrite_final_checkpoint": True,
            "plot": False,
            "savefig": False,
            "verbose": False,
        },
        "data": {
            "data_path": temp_data,
            "data_engine": "DataLoader",
            "data_prescription": "linear_homoskedastic",
            "size_df": size_df,
            "noise_level": noise_level,
            "val_proportion": 0.1,
            "randomseed": 42,
            "batchsize": 100,
        },
        "analysis": {"run_analysis": False},
    }
    print("theoretically dumping here", str(temp_directory) + "yamls/DE.yaml")
    yaml.dump(input_yaml, open(str(temp_directory) + "yamls/DE.yaml", "w"))


def create_test_config_DER(
    temp_directory, temp_data, n_epochs, noise_level="low", size_df=10
):
    input_yaml = {
        "common": {"out_dir": str(temp_directory)},  # +"results/"},
        "model": {
            "model_engine": "DER",
            "model_type": "DER",
            "loss_type": "DER",
            "init_lr": 0.001,
            "COEFF": 0.01,
            "n_epochs": n_epochs,
            "save_all_checkpoints": True,
            "save_final_checkpoint": True,
            "overwrite_final_checkpoint": True,
            "plot": False,
            "savefig": False,
            "save_chk_random_seed_init": False,
            "rs": 42,
            "save_n_hidden": False,
            "n_hidden": 64,
            "verbose": False,
        },
        "data": {
            "data_path": temp_data,
            "data_engine": "DataLoader",
            "data_prescription": "linear_homoskedastic",
            "size_df": size_df,
            "noise_level": noise_level,
            "val_proportion": 0.1,
            "randomseed": 42,
            "batchsize": 100,
        },
        "analysis": {"run_analysis": False},
    }
    print("theoretically dumping here", str(temp_directory) + "yamls/DER.yaml")
    print("this is the yaml", input_yaml)
    yaml.dump(input_yaml, open(str(temp_directory) + "yamls/DER.yaml", "w"))


class TestAleatoric:
    def test_aleatoric_from_config(
        self, temp_directory, temp_data,
    ):
        # first you'll run both the DE and DER
        n_epochs = 2
        n_models = 2
        # do this three times for the different noise levels
        noise_level_list = ["low", "medium", "high"]
        for noise in noise_level_list:
            create_test_config_DE(temp_directory + "/", temp_data, n_epochs,
                                  noise_level=noise)
            subprocess_args = [
                "python",
                "src/scripts/DeepEnsemble.py",
                "--config",
                str(temp_directory) + "/yamls/DE.yaml",
            ]
            # now run the subprocess
            subprocess.run(subprocess_args, check=True)
            create_test_config_DER(temp_directory + "/", temp_data, n_epochs,
                                   noise_level=noise)
            subprocess_args = [
                "python",
                "src/scripts/DeepEvidentialRegression.py",
                "--config",
                str(temp_directory) + "/yamls/DER.yaml",
            ]
            # now run the DER subprocess
            subprocess.run(subprocess_args, check=True)

        # list out what is saved
        models_folder = os.path.join(temp_directory, "checkpoints")
        print("this is the checkpoints folder", models_folder)
        # list all files in the "models" folder
        files_in_models_folder = os.listdir(models_folder)
        print("files in checkpoints folder", files_in_models_folder)
        # now we run the analysis
        create_test_config_aleatoric(temp_directory + "/",
                                     n_models,
                                     n_epochs)
        subprocess_args = [
            "python",
            "src/scripts/Aleatoric.py",
            "--config",
            str(temp_directory) + "/yamls/Aleatoric.yaml",
        ]
        # now run the subprocess
        subprocess.run(subprocess_args, check=True)
        # check if the right number of checkpoints are saved
        analysis_folder = os.path.join(temp_directory, "analysis")
        print("this is the analysis folder", analysis_folder)
        # list all files in the "models" folder
        files_in_analysis_folder = os.listdir(analysis_folder)
        print("files in analysis folder", files_in_analysis_folder)
        # assert that the number of files is equal to 10
        assert (
            len(files_in_analysis_folder) == 1
        ), "Expected 1 file in the 'analysis' folder"
        # now change the number of models
        n_models = 1
        create_test_config_aleatoric(temp_directory + "/",
                                     n_models,
                                     n_epochs)
        subprocess_args = [
            "python",
            "src/scripts/Aleatoric.py",
            "--config",
            str(temp_directory) + "/yamls/Aleatoric.yaml",
            "--n_models",
            str(n_models)
        ]
        # now run the subprocess
        subprocess.run(subprocess_args, check=True)
        # check if the right number of checkpoints are saved
        analysis_folder = os.path.join(temp_directory, "analysis")
        print("this is the analysis folder", analysis_folder)
        # list all files in the "models" folder
        files_in_analysis_folder = os.listdir(analysis_folder)
        print("files in analysis folder", files_in_analysis_folder)
        # assert that the number of files is equal to 10
        assert (
            len(files_in_analysis_folder) == 2
        ), "Expected 2 file in the 'analysis' folder"
