import sys
import pytest
import torch
import numpy as np
import sbi
import os
import subprocess
import tempfile
import shutil
import unittest

# flake8: noqa
sys.path.append("..")
#print(sys.path)
#from scripts.evaluate import Diagnose_static, Diagnose_generative
#from scripts.io import ModelLoader
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
    '''
    for dir_path in [models_dir, animations_dir, temp_dir]:
        os.rmdir(dir_path)
        # Teardown: Remove the temporary directory and its contents
    '''
    shutil.rmtree(temp_dir)


'''
@pytest.fixture
def temp_directory(tmpdir):
    # Setup: Create a temporary directory
    #temp_dir = tmpdir.mkdir("temp_test_directory")
    
    #temp_dir = tmpdir.join("temp_test_directory")
    #os.mkdir(temp_dir + '/models/')
    #os.mkdir(temp_dir + '/images/animations/')
    temp_dir = tmpdir / "temp_test_directory"
    temp_dir.mkdir()
    
    yield temp_dir  # Provide the temporary directory to the test function
    # Teardown: Remove the temporary directory and its contents
    temp_dir.remove(rec=True)
'''

'''
class TestMoveEmbargoArgs(unittest.TestCase):
    def setUp(self):
        """
        Performs the setup necessary to run
        all tests
        """
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, "temp_test/")
        self.temp_dir = temp_dir
        self.temp_path = temp_path
        

    def tearDown(self):
        """
        Removes all test files created by tests
        """
        shutil.rmtree(self.temp_dir.name, ignore_errors=True)

'''

def test_run_simple_ensemble(temp_directory):
    noise_level = 'low'
    n_models = '10'
    #here = os.getcwd()
    #wd = self.temp_path
    #os.path.dirname(here) + str(temp_directory) + '/'
    wd = str(temp_directory) + '/'
    print('wd', wd)

    subprocess_args = [
            "python",
            "../src/scripts/DeepEnsemble.py",
            noise_level,
            n_models,
            wd,
            "--n_epochs",
            '2']
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
'''
@pytest.mark.xfail(strict=True)
def test_missing_req_arg():
    noise_level = 'low'
    n_models = 10
    subprocess_args = [
            "python",
            "../src/scripts/DeepEnsemble.py",
            noise_level,
            n_models,
            "--n_epochs",
            '1']
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
'''

'''
def run_ensemble(noise_level,
                 n_models,
                 wd):
    subprocess_args = [
            "python",
            "../src/scripts/DeepEnsemble.py",
            noise_level,
            n_models,
            wd,

            temp_to,
            "LATISS",
            "--embargohours",
            str(embargo_hours),
            "--datasettype",
            *iterable_datasettype,
            "--collections",
            *iterable_collections,
            "--nowtime",
            now_time_embargo,
            "--log",
            log,
            "--desturiprefix",
            desturiprefix,
        ]
    # now run the subprocess
    subprocess.run(subprocess_args, check=True)
'''

"""
@pytest.fixture
def diagnose_static_instance():
    return Diagnose_static()

@pytest.fixture
def diagnose_generative_instance():
    return Diagnose_generative()


@pytest.fixture
def posterior_generative_sbi_model():
    # create a temporary directory for the saved model
    #dir = "savedmodels/sbi/"
    #os.makedirs(dir)

    # now save the model
    low_bounds = torch.tensor([0, -10])
    high_bounds = torch.tensor([10, 10])

    prior = sbi.utils.BoxUniform(low = low_bounds, high = high_bounds)

    posterior = sbi.inference.base.infer(simulator, prior, "SNPE", num_simulations=10000)

    # Provide the posterior to the tests
    yield prior, posterior

    # Teardown: Remove the temporary directory and its contents
    #shutil.rmtree(dataset_dir)

@pytest.fixture
def setup_plot_dir():
    # create a temporary directory for the saved model
    dir = "tests/plots/"
    os.makedirs(dir)
    yield dir

def simulator(thetas):  # , percent_errors):
    # convert to numpy array (if tensor):
    thetas = np.atleast_2d(thetas)
    # Check if the input has the correct shape
    if thetas.shape[1] != 2:
        raise ValueError(
            "Input tensor must have shape (n, 2) \
            where n is the number of parameter sets."
        )

    # Unpack the parameters
    if thetas.shape[0] == 1:
        # If there's only one set of parameters, extract them directly
        m, b = thetas[0, 0], thetas[0, 1]
    else:
        # If there are multiple sets of parameters, extract them for each row
        m, b = thetas[:, 0], thetas[:, 1]
    x = np.linspace(0, 100, 101)
    rs = np.random.RandomState()  # 2147483648)#
    # I'm thinking sigma could actually be a function of x
    # if we want to get fancy down the road
    # Generate random noise (epsilon) based
    # on a normal distribution with mean 0 and standard deviation sigma
    sigma = 5
    ε = rs.normal(loc=0, scale=sigma, size=(len(x), thetas.shape[0]))

    # Initialize an empty array to store the results for each set of parameters
    y = np.zeros((len(x), thetas.shape[0]))
    for i in range(thetas.shape[0]):
        m, b = thetas[i, 0], thetas[i, 1]
        y[:, i] = m * x + b + ε[:, i]
    return torch.Tensor(y.T)


def test_generate_sbc_samples(diagnose_generative_instance,
                              posterior_generative_sbi_model):
    # Mock data
    #low_bounds = torch.tensor([0, -10])
    #high_bounds = torch.tensor([10, 10])

    #prior = sbi.utils.BoxUniform(low=low_bounds, high=high_bounds)
    prior, posterior = posterior_generative_sbi_model
    #inference_instance  # provide a mock posterior object
    simulator_test = simulator  # provide a mock simulator function
    num_sbc_runs = 1000
    num_posterior_samples = 1000

    # Generate SBC samples
    thetas, ys, ranks, dap_samples = diagnose_generative_instance.generate_sbc_samples(
        prior, posterior, simulator_test, num_sbc_runs, num_posterior_samples
    )

    # Add assertions based on the expected behavior of the method
"""
