"""
Test the io module
"""
import pytest
import unittest
import numpy as np
import os
import shutil
from scripts.io import ModelLoader, DataLoader, DataPreparation

# first define some fixtures
# fixtures are useful for cases where you need to reuse something
# multiple times in testing


@pytest.fixture
def setup_dir():
    # create a temporary directory for the saved model
    dir = "data/"
    os.makedirs(dir)
    yield dir

#@pytest.fixture
#def teardown(path="data/"):
#    shutil.rmtree(path)


def test_modelloader():
    model = ModelLoader()
    assert 0 == 0


def test_dataloader():
    data = DataLoader()
    assert 0 != 1


def test_datapreparation():
    data = DataPreparation()
    size_df = 1000
    noise = 'vhigh'
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
    print('shape output', np.shape(data.output[0]))
    data.output
    print('shape params', np.shape(data.params))
    assert np.shape(data.input) != np.shape(data.output), \
        f"shape of input {np.shape(data.input)} does not match \
              shape of output {np.shape(data.output)}"


def test_datapreparation_and_saver(setup_dir):
    data = DataPreparation()
    size_df = 1000
    noise = 'vhigh'
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
    print('shape output', np.shape(data.output[0]))
    data.output
    print('shape params', np.shape(data.params))
    assert np.shape(data.input) != np.shape(data.output), \
        f"shape of input {np.shape(data.input)} does not match \
              shape of output {np.shape(data.output)}"
    datadict = data.get_dict()
    assert isinstance(datadict, dict)
    saver = DataLoader()
    # save the dataframe
    name = 'linear_sigma_'+str(sigma) + '_size_'+str(size_df)
    saver.save_data_h5(name, datadict, path="data/")
    assert os.path.exists(name+".h5"), f"File {name} does not exist"
    #shutil.rmtree("data/")
    #teardown(path="data/")



"""
To run this suite of tests, run 'pytest' in the main directory
"""
if __name__ == "__main__":
    unittest.main()
