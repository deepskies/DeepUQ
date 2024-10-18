# DeepUQ
DeepUQ is a package for injecting and measuring different types of uncertainty in ML models.

[![PyPi](https://img.shields.io/badge/PyPi-0.1.0-blue)](https://pypi.org/project/deepuq/)

[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://opensource.org/licenses/MIT)

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/deepskies/deepuq/build-repo/main)](https://github.com/deepskies/deepuq/actions/workflows)

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/deepskies/DeepUQ/main)](https://github.com/deepskies/DeepUQ/actions/workflows/test.yaml)


## Installation

### Install via pypi
> pip install deepuq

(Make sure your python version is 3.10)

Verify the install works by also pip installing pytest, cd'ing to the DeepUQ/ directory, and running
> pytest

### Clone this repo
First, cd to where you'd like to put this repo and type:
> git clone https://github.com/deepskies/DeepUQ.git

Then, cd into the repo:
> cd DeepUQ

### Install and use poetry to set up the environment
Poetry is our recommended method of handling a package environment as publishing and building is handled by a toml file that handles all possibly conflicting dependencies. 
Full docs can be found [here](https://python-poetry.org/docs/basic-usage/).

Install instructions: 

Add poetry to your python install 
> pip install poetry

Then, from within the DeepUQ repo, run the following:

Install the pyproject file
> poetry install 

Begin the environment
> poetry shell

### Verify DeepUQ is installed correctly

After following the installation instructions, verify installation is functional is all tests are passing by running the following in the root directory:
> pytest

## How to run the workflow
![Folder structure overview](images/DeepUQWorkflow_Maggie.png)

The scripts can be accessed via the ipython example notebooks or via the model modules (ie `DeepEnsemble.py`). For example, to ingest data and train a Deep Ensemble from the DeepUQ/ directory:

> python src/scripts/DeepEnsemble.py

With no config file specified, this command will pull settings from the `default.py` file within `utils`. For the `DeepEnsemble.py` script, it will automatically select the `DefaultsDE` dictionary.

Another option is to specify your own config file:

> python src/scripts/DeepEnsemble.py --config "path/to/config/myconfig.yaml"

Where you would modify the "path/to/config/myconfig.yaml" to specify where your own yaml lives.

The third option is to input settings on the command line. These choices are then combined with the default settings and output in a temporary yaml.

> python src/scripts/DeepEnsemble.py --noise_level "low" --n_models 10 --out_dir ./DeepUQResources/results/ --save_final_checkpoint True --savefig True --n_epochs 10

This command will train a 10 network, 10 epoch ensemble on the low noise data and will save figures and final checkpoints to the specified directory. Required arguments are the noise setting (low/medium/high), the number of ensembles, and the working directory.

For more information on the arguments:
> python src/scripts/DeepEnsemble.py --help

The other available script is the `DeepEvidentialRegression.py` script:
> python src/scripts/DeepEvidentialRegression.py --help



