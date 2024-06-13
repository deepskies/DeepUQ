# DeepUQ
DeepUQ is a package for injecting and measuring different types of uncertainty in ML models.

![status](https://img.shields.io/badge/arXiv-000.000-red)(arxiv link if applicable)

![status](https://img.shields.io/badge/PyPi-0.0.0.0-blue)(pypi link if applicable)

![status](https://img.shields.io/badge/License-MIT-lightgrey)(MIT or Apache 2.0 or another requires link changed)

![test](https://github.com/deepskies/DeepUQ/.github/workflows/test.yml/badge.svg)

## Installation

### Install via pypi
To be updated, not yet released.

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

## How to reproduce the results of the paper

The config settings for the models used in the paper can be found in `src/utils/defaults.py`.

The user should run the following commands from the cli:
> python src/scripts/DeepEnsemble.py --save_all_checkpoints --noise_level "low"

The noise level argument should be modified to run the medium and high settings as well.

Repeat for the DER:
> python src/scripts/DeepEvidentialRegression.py --save_all_checkpoints --noise_level "low"

Next run the analysis scripts:
> python src/scripts/AleatoricandEpistemic.py

> python src/scripts/LossFunctions.py

> python src/scripts/ParitySigma.py

To reproduce the random initialization runs for the DER (these already exist for the DE):
> python src/scripts/DeepEvidentialRegression.py --save_all_checkpoints --noise_level "low" --save_chk_random_seed_init --rs 10

Change the value of the random seed to match those given in the `src/scripts/Aleatoric_and_inits.py` script.

Finally:
> python src/scripts/Aleatoric_and_inits.py

