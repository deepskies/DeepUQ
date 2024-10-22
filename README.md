# DeepUQ
DeepUQ is a package for injecting and measuring different types of uncertainty in ML models.

[![PyPi](https://img.shields.io/badge/PyPi-0.1.0-blue)](https://pypi.org/project/deepuq/)

[![License](https://img.shields.io/badge/License-MIT-lightgrey)](https://opensource.org/licenses/MIT)

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/deepskies/deepuq/build-repo/main)](https://github.com/deepskies/deepuq/actions/workflows)

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/deepskies/DeepUQ/main)](https://github.com/deepskies/DeepUQ/actions/workflows/test.yaml)

![Codecov](https://codecov.io/gh/deepskies/DeepUQ/main/graph/badge.svg)


## Installation

### Install the deepuq package via venv and pypi
> python3.10 -m venv name_of_your_virtual_env

> source name_of_your_virtual_env/bin/activate

> pip install deepuq

Now you can run some of the scripts!
> UQensemble --generatedata

^`generatedata` is required if you don't have any saved data. You can set other keywords like so.

It's also possible to verify the install works by running:
> pytest

### Preferred dev install option: Poetry
If you'd like to contribute to the package development, please follow these instructions.

First, navigate to where you'd like to put this repo and type:
> git clone https://github.com/deepskies/DeepUQ.git

Then, cd into the repo:
> cd DeepUQ

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

Now you have access to all the dependencies necessary to run the package.

## Package structure
DeepUQ/
├── CHANGELOG.md
├── LICENSE.txt
├── README.md
├── DeepUQResources/
├── data/
├── dist/
├── environment.yml
├── images/
├── models/
├── notebooks/
├── poetry.lock
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── analyze/
│   ├── data/
│   ├── models/
│   ├── scripts/
│   ├── train/
│   └── utils/
├── test/
│   ├── DeepUQResources/
│   ├── data/
│   ├── test_DeepEnsemble.py
│   └── test_DeepEvidentialRegression.py


## How to run the workflow
The scripts can be accessed via the ipython example notebooks or via the model modules (ie `DeepEnsemble.py`). For example, to ingest data and train a Deep Ensemble from the DeepUQ/ directory:

> python src/scripts/DeepEnsemble.py

The equivalent shortcut command:
> UQensemble

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

The shortcut:
> UQder



