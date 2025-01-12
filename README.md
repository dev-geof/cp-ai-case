# cp-ai-case

Code repository designed for solving CP AI case. 

## Software Implementation

The code provide a semi-modular python architecture designed to facilitate the implementation, the training and the evaluation of the ML model used for the CP AI case analyses. The core structure is organized into three main tools and several utilities. The code is is formated with black and more internal documentation is provide through doctstrings. 

### Main Tools and Utilities

The main tools provide essential functionalities for various stages of the ML pipeline. These stages include data preparation, model training, and evaluation. Data preparation tools ensure that raw data is cleaned, formatted, and features are engineered for optimal input into the model. Training tools encompass scripts and utilities for initiating and monitoring the training process, optimizing hyper-parameters. Evaluation tools offer methods for calculating performance metrics, validating models, and interpreting model outputs.

The utilities support the main tools and enhance their functionality. The Model utilities define and manage the architecture and parameters of the ML models, including saving and loading models. Plotting utilities provide tools for visualizing, input features, training progress and model performance. Data and NLP utilities handle efficient loading, batching, and management of datasets. Configuration utilities manage settings and parameters through configuration files, ensuring consistent experiment setups. Finally, Callbacks offer functions for logging progress, early stopping, and checkpointing models during training, improving the training process’s efficiency and reliability.

### Entry Points

The setup configuration specifies entry points for the tree main tool scripts, enabling users to execute data preparation, model training, and evaluation directly from the command line. This facilitates streamlined workflows and simplifies the execution of common tasks. The toolkit leverages YAML configuration files to allow users to pilot and manage all these scripts and tools efficiently. These configuration files store settings for data paths, hyper-parameters, training schedules, model configurations, and other parameters. The configuration system parses these files, ensuring that the entire pipeline - from data preparation to model evaluation adheres to the specified settings.

```
usage: 	preparation [-h] [--configfile CONFIGFILE]
	training [-h] [--configfile CONFIGFILE]
	validation [-h] [--configfile CONFIGFILE]

optional arguments:
		-h, --help 			Show this help message and exit
		--configfile CONFIGFILE 	YAML configuration file path
```

## Installation

Start by cloning the repository from GitHub:
```
git clone git@github.com:dev-geof/cp-ai-case.git
cd cp-ai-case.git
```
It is suggested to install the code within a virtual environment. The python "venv" environment is a lightweight solution to install the necessary dependencies.
```
python3 -m venv env
source env/bin/activate
python -m pip install -e . -r requirements.txt
```


NB: Code run with Python 3.9.13 on Macbook Air M1. 

