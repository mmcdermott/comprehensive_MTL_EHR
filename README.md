# A comprehensive evaluation of Multi-task learning on EHR timeseries
# Install

Set up the repository
```
conda env create --name comprehensive_MTL_EHR -f env.yml
conda activate comprehensive_MTL_EHR
```

# Obtaining Data
To obtain copies of our pre-processed datasets, contact us at `mmd@mit.edu`. Once we verify your CITI & PhysiNet certifications, we'll happily share the pre-processed datasets with you, as well as the precise storage structure you should use to work seamlessly with the code in this release. Dataset code release is forthcoming.

# Usage
## Args in General
Arguments for all scripts are described in the `latent_patient_trajectories/representation_learner/args.py` file. This file has some base classes, then argument classes (with specific args requested) for all functions. It is a good reference to determine what a specific script expects. Note this class allows you to (and we recommend) pre-setting all args for scripts in (appropriately named) json files in the relevant experiment directories, then simply passing the directory to the given script (according to the appropriate `arg`) and adding `--do_load_from_dir`, at which point the script will load all arguments from the json file automatically. Note that some args (specifically `rotation`, which should always be 0, and `notes`, which should always be `no_notes`) are held-out args from older versions of the code, and can be largely ignored.

## Hyperparameter Tuning
To perform hyperparameter tuning, set up a base experimental directory, and add a config file describing your hyperparameter search in that directory. This file must be named according to the `HYP_CONFIG_FILENAME` constant in `latent_patient_trajectories/constants.py` file, which is (as of 7/20/20) set to `hyperparameter_search_config.json`. A sample config file is shown in the file `latent_patient_trajectories/representation_learner/sample_hyperopt_config.json`. 

Then, run the script `Scripts/End to End/hyperopt_model.py` (with appropriate args, as described in the `args.py` file referenced above under the class `HyperparameterSearchArgs`) to kick off your search. 

## Generic Runs
To perform a generic run, training a multi- or single-task model, use the `Args` class in `args.py` and the `run_model.py` script. As with everything else, you will need to specify a base directory, and many other args to describe the architecture you want to use and training details.

### Evaluation
Evaluating a pre-trained run can be accomplished with the `EvalArgs` class and the `evaluate.py` script. You will need to specify the model's training directory (e.g., the directory passed to `run_model.py`) so the script knows what model to reload.

## Task-Generalizability Analyses
These runs consist of pre-training a model on all tasks except for one held-out task, then fine-tuning the model on that held-out task, and evaluating both the pre-trained and fine-tuned models on all tasks. This could be done manually, through repeated use of the `run_model.py` script and the use of the `--ablate` arg, but we have a helper script that can manage doing all requisite runs across multiple GPUs in parallel on a single-machine. To use this, you must first create a base directory for this experiment, which will ultimately hold all runs associated with this experiment (including pre-trained and fine=tuned). In this directory, you will specify the model's base args (which will be duplicated and used in all pre-training and fine-tuning experiments, with the ablate arg automatically adjusted to perform the appropriate experiments) as a `json` file which is parseable by the `Args` class (note that when run generally, models will write such a file to disk in their directory, so you can just copy and paste the file from the model you want to examine), as well as a configuration describing which GPUs you have available on the system and how many models you want to run on each GPU at a given time, and how many GPUs each model needs (usually both of the latter are 1). There is a sample config available in `latent_patient_trajectories/representation_learner/sample_task_generalizability_config.json`, and note that the config must be renamed to `task_generalizability_config.json` for actual use. You additionally can specify args according to the `TaskGeneralizabilityArgs` class in the `args.py` file.

## Analysis Notebooks
We've kept several analysis notebooks in the `Jupyter Notebooks` subfolder, largely to serve as an inspiration
for analyzing the runs and experiments you may perform with this repository. The raw results files that back
these analyses in our work are not distributed with this repository, as they need to go through the same
validation procedure as our data would, but they are what promted the analytic notebooks presented here. This
allows interested parties to see more details around what our analyses suggest and some of our experimental
results.
