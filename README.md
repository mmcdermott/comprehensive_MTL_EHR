# A Comprehensive EHR Timeseries Pre-training Benchmark
Source code for our paper (link forthcoming) defining a pre-training benchmark system for EHR timeseries data.
Contact mmd@mit.edu and bretnestor@cs.toronto.edu with any questions. Pending interest from the community, we're eager to make this as usable as possible, and will respond promptly to any issues or questions.

# Install

Set up the repository
```
conda env create --name comprehensive_EHR_PT -f env.yml
conda activate comprehensive_EHR_PT
```

# Obtaining Data
Copies of pre-processed dataset splits used in the paper can be obtained via Google Cloud. To access them, you
must ensure that you have obtained GCP access via physionet.org for the requisite datasets. See
[https://mimic.physionet.org/gettingstarted/cloud/](https://mimic.physionet.org/gettingstarted/cloud/) for
instructions on obtaining Physionet GCP access.

  1. MIMIC-III Dataset: [https://console.cloud.google.com/storage/browser/ehr_pretraining_benchmark_mimic](https://console.cloud.google.com/storage/browser/ehr_pretraining_benchmark_mimic)
  2. eICU Dataset: [https://console.cloud.google.com/storage/browser/ehr_pretraining_benchmark_eicu](https://console.cloud.google.com/storage/browser/ehr_pretraining_benchmark_eicu)

# Usage
## Args in General
Arguments for all scripts are described in the `latent_patient_trajectories/representation_learner/args.py`
file. This file has some base classes, then argument classes (with specific args requested) for all functions.
It is a good reference to determine what a specific script expects. Note this class allows you to (and we
recommend) pre-setting all args for scripts in (appropriately named) json files in the relevant experiment
directories, then simply passing the directory to the given script (according to the appropriate `arg`) and
adding `--do_load_from_dir`, at which point the script will load all arguments from the json file
automatically. Note that some args (specifically `regression_task_weight`, which should always be 0, `notes`, which
should always be `no_notes`, and `task_weights_filepath`, which should always be `''`) are held-out args from older versions of the code, and can be largely ignored. Similarly, the modeltype specific args corresponding to CNN, Self-attention, or Linear projection models are also no longer used. Some sample args for different settings are given in `Sample Args`. Please raise a github issue or contact mmd@mit.edu or bretnestor@cs.toronto.edu with any questions.

## Hyperparameter Tuning
To perform hyperparameter tuning, set up a base experimental directory, and add a config file describing your
hyperparameter search in that directory. This file must be named according to the `HYP_CONFIG_FILENAME`
constant in `latent_patient_trajectories/constants.py` file, which is (as of 7/20/20) set to
`hyperparameter_search_config.json`. A sample config file is shown in the file
`latent_patient_trajectories/representation_learner/sample_hyperopt_config.json`. 

Then, run the script `Scripts/End to End/hyperopt_model.py` (with appropriate args, as described in the
`args.py` file referenced above under the class `HyperparameterSearchArgs`) to kick off your search. 

## Generic Runs
To perform a generic run, training a multi- or single-task model, or a masked imputation model, use the `Args`
class in `args.py` and the `run_model.py` script. As with everything else, you will need to specify a base
directory, and many other args to describe the architecture you want to use and training details.

### Evaluation
Evaluating a pre-trained run can be accomplished with the `EvalArgs` class and the `evaluate.py` script. You
will need to specify the model's training directory (e.g., the directory passed to `run_model.py`) so the
script knows what model to reload.

## Task-Generalizability Analyses
These runs consist of pre-training a model either via masked imputation or via multi-task pre-training, in
which the model is pre-trained on all tasks except for one held-out task, then fine-tuning the
model on that held-out task, and evaluating both the pre-trained and fine-tuned models on all tasks. This
could be done manually, through repeated use of the `run_model.py` script and the use of the `--ablate` arg,
but we have a helper script that can manage doing all requisite runs across multiple GPUs in parallel on a
single-machine. To use this, you must first create a base directory for this experiment, which will ultimately
hold all runs associated with this experiment (including pre-trained and fine=tuned). In this directory, you
will specify the model's base args (which will be duplicated and used in all pre-training and fine-tuning
experiments, with the ablate arg automatically adjusted to perform the appropriate experiments) as a `json`
file which is parseable by the `Args` class (note that when run generally, models will write such a file to
disk in their directory, so you can just copy and paste the file from the model you want to examine), as well
as a configuration describing which GPUs you have available on the system and how many models you want to run
on each GPU at a given time, and how many GPUs each model needs (usually both of the latter are 1). There is a
sample config available in
`latent_patient_trajectories/representation_learner/sample_task_generalizability_config.json`, and note that
the config must be renamed to `task_generalizability_config.json` for actual use. You additionally can specify
args according to the `TaskGeneralizabilityArgs` class in the `args.py` file.

## Analysis Notebooks
All our results can be analyzed via the `All Results.ipynb` notebook. Input files for this notebook are
available upon request -- as they aggregate both within MIMIC-III and eICU, we cannot use GCP so instead must
validate your physionet access directly.
