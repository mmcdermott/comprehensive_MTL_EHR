# Generic Imports
import copy, math, itertools, json, os, queue, subprocess, sys, time
import multiprocessing as mp

from tqdm import tqdm

# LPT Imports
from . import run_model

from ..constants import *
from .args import *
from .evaluator import *

PYTHON_EXECUTABLE_PATH = sys.executable
SCRIPTS_DIR = '/%s' % os.path.join(*(os.path.realpath(__file__).split('/')[:-3] + ['Scripts','End to End']))

def task_setting_to_str(task_setting):
    return task_setting.replace(' ', '_')


def read_config_and_args(exp_dir):
    """
    Reads a json task generalizability config, e.g.:
    {
      "gpus_per_model": 4,
      "models_per_gpu": 1,
      "gpus": [0, 1, 2, 3]
    }
    """
    config_filepath = os.path.join(exp_dir, TASK_GEN_CFG_FILENAME)
    with open(config_filepath, mode='r') as f: config = json.loads(f.read())

    args_filepath = os.path.join(exp_dir, TASK_GEN_BASE_ARGS_FILENAME)
    args = Args.from_json_file(args_filepath)

    return config, args

class Runner():
    def __init__(
        self,
        gpus_per_model,
        run_dir,
        task_setting_fine_tune_dir,
        gpu_queue,
        do_train = True,
        do_eval  = True,
        do_fine_tune = True,
        do_fine_tune_eval = True,
        slurm =False,
        partition='gpu',
        slurm_args=None,
        do_small_data=False,
        do_imbalanced_sex_data=False,
        do_imbalanced_race_data=False,
        do_frozen_representation=True,
        do_free_representation=False,
        do_single_task=False,
        do_masked_imputation_PT = False,
        do_copy_masked_imputation_PT = False,
    ):
        assert do_train or do_eval or do_fine_tune or do_fine_tune_eval or do_small_data or do_imbalanced_sex_data or do_imbalanced_race_data, "Must do something!"

        self.gpus_per_model               = gpus_per_model
        self.run_dir                      = run_dir
        self.task_setting_fine_tune_dir   = task_setting_fine_tune_dir
        self.timings_file                 = os.path.join(run_dir, 'timings.json')
        self.stdout_file                  = os.path.join(run_dir, '{key}_stdout.txt')
        self.stderr_file                  = os.path.join(run_dir, '{key}_stderr.txt')
        self.gpu_queue                    = gpu_queue
        self.do_train                     = do_train
        self.do_eval                      = do_eval
        self.do_fine_tune                 = do_fine_tune
        self.do_fine_tune_eval            = do_fine_tune_eval
        self.slurm                        = slurm
        self.partition                    = partition
        self.slurm_args                   = slurm_args
        self.do_small_data                = do_small_data
        self.do_imbalanced_sex_data       = do_imbalanced_sex_data
        self.do_imbalanced_race_data      = do_imbalanced_race_data
        self.do_frozen_representation     = do_frozen_representation
        self.do_free_representation       = do_free_representation
        self.do_single_task               = do_single_task
        self.do_masked_imputation_PT      = do_masked_imputation_PT
        self.do_copy_masked_imputation_PT = do_copy_masked_imputation_PT

        if self.do_single_task or (self.do_masked_imputation_PT and self.do_copy_masked_imputation_PT):
            assert not self.do_train, "Shouldn't pre-train a single-task/MI model!"
            assert not self.do_eval, "Shouldn't eval a (non-existent) PT model in single-task/MI mode!"

    def run(self, args, env, key, timings, results):
        st = time.time()
        stdout_file = self.stdout_file.format(key=key)
        stderr_file = self.stderr_file.format(key=key)
        with open(stdout_file, mode='w') as stdout_h, open(stderr_file, mode='w') as stderr_h:
            ev_st = time.time()
            results.append(subprocess.run(
                [PYTHON_EXECUTABLE_PATH] + args, env=env, cwd=SCRIPTS_DIR, stdout=stdout_h, stderr=stderr_h,
                check = True
            ))
            timings[key] = time.time() - ev_st

    def call_no_slurm(self):
        gpus=set()
        while len(gpus) < self.gpus_per_model:
            try:
                new_gpu = self.gpu_queue.get(block=True, timeout=30)
                if new_gpu in gpus: self.gpu_queue.put(new_gpu)
                else: gpus.update([new_gpu])
            except queue.Empty as e:
                if gpus:
                    for gpu in gpus: self.gpu_queue.put(gpu)
                    time.sleep(90)
                pass

        gpus = [str(g) for g in gpus]

        env = {'PROJECTS_BASE': os.environ['PROJECTS_BASE'], 'CUDA_VISIBLE_DEVICES': ','.join(gpus)}
        prog = PYTHON_EXECUTABLE_PATH

        pretrain_args = ['run_v2_model.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']
        eval_args = ['evaluate.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']
        fine_tune_args = ['fine_tune_task.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']

        print("Running for task %s with gpus %s" % (self.run_dir, ', '.join(gpus)))

        data_fracs = [1]
        if self.do_small_data: data_fracs.extend(SMALL_DATA_FRACS)

        try:
            st = time.time()
            results = []
            timings = {}
            if self.do_train: self.run(pretrain_args, env, 'train', timings, results)
            if self.do_eval: self.run(eval_args, env, 'eval', timings, results)
            if self.do_fine_tune: self.run(fine_tune_args, env, 'fine_tune', timings, results)
            if self.do_fine_tune_eval:
                for frac in data_fracs:
                    for do, suffix in [
                        (self.do_frozen_representation, "FTD"), (self.do_free_representation, "FTF"),
                    ]:
                        if not do: continue

                        fine_tune_dir_name = self.task_setting_fine_tune_dir
                        if frac != 1: fine_tune_dir_name += f"_{str(frac).replace('.', '-')}"

                        fine_tune_dir = os.path.join(fine_tune_dir_name, suffix)
                        assert os.path.isdir(fine_tune_dir)

                        fine_tune_eval_args = [
                            'evaluate.py', f"--run_dir={fine_tune_dir}", '--do_load_from_dir'
                        ]
                        self.run(
                            fine_tune_eval_args, env, f"{100*frac}%_data_{suffix}_eval", timings, results
                        )

            with open(self.timings_file, mode='w') as f: f.write(json.dumps(timings))
        except Exception as e:
            print("run dir %s failed! Exception: %s" % (self.run_dir, e))
            result = e
        finally:
            for gpu in gpus: self.gpu_queue.put(gpu)

        return tuple(results)

    def __call__(self):
        if self.slurm:
            raise NotImplementedError("Slurm support has been deprecated.")
        else:
            return self.call_no_slurm()

def main(task_generalizability_args, tqdm=tqdm):
    exp_dir = task_generalizability_args.exp_dir
    task_generalizability_args.to_json_file(os.path.join(exp_dir, TASK_GEN_EXP_ARGS_FILENAME))
    print(task_generalizability_args, TASK_GEN_EXP_ARGS_FILENAME)

    config, base_model_args = read_config_and_args(exp_dir)
    assert os.path.exists(base_model_args.dataset_dir), f'{base_model_args.dataset_dir} does not exist'

    assert (
        task_generalizability_args.do_frozen_representation or
        task_generalizability_args.do_free_representation
    ), "Need to do either FTF or FTD!"

    assert task_generalizability_args.do_eicu == base_model_args.do_eicu

    if task_generalizability_args.do_single_task:
        assert not task_generalizability_args.do_train, "Can't pre-train a single-task model!"
        assert not task_generalizability_args.do_eval, "Can't eval a pre-trained model in single-task mode!"
        assert not task_generalizability_args.do_frozen_representation, \
            "FTD doesn't make sense in single-task mode!"
        assert not task_generalizability_args.do_masked_imputation_PT, \
            "Can't do both single-task and Masked Imputation!"
    elif task_generalizability_args.do_masked_imputation_PT:
        some_PT = (
            task_generalizability_args.do_train or task_generalizability_args.do_copy_masked_imputation_PT
        )
        assert some_PT, "It isn't masked imputation PT with PT!"
        if task_generalizability_args.do_copy_masked_imputation_PT:
            assert not task_generalizability_args.do_train, "Shouldn't copy and PT!"
            assert not task_generalizability_args.do_eval, "Shouldn't copy and eval!"

    assert len(config['gpus']) >= config['gpus_per_model'], "Invalid config!"
    if config['gpus_per_model'] > 1: assert config['models_per_gpu'] == 1, "Not yet supported."

    rotation = task_generalizability_args.rotation
    base_dir = os.path.join(exp_dir, str(rotation))
    if not os.path.isdir(base_dir): os.makedirs(base_dir)

    do_masked_imputation_PT = task_generalizability_args.do_masked_imputation_PT
    do_copy_masked_imputation_PT = task_generalizability_args.do_copy_masked_imputation_PT

    base_model_args.rotation = rotation
    if do_masked_imputation_PT and do_copy_masked_imputation_PT: base_model_args.do_overwrite = False
    else: base_model_args.do_overwrite = True

    gpus_available = mp.Queue(maxsize=len(config['gpus']) * config['models_per_gpu'])
    for gpu in config['gpus']:
        for _ in range(config['models_per_gpu']): gpus_available.put_nowait(gpu)
    print("Loaded %d gpus into the queue" % gpus_available.qsize())

    do_single_task = task_generalizability_args.do_single_task
    single_task = task_generalizability_args.single_task

    runners = []
    # TODO(mmd): Put in config
    # TODO(mmd): Wrong granularity
    ablation_settings = EICU_ABLATION_GROUPS if task_generalizability_args.do_eicu else ABLATION_GROUPS.keys()
    print(f"Running on {', '.join(ablation_settings)}")

    if do_masked_imputation_PT:
        assert do_copy_masked_imputation_PT, \
            "Currently doesn't support masked imputation PT, so must be copied."

        if do_copy_masked_imputation_PT:
            tuning_eval_file = 'tuning_perf_metrics.pkl'
            test_eval_file = 'test_perf_metrics.pkl'
            last_model_file = f"model.epoch-{base_model_args.epochs-1}"

            for fn in (tuning_eval_file, test_eval_file, last_model_file):
                assert os.path.isfile(os.path.join(base_dir, fn)), "Missing required file for copied PT!"
        else:
            raise NotImplementedError("Not yet supported.")
            # TODO(mmd): Here is where we'd re-do PT training.

    for ablation_setting in ablation_settings:
        if do_single_task and (ablation_setting != single_task): continue
        print(f"Setting up Runner for {ablation_setting}")

        task_setting_str = ablation_setting
        if do_single_task:
            task_dir = base_dir
        else:
            task_dir = os.path.join(base_dir, task_setting_str)
            if not os.path.isdir(task_dir): os.makedirs(task_dir)

        if not do_masked_imputation_PT:
            task_setting_args = copy.deepcopy(base_model_args)
            task_setting_args.run_dir = task_dir

            if do_single_task: task_setting_args.ablate = [t for t in ablation_settings if t != task_setting_str]
            else: task_setting_args.ablate = task_setting_str

            task_setting_args.to_json_file(os.path.join(task_dir, ARGS_FILENAME))

        if (not do_single_task) and task_generalizability_args.do_eval:
            task_setting_eval_args = EvalArgs(
                run_dir = task_dir,
                rotation = rotation,
                do_save_all_reprs = False,
                do_eval_train = False,
                do_eval_tuning = True,
                do_eval_test = True,
                do_eicu = task_generalizability_args.do_eicu,
                num_dataloader_workers = 8,
            )
            task_setting_eval_args.to_json_file(os.path.join(task_dir, EVAL_ARGS_FILENAME))

        task_setting_fine_tune_args = FineTuneArgs(
            run_dir = task_dir,
            fine_tune_task = task_setting_str,
            num_dataloader_workers = 8, # should be in arg...
            do_match_train_windows = task_generalizability_args.do_match_FT_train_windows, #todo add early stopping
            train_embedding_after = task_generalizability_args.train_embedding_after,
            balanced_race = task_generalizability_args.do_imbalanced_race_data,
            do_eicu = task_generalizability_args.do_eicu,
            do_frozen_representation = task_generalizability_args.do_frozen_representation,
            do_free_representation = task_generalizability_args.do_free_representation,
            do_single_task = task_generalizability_args.do_single_task,
            do_small_data = task_generalizability_args.do_small_data,
            do_masked_imputation_PT = do_masked_imputation_PT,
        )
        task_setting_fine_tune_args.to_json_file(os.path.join(task_dir, FINE_TUNE_ARGS_FILENAME))

        task_setting_fine_tune_dir = os.path.join(task_dir, task_setting_str)

        # TODO(mmd): This is terrible. This should be standardized, and it should only be needed in one place.
        # We should just move the eval stuff to the fine_tune.py code.
        data_fracs = [1]
        if task_generalizability_args.do_small_data: data_fracs.extend(SMALL_DATA_FRACS)

        for frac in data_fracs:
            for do, suffix in [
                (task_generalizability_args.do_frozen_representation, "FTD"),
                (task_generalizability_args.do_free_representation, "FTF"),
            ]:
                if not do: continue

                fine_tune_dir_name = task_setting_str
                if frac != 1: fine_tune_dir_name += f"_{str(frac).replace('.', '-')}"

                fine_tune_dir = os.path.join(task_dir, fine_tune_dir_name, suffix)
                if not os.path.isdir(fine_tune_dir): os.makedirs(fine_tune_dir)

                task_setting_fine_tune_eval_args = EvalArgs(
                    run_dir = fine_tune_dir,
                    rotation = rotation,
                    do_save_all_reprs = False,
                    do_eval_train = False,
                    do_eval_tuning = True,
                    do_eval_test = True,
                    do_eicu = task_generalizability_args.do_eicu,
                    num_dataloader_workers = 8,
                )
                task_setting_fine_tune_eval_args.to_json_file(os.path.join(fine_tune_dir, EVAL_ARGS_FILENAME))

        runners.append(Runner(
            gpus_per_model = config['gpus_per_model'],
            run_dir = task_dir,
            task_setting_fine_tune_dir = task_setting_fine_tune_dir,
            gpu_queue = gpus_available,
            do_train = task_generalizability_args.do_train,
            do_eval = task_generalizability_args.do_eval,
            do_fine_tune = task_generalizability_args.do_fine_tune,
            do_fine_tune_eval = task_generalizability_args.do_fine_tune_eval,
            slurm = task_generalizability_args.slurm,
            partition = task_generalizability_args.partition,
            slurm_args = task_generalizability_args.slurm_args,
            do_small_data=task_generalizability_args.do_small_data,
            do_imbalanced_sex_data = task_generalizability_args.do_imbalanced_sex_data,
            do_imbalanced_race_data = task_generalizability_args.do_imbalanced_race_data,
            do_frozen_representation = task_generalizability_args.do_frozen_representation,
            do_free_representation = task_generalizability_args.do_free_representation,
            do_single_task = task_generalizability_args.do_single_task,
            do_masked_imputation_PT = do_masked_imputation_PT,
            do_copy_masked_imputation_PT = not do_copy_masked_imputation_PT,
        ))


    processes = [mp.Process(target=r) for r in runners]
    for process in processes: process.start()

    results = [process.join() for process in processes]

    if task_generalizability_args.slurm:
        return

    with open(os.path.join(exp_dir, 'results.pkl'), mode='wb') as f: pickle.dump(results, f)
