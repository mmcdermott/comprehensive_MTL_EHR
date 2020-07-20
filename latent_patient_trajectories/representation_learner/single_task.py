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
    config_filepath = os.path.join(exp_dir, ST_CFG_FILENAME)
    with open(config_filepath, mode='r') as f: config = json.loads(f.read())

    args_filepath = os.path.join(exp_dir, ST_BASE_ARGS_FILENAME)
    args = Args.from_json_file(args_filepath)

    return config, args

class Runner():
    #    #&> {{log_file}}\
    #COMMAND_TEMPLATE = """\
    #    cd {scripts_dir}
    #    PROJECTS_BASE={projects_base} CUDA_VISIBLE_DEVICES={{gpus}} {python_path} run_v2_model.py \
    #      --run_dir="{{run_dir}}" \
    #      --do_load_from_dir
    #""".format(
    #    scripts_dir=SCRIPTS_DIR,
    #    projects_base=os.environ['PROJECTS_BASE'],
    #    python_path=PYTHON_EXECUTABLE_PATH
    #).strip()
    #ENV =

    def __init__(
        self,
        gpus_per_model,
        run_dir,
#        task_setting_fine_tune_dir,
        gpu_queue,
        do_train = True,
        do_eval  = True,
        do_fine_tune = True,
        do_fine_tune_eval = True,
        slurm =False,
        slurm =False,
        partition='gpu',
    ):
        assert do_train or do_eval or do_fine_tune or do_fine_tune_eval, "Must do something!"

        self.gpus_per_model = gpus_per_model
        self.run_dir = run_dir
#        self.task_setting_fine_tune_dir = task_setting_fine_tune_dir
        self.timings_file      = os.path.join(run_dir, 'timings.json')
        self.stdout_file       = os.path.join(run_dir, 'stdout.txt')
        self.stderr_file       = os.path.join(run_dir, 'stderr.txt')
        self.gpu_queue         = gpu_queue
        self.do_train          = do_train
        self.do_eval           = do_eval
        self.do_fine_tune      = do_fine_tune
        self.do_fine_tune_eval = do_fine_tune_eval
        self.slurm             = slurm
        self.partition         = partition
        
    def call_slurm(self):
        env = {'PROJECTS_BASE': os.environ['PROJECTS_BASE']}
        prog = PYTHON_EXECUTABLE_PATH
        train_args = ['run_v2_model.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']
        eval_args  = ['evaluate.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']

        fine_tune_args = ['fine_tune_task.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']
        fine_tune_eval_args = [
            'evaluate.py', '--run_dir=%s' % self.task_setting_fine_tune_dir, '--do_load_from_dir'
        ]
        
        path_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))

        # just print the bash script
        bash_script=f"""#!/bin/bash
#SBATCH -p {self.partition}
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH --output {os.path.join(self.run_dir, "train_%j.log")}"""
        bash_script_name=os.path.join(self.run_dir, 'train_task_gen.sh')
        
        with open(bash_script_name, 'w') as f:
            f.writelines(bash_script)
        bash_script=f"""
SEARCH_DIR={self.run_dir}
cd {os.path.join(path_root,'Scripts/End to End')}"""

        if self.do_train:
            bash_script=f"""
python run_v2_model.py --run_dir $SEARCH_DIR --do_load_from_dir"""
#             bash_script=f"""
# {prog} run_v2_model.py $SEARCH_DIR --do_load_from_dir"""
            with open(bash_script_name, 'a') as f:
                f.writelines(bash_script)

        if self.do_eval:
            bash_script=f"""
python -u evaluate.py --run_dir $SEARCH_DIR --do_load_from_dir"""
#             bash_script=f"""
# {prog} -u evaluate.py --run_dir $SEARCH_DIR --do_load_from_dir"""
            with open(bash_script_name, 'a') as f:
                f.writelines(bash_script)
            
        print(f"sbatch {bash_script_name};\n")
        try:

            with open(self.stdout_file, mode='w') as stdout_h, open(self.stderr_file, mode='w') as stderr_h:
                subprocess.run(f"sbatch -W {bash_script_name}",shell=True, env=os.environ.copy(), cwd=SCRIPTS_DIR, stdout=stdout_h, stderr=stderr_h)
        except Exception as e:
            print("run dir %s failed! Exception: %s" % (self.run_dir, e))
            result = e
        

        return ('submitted',)

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
        train_args = ['run_v2_model.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']
        eval_args  = ['evaluate.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']

#        fine_tune_args = ['fine_tune_task.py', '--run_dir=%s' % self.run_dir, '--do_load_from_dir']
#        fine_tune_eval_args = [
#            'evaluate.py', '--run_dir=%s' % self.task_setting_fine_tune_dir, '--do_load_from_dir'
#        ]
        
        

        print("Running for task %s with gpus %s" % (self.run_dir, ', '.join(gpus)))
        
        

        try:
            st = time.time()
            results = []
            timings = {}
            with open(self.stdout_file, mode='w') as stdout_h, open(self.stderr_file, mode='w') as stderr_h:
                if self.do_train:
                    tr_st = time.time()
                    results.append(subprocess.run(
                        [prog] + train_args, env=env, cwd=SCRIPTS_DIR, stdout=stdout_h, stderr=stderr_h
                    ))
                    timings['train'] = time.time() - tr_st
                if self.do_eval:
                    ev_st = time.time()
                    results.append(subprocess.run(
                        [prog] + eval_args, env=env, cwd=SCRIPTS_DIR, stdout=stdout_h, stderr=stderr_h
                    ))
                    timings['eval'] = time.time() - ev_st
                if self.do_fine_tune:
                    ft_st = time.time()
                    results.append(subprocess.run(
                        [prog] + fine_tune_args, env=env, cwd=SCRIPTS_DIR, stdout=stdout_h, stderr=stderr_h
                    ))
                    timings['fine_tune'] = time.time() - ft_st
                if self.do_fine_tune_eval:
                    fte_st = time.time()
                    results.append(subprocess.run(
                        [prog] + fine_tune_eval_args, env=env, cwd=SCRIPTS_DIR, stdout=stdout_h,
                        stderr=stderr_h,
                    ))
                    timings['fine_tune_eval'] = time.time() - fte_st
            timings['total'] = time.time() - st

            with open(self.timings_file, mode='w') as f: f.write(json.dumps(timings))
        except Exception as e:
            print("run dir %s failed! Exception: %s" % (self.run_dir, e))
            result = e
        finally:
            for gpu in gpus: self.gpu_queue.put(gpu)

        return tuple(results)
    
    def __call__(self):
        if self.slurm:
            return self.call_slurm()
        else:
            return self.call_no_slurm()

def main(single_task_args, tqdm=tqdm):
    exp_dir = single_task_args.exp_dir
    single_task_args.to_json_file(os.path.join(exp_dir, ST_EXP_ARGS_FILENAME))
    print(single_task_args, ST_EXP_ARGS_FILENAME)

    config, base_model_args = read_config_and_args(exp_dir)

    assert len(config['gpus']) >= config['gpus_per_model'], "Invalid config!"
    if config['gpus_per_model'] > 1: assert config['models_per_gpu'] == 1, "Not yet supported."

    rotation = single_task_args.rotation
    base_dir = os.path.join(exp_dir, str(rotation))
    if not os.path.isdir(base_dir): os.makedirs(base_dir)

    base_model_args.rotation = rotation
    base_model_args.do_overwrite = True

    expected_filenames = [
        'task_weights.pkl',
        #'tuning_task_info.pkl',
        #'tuning_perf_metrics.pkl',
        #'test_task_info.pkl',
        #'test_perf_metrics.pkl',
        'model.epoch-%d' % (base_model_args.epochs - 1),
    ]

    #if task_generalizability_args.do_save_all_reprs:
    #    expected_filenames.extend(['tuning_reprs.pkl', 'test_reprs.pkl'])

    gpus_available = mp.Queue(maxsize=len(config['gpus']) * config['models_per_gpu'])
    for gpu in config['gpus']:
        for _ in range(config['models_per_gpu']): gpus_available.put_nowait(gpu)
    print("Loaded %d gpus into the queue" % gpus_available.qsize())

    runners = []
    # TODO(mmd): Put in config
    # TODO(mmd): Wrong granularity
    for ablation_setting in ABLATION_GROUPS.keys():
        task_setting_str = ablation_setting
        task_dir = os.path.join(base_dir, task_setting_str)
        if not os.path.isdir(task_dir): os.makedirs(task_dir)

        expected_final_filepaths = [os.path.join(task_dir, fn) for fn in expected_filenames]
        task_complete = all(os.path.isfile(fp) for fp in expected_final_filepaths)

        if task_complete: continue

        task_setting_args = copy.deepcopy(base_model_args)
        task_setting_args.run_dir = task_dir
        # everything but current task    
        task_setting_args.ablate  = [k for k in ABLATION_GROUPS.keys() if k != task_setting_str] 

        task_setting_args.to_json_file(os.path.join(task_dir, ARGS_FILENAME))

#        task_setting_fine_tune_args = FineTuneArgs(
#            run_dir = task_dir,
#            fine_tune_task = task_setting_str,
#            num_dataloader_workers = 8, # should be in arg...
#            do_match_train_windows = single_task_args.do_match_FT_train_windows,
#        )
#        task_setting_fine_tune_args.to_json_file(os.path.join(task_dir, FINE_TUNE_ARGS_FILENAME))

        task_setting_eval_args = EvalArgs(
            run_dir = task_dir,
            rotation = rotation,
            do_save_all_reprs = True,
            do_eval_train = True,
            do_eval_tuning = True,
            do_eval_test = True,
            num_dataloader_workers = 8,
        )
        task_setting_eval_args.to_json_file(os.path.join(task_dir, EVAL_ARGS_FILENAME))

#        task_setting_fine_tune_dir = os.path.join(task_dir, task_setting_str)
#        if not os.path.exists(task_setting_fine_tune_dir): os.makedirs(task_setting_fine_tune_dir)

#        task_setting_fine_tune_eval_args = EvalArgs(
#            run_dir = task_setting_fine_tune_dir,
#            rotation = rotation,
#            do_save_all_reprs = False,
#            do_eval_train = False,
#            do_eval_tuning = True,
#            do_eval_test = True,
#            num_dataloader_workers = 8,
#        )
#        task_setting_fine_tune_eval_args.to_json_file(os.path.join(task_setting_fine_tune_dir, EVAL_ARGS_FILENAME))
        
            
            

        runners.append(Runner(
            gpus_per_model = config['gpus_per_model'],
            run_dir = task_dir,
#            task_setting_fine_tune_dir = task_setting_fine_tune_dir,
            gpu_queue = gpus_available,
            do_train = single_task_args.do_train,
            do_eval = single_task_args.do_eval,
            do_fine_tune = single_task_args.do_fine_tune,
            do_fine_tune_eval = single_task_args.do_fine_tune_eval,
            slurm = task_generalizability_args.slurm,
            partition = task_generalizability_args.partition,
        ))
    

    processes = [mp.Process(target=r) for r in runners]
    for process in processes: process.start()

    results = [process.join() for process in processes]
    
    if single_task_args.slurm:
        return

    with open(os.path.join(exp_dir, 'results.pkl'), mode='wb') as f: pickle.dump(results, f)
