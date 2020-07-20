# Generic Imports
import copy, math, itertools, json, os, shutil, time
import matplotlib.pyplot as plt

from hyperopt import fmin, hp, pyll, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm

# LPT Imports
from . import run_model

from ..constants import *
from .args import *
from .evaluator import *

def null_and_raise(*args, **kwargs):
    raise NotImplementedError("This shouldn't be called..." + str(args) + str(kwargs))

HP_QUANTIZATIN = 1
HP_METHODS = {
    'quniform': hp.quniform,
    'uniform': hp.uniform,
    'choice': hp.choice,
    'loguniform': hp.loguniform,
    'lognormal': hp.lognormal,
    'nested_choice': null_and_raise,
}
HP_ALGS = {
    'tpe.suggest': tpe.suggest,
}

def update_perf_metrics(hyperparameter_search_dir, dataset, tqdm=None):
    rotations = set(x for x in os.listdir(hyperparameter_search_dir)).intersection(set(str(i) for i in range(10)))
    print("Observe runs for rotations: %s" % ', '.join(rotations))

    binary_multilabel_tasks = dataset.get_binary_multilabel_keys()
    next_timepoint_was_measured_tasks = dataset.dfs['next_timepoint_was_measured'].columns
    next_timepoint_tasks = [c for c in dataset.dfs['next_timepoint'].columns if c[1] == 'mean']

    rotations_rng = rotations if len(rotations) < 4 or tqdm is None else tqdm(rotations)
    for rotation in rotations_rng:
        rotation_dir = os.path.join(hyperparameter_search_dir, rotation)

        run_names = [r for r in os.listdir(rotation_dir) if r != 'trials.pkl']
        run_names_rng = run_names if tqdm is None else tqdm(run_names)

        for run_name in run_names:
            print(run_name)
            run_dir = os.path.join(rotation_dir, run_name)
            if not os.path.isdir(run_dir):
                print(run_dir)
                continue

            args_filepath = os.path.join(run_dir, ARGS_FILENAME)
            if not os.path.isfile(args_filepath): continue

            tuning_task_info_filepath = os.path.join(run_dir, 'tuning_task_info.pkl')
            if not os.path.isfile(tuning_task_info_filepath): continue

            with open(os.path.join(tuning_task_info_filepath), mode='rb') as f:
                tuning_task_info = pickle.load(f)

            tuning_perf_metrics = []
            for task_performance in tuning_task_info:
                perf_metrics = get_performance_metrics(task_performance, all_vocabs=dataset.all_vocabs)

                for k, v in perf_metrics.items():
                    if k[0] == 'next_timepoint' and k[1].endswith('all)'):
                        perf_metrics[k] = pd.Series(perf_metrics[k], index=next_timepoint_tasks)
                    elif k[0] == 'next_timepoint_was_measured' and k[1].endswith('all)'):
                        perf_metrics[k] = pd.Series(v, index=next_timepoint_was_measured_tasks)
                    elif k[0] == 'tasks_binary_multilabel' and k[1].endswith('all)'):
                        perf_metrics[k] = pd.Series(v, index=binary_multilabel_tasks)
                tuning_perf_metrics.append(perf_metrics)

            tuning_perf_metrics_filepath = os.path.join(run_dir, 'tuning_perf_metrics.pkl')
            if os.path.exists(tuning_perf_metrics_filepath):
                shutil.copy(tuning_perf_metrics_filepath, '%s.bkp' % tuning_perf_metrics_filepath)

            with open(tuning_perf_metrics_filepath, mode='wb') as f:
                pickle.dump(tuning_perf_metrics, f)

            test_task_info_filepath = os.path.join(run_dir, 'test_task_info.pkl')
            if not os.path.isfile(test_task_info_filepath): continue

            with open(os.path.join(test_task_info_filepath), mode='rb') as f:
                test_task_info = pickle.load(f)

            test_perf_metrics = []
            for task_performance in test_task_info:
                perf_metrics = get_performance_metrics(task_performance, all_vocabs=dataset.all_vocabs)

                for k, v in perf_metrics.items():
                    if k[0] == 'next_timepoint' and k[1].endswith('all)'):
                        perf_metrics[k] = pd.Series(perf_metrics[k], index=next_timepoint_tasks)
                    elif k[0] == 'next_timepoint_was_measured' and k[1].endswith('all)'):
                        perf_metrics[k] = pd.Series(v, index=next_timepoint_was_measured_tasks)
                    elif k[0] == 'tasks_binary_multilabel' and k[1].endswith('all)'):
                        perf_metrics[k] = pd.Series(v, index=binary_multilabel_tasks)
                test_perf_metrics.append(perf_metrics)

            test_perf_metrics_filepath = os.path.join(run_dir, 'test_perf_metrics.pkl')
            if os.path.exists(test_perf_metrics_filepath):
                shutil.copy(test_perf_metrics_filepath, '%s.bkp' % test_perf_metrics_filepath)

            with open(test_perf_metrics_filepath, mode='wb') as f:
                pickle.dump(test_perf_metrics, f)

def estimate_growth_rate(l):
    return np.mean([(e2/e1) for e1, e2 in zip(l[:-1], l[1:])])

def args_to_params(args, raw_config_json):
    global nested_args_exist
    nested_args_exist=False
    constant_args, global_args, nested_args = set(), set(), {}
    # old models did not have nested parameters
    def get_params(r, container=global_args):
        for k, v in r.items():
            if v['method'] == 'constant': constant_args.update([k])
            elif v['method'] == 'nested_choice':
                nested_args_exist=True
                container.update([k])
                for k2, r2 in v['params']:
                    nested_args[k2] = set()
                    get_params(r2, nested_args[k2])
            else: container.update([k])

    get_params(raw_config_json)

    spec = {}
    modeltype = args.modeltype
    spec['modeltype'] = (modeltype, {})
    for k, v in vars(args).items():
        if k in global_args and k != 'modeltype': spec[k] = v
        elif nested_args_exist:
            if k in nested_args[modeltype]: spec['modeltype'][1][k] = v

    if modeltype == 'self_attention':
        spec['modeltype'][1]['hidden_size_multiplier'] = args.hidden_size // args.num_attention_heads
    elif modeltype.lower() == 'cnn':
        spec['modeltype'][1]['num_conv_layers'] = len(args.kernel_sizes)
        spec['modeltype'][1]['num_filters_base'] = args.num_filters[0]
        spec['modeltype'][1]['num_filters_growth_rate'] = estimate_growth_rate(args.num_filters)
        spec['modeltype'][1]['kernel_size_base'] = args.kernel_sizes[0]
        spec['modeltype'][1]['kernel_size_growth_rate'] = estimate_growth_rate(args.kernel_sizes)
        spec['modeltype'][1]['num_fc_layers'] = len(args.fc_layer_sizes)
        spec['modeltype'][1]['fc_layer_size_base'] = args.fc_layer_sizes[0]
        spec['modeltype'][1]['fc_layer_growth_rate'] = estimate_growth_rate(args.fc_layer_sizes)
    elif modeltype.lower() == 'gru':
        spec['modeltype'][1]['gru_num_fc_layers'] = len(args.fc_layer_sizes)
        spec['modeltype'][1]['gru_fc_layer_size_base'] = args.fc_layer_sizes[0]
        spec['modeltype'][1]['gru_fc_layer_growth_rate'] = estimate_growth_rate(args.fc_layer_sizes)

    return spec

def merge_dicts(*dicts):
    out_d = {}
    for d in dicts:
        for k, v in d.items():
            if k in out_d and type(v) is dict and type(out_d[k]) is dict: out_d[k] = merge_dicts(out_d[k], v)
            else: out_d[k] = v
    return out_d

def get_errors(analysis_dirs):
    return merge_dicts(*(get_errors_single(d) for d in analysis_dirs))

def get_errors_single(analysis_dir):
    subdirs = os.listdir(analysis_dir)
    errors = {}
    for rotation in range(10):
        rotation = str(rotation)
        if rotation not in subdirs: continue
        rotation_dir = os.path.join(analysis_dir, rotation)
        errors[rotation] = {}

        for run_name in os.listdir(rotation_dir):
            run_dir = os.path.join(rotation_dir, run_name)
            error_filepath = os.path.join(run_dir, 'error.pkl')
            if not os.path.isfile(os.path.join(run_dir, 'error.pkl')): continue
            error_time = datetime.fromtimestamp(os.path.getmtime(error_filepath))

            try:
                with open(error_filepath, mode='rb') as f: error = pickle.load(f)

                params_filepath = os.path.join(run_dir, PARAMS_FILENAME)
                args_filepath = os.path.join(run_dir, ARGS_FILENAME)

                if os.path.isfile(params_filepath):
                    with open(params_filepath, mode='rb') as f: raw_params = pickle.load(f)
                else: raw_params = None

                if os.path.isfile(args_filepath):
                    args = Args.from_json_file(args_filepath)
                    num_epochs = args.epochs
                    completed_training = os.path.isfile(os.path.join(
                        run_dir, 'model.epoch-%d' % (num_epochs-1)
                    ))
                else: args, completed_training = None, None

                errors[rotation][run_name] = (error, error_time, completed_training, raw_params, args)
            except Exception as e:
                print("Can't parse errors!", e)
                errors[rotation][run_name] = (True, error_time, None, None, None)
    return errors

def read_many_dirs(search_dirs, **kwargs):
    all_configs, all_results, all_args, all_params, all_trials = {}, [], [], [], None
    for d in search_dirs:
        config, results, args, params, trials = read_or_recreate_trials(d, **kwargs)
        all_configs[d] = config
        all_results.append(results)
        all_args.append(args)
        all_params.append(params)

        if all_trials is None: all_trials = trials
        else:
            for k, ts in trials.items():
                if k not in all_trials: all_trials[k] = ts
                else:
                    for t in ts.trials:
                        all_trials[k].insert_trial_doc(t)

    for k in all_trials: all_trials[k].refresh()
    return all_configs,merge_dicts(*all_results),merge_dicts(*all_args),merge_dicts(*all_params),all_trials

def read_or_recreate_trials(
    hyperparameter_search_dir, tuning_dataset=None, test_dataset=None, tqdm=None, overwrite=False
):
    config = read_config(hyperparameter_search_dir)[0]

    filepath = os.path.join(hyperparameter_search_dir, HYP_CONFIG_FILENAME)
    with open(filepath, mode='r') as f: raw_config = json.loads(f.read())

    rotations = set(x for x in os.listdir(hyperparameter_search_dir)).intersection(set(str(i) for i in range(10)))
    print("Observe runs for rotations: %s" % ', '.join(rotations))

    all_trials  = {}
    all_results = {}
    all_args    = {}
    all_params = {}

    rotations_rng = rotations if len(rotations) < 4 or tqdm is None else tqdm(rotations)
    for rotation in rotations_rng:
        rotation_results = {}
        rotation_args = {}
        rotation_params = {}

        rotation_dir = os.path.join(hyperparameter_search_dir, rotation)

        run_names = [r for r in os.listdir(rotation_dir) if r != 'trials.pkl']
        run_names_rng = run_names if tqdm is None else tqdm(run_names)

        for run_name in run_names:
            run_dir = os.path.join(rotation_dir, run_name)
            if not os.path.isdir(run_dir):
                print(run_dir)
                continue

            if os.path.isfile(os.path.join(run_dir, 'error.pkl')): continue

            args_filepath = os.path.join(run_dir, ARGS_FILENAME)
            if not os.path.isfile(args_filepath): continue
            args = Args.from_json_file(args_filepath)
            rotation_args[run_name] = args

            params_filepath = os.path.join(run_dir, PARAMS_FILENAME)
            if os.path.isfile(params_filepath):
                with open(params_filepath, mode='rb') as f: rotation_params[run_name] = pickle.load(f)
            else:
                rotation_params[run_name] = args_to_params(rotation_args[run_name], raw_config)

            num_epochs = args.epochs
            completed_training = os.path.isfile(os.path.join(run_dir, 'model.epoch-%d' % (num_epochs - 1)))
            if not completed_training:
                print("Run %s Still training (or errored and didn't report)" % run_name)
                continue

            tuning_result_filepath = os.path.join(run_dir, 'tuning_perf_metrics.pkl')
            if os.path.isfile(tuning_result_filepath):
                with open(tuning_result_filepath, mode='rb') as f:
                    tuning = pickle.load(f)
            else:
                print('Missing tuning for %s' % run_name)
                if tuning_dataset is not None:
                    _, _, tuning = evaluate_multi(
                        tuning_dataset, model_rundir=run_dir, num_random_endpoints=10, batch_size=1024, num_workers=27,
                        evaluate_on_25=True, get_all_reprs=False, tqdm=tqdm
                    )
                    with open(tuning_result_filepath, mode='wb') as f:
                        pickle.dump(tuning, f)
                else:
                    print("Wasn't given a tuning dataset!")
                    continue

            test_result_filepath = os.path.join(run_dir, 'test_perf_metrics.pkl')
            if os.path.isfile(test_result_filepath):
                with open(test_result_filepath, mode='rb') as f:
                    test = pickle.load(f)
            else:
                print('Have tuning but missing test for %s' % run_name)
                if test_dataset is not None:
                    _, _, test = evaluate_multi(
                        test_dataset, model_rundir=run_dir, num_random_endpoints=10, batch_size=1024, num_workers=27,
                        evaluate_on_25=True, get_all_reprs=False, tqdm=tqdm
                    )
                    with open(test_result_filepath, mode='wb') as f:
                        pickle.dump(test, f)
                else:  "Wasn't given a test dataset!"

            rotation_results[run_name] = (tuning, test)

        all_results[rotation] = rotation_results
        all_args[rotation] = rotation_args
        all_params[rotation] = rotation_params

        trials_filepath = os.path.join(rotation_dir, 'trials.pkl')
        if os.path.exists(trials_filepath) and not overwrite:
            with open(trials_filepath, mode='rb') as f: all_trials[rotation] = pickle.load(f)
            continue

        # Rebuild Trials
        # TODO(mmd): Something wrong in misc.idxs...
        trials = Trials(exp_key = 'exp') #hyperparameter_search_dir
        for run_name in rotation_results:
            args = rotation_args[run_name]
            params = rotation_params[run_name]
            perf_metrics, test_perf_metrics = rotation_results[run_name]
            tuning_scores = -pd.Series(ObjectiveFntr.perf_metrics_to_trial_result(perf_metrics))
            test_scores = -pd.Series(ObjectiveFntr.perf_metrics_to_trial_result(test_perf_metrics))

            loss = tuning_scores.mean()
            loss_variance = tuning_scores.std()**2
            test_loss = test_scores.mean()
            test_loss_variance = test_scores.std()**2

            result = {
                'status': STATUS_OK,
                'loss': loss,
                'loss_variance': loss_variance,
                'test_loss': test_loss,
                'test_loss_variance': test_loss_variance,
            }
            spec = params

            trials.insert_trial_doc({
                'tid': run_name,
                'spec': spec,
                'result': result,
                'misc': {
                    'tid': run_name,
                    'cmd': '',
                    'idxs': [],
                    'vals': {k: [v] for k, v in spec.items()},
                },
                'state': '',
                'owner': '',
                'book_time': 0,
                'refresh_time': 0,
                'exp_key': 'exp',# hyperparameter_search_dir,
            })
        trials.refresh()
        all_trials[rotation] = trials

    return config, all_results, all_args, all_params, all_trials

def read_config(search_dir):
    """
    Reads a json hyperparameter search config, e.g.:
    {
      ...
      "batches_per_gradient": {"method": "quniform", "params": [1, 10]},
      "notes":                {"method": "choice", "params": ["no_notes", "integrate_note_bert"]},
      "batch_size":           {"method": "constant", "params": 8},
      "learning_rate":        {"method": "loguniform", "params": [-5, -1]},
      ...
    }
    'method' must be in {'constant'} or any hp.<DIST> name.
    """
    return read_config_blob(read_raw_config_from_dir(search_dir))

def read_raw_config_from_dir(search_dir):
    filepath = os.path.join(search_dir, HYP_CONFIG_FILENAME)

    with open(filepath, mode='r') as f: raw_config = json.loads(f.read())
    return raw_config

def read_config_blob(raw_config):
    constant_params = {}
    hyperopt_space = {}

    for param, param_config in raw_config.items():
        if param_config['method'] == 'constant':
            constant_params[param] = param_config['params']
            continue

        assert param_config['method'] in HP_METHODS, "method %s not yet supported" % param_config['method']
        method = HP_METHODS[param_config['method']]
        is_quantized = param_config['method'].startswith('q')
        is_choice = param_config['method'] == 'choice'
        is_nested_choice = param_config['method'] == 'nested_choice'

        if is_quantized: hyperopt_space[param] = method(param, *param_config['params'], q=1)
        elif is_choice: hyperopt_space[param] = method(param, param_config['params'])
        elif is_nested_choice: hyperopt_space[param] = hp.choice(
            param, [
                (opt, read_config_blob(cfg)[0]) for opt, cfg in param_config['params']
            ]
        )
        else: hyperopt_space[param] = method(param, *param_config['params'])

    return hyperopt_space, constant_params

def get_samples_of_config(search_dir, N=1000, overwrite=False, tqdm=None):
    params_samples_filepath = os.path.join(search_dir, 'config_samples.pkl')
    if not overwrite and os.path.isfile(params_samples_filepath):
        with open(params_samples_filepath, mode='rb') as f: return pickle.load(f)

    cfg, _ = read_config(search_dir)
    items = tqdm(cfg.items()) if tqdm is not None else cfg.items()
    samples = {p: [pyll.stochastic.sample(g) for _ in range(N)] for p, g in items}

    with open(params_samples_filepath, mode='wb') as f: pickle.dump(samples, f)

    return samples

def plot_config(search_dir, N=1000, overwrite=False, tqdm=None):
    samples = get_samples_of_config(search_dir, N=N, overwrite=overwrite, tqdm=tqdm)

    plot_samples_set(samples)

def flatten_samples(samples):
    new_samples = {}
    nested_keys = []
    for k, v in samples.items():
        if type(v[0]) is not tuple: new_samples[k] = v
        else: nested_keys.append(k)

    if not nested_keys: return samples

    N = len(samples[nested_keys[0]])
    keys_to_add = set(nested_keys)
    for k in nested_keys: 
        for i in range(N): keys_to_add.update(samples[k][i][1].keys())
    for k in keys_to_add: new_samples[k] = [np.NaN for _ in range(N)]

    for i in range(N):
        for k in nested_keys:
            s, v = samples[k][i]
            new_samples[k][i] = s
            for k2, v2 in v.items(): new_samples[k2][i] = v2

    for k, v in new_samples.items():
        if len(set(v)) == 1: new_samples.pop(k)

    return new_samples

def plot_samples_set(samples):
    samples = flatten_samples(samples)

    W = math.floor(math.sqrt(len(samples)))
    H = math.ceil(len(samples) / W)

    fig, axes = plt.subplots(nrows=H, ncols=W, figsize=(7*W, 7*H))
    axes = itertools.chain.from_iterable(axes)

    for (p, vals), pdf_ax in zip(samples.items(), axes):
        vals = [v for v in vals if not (type(v) is float and np.isnan(v))]
        if p == 'pooling_stride': vals = [0 if str(v).lower() == 'none' else v for v in vals]
        if len(set(type(v) for v in vals)) > 1:
            print(p, vals)
            raise NotImplementedError

        pdf_ax.set_title(p)
        pdf_ax.set_xlabel(p)
        pdf_ax.set_ylabel('Count of bucketed parameter value')

        cdf_ax = pdf_ax.twinx()
        cdf_ax.set_ylim(0, 1)
        cdf_ax.set_ylabel('CDF of parameter value')
        cdf_ax.grid(False)

        X = sorted(list(vals))
        if len(set(X)) < 100:
            # X might be oversampled.
            X = sorted(list(set(X)))
            Y = [len([x2 for x2 in vals if x2 == x]) for x in X]
            pdf_ax.bar(X, Y, alpha=0.5)
        else:
            pdf_ax.hist(vals, bins=50, alpha=0.5)

        cdfs = [i/len(X) for i, x in enumerate(X)]
        cdf_ax.plot(X, cdfs)

INT_PARAMS = [
  "batches_per_gradient",
  "batch_size",
  "in_dim",
  "hidden_size",
  "hidden_size_multiplier",
  "intermediate_size",
  "num_attention_heads",
  "num_hidden_layers",
  "gru_num_hidden",
  "gru_hidden_layer_size",
  "pooling_kernel_size",
  "epochs",
  "max_seq_len",
  "learning_rate_step",
]

def make_list_param(size, base, growth, type_fn=int, minimum=4):
    try: return [max(type_fn(base * (growth**i)), minimum) for i in range(int(size))]
    except:
        print(type(size), size, type(base), base, type(growth), growth)
        raise

def resolve(params):
    print(params)
    if 'modeltype' in params and type(params['modeltype']) is tuple:
        modeltype, model_specific_params = params['modeltype']
        params['modeltype'] = modeltype
        params.update(model_specific_params)

    params = {k: int(v) if k in INT_PARAMS else v for k, v in params.items()}
    

    # TODO(mmd): get rid of unnecessary projection.
    if 'hidden_size_multiplier' in params:
        #assert 'hidden_size' not in params, "Can only specify one of hidden_size_multiplier and hidden_size"
        hidden_size_multiplier = params['hidden_size_multiplier']
        params['hidden_size'] = int(params['num_attention_heads'] * hidden_size_multiplier)

        params.pop('hidden_size_multiplier')

    assert 'hidden_size' in params
    if params['modeltype'] == 'self_attention':
        assert params['hidden_size'] % params['num_attention_heads'] == 0
    elif params['modeltype'] == 'CNN':
        params['num_filters'] = make_list_param(
            params['num_conv_layers'], params['num_filters_base'], params['num_filters_growth_rate']
        )
        params['kernel_sizes'] = make_list_param(
            params['num_conv_layers'], params['kernel_size_base'], params['kernel_size_growth_rate']
        )
        params['fc_layer_sizes'] = make_list_param(
            params['num_fc_layers'], params['fc_layer_size_base'], params['fc_layer_growth_rate']
        )
        params['pooling_stride'] = None if params['pooling_stride']=='None' else int(params['pooling_stride'])

        for item in [
            'num_conv_layers', 'num_filters_base', 'num_filters_growth_rate', 'kernel_size_base',
            'kernel_size_growth_rate', 'num_fc_layers', 'fc_layer_size_base', 'fc_layer_growth_rate',
        ]:
            params.pop(item)
    elif params['modeltype'] == 'GRU':
        params['gru_fc_layer_sizes'] = make_list_param(
            params['gru_num_fc_layers'], params['gru_fc_layer_size_base'], params['gru_fc_layer_growth_rate']
        )
        for item in ['gru_num_fc_layers', 'gru_fc_layer_size_base', 'gru_fc_layer_growth_rate']:
            params.pop(item)


    # conv layers
    if ('conv_filt1' in params.keys()) or ('num_filt1' in params.keys()):
        params['filter_sizes'] = [
            int(params['conv_filt1']),  int(params['conv_filt2']), int(params['conv_filt3']),
            int(params['conv_filt4'])
        ]
        params['num_filters'] = [
            params['num_filt1'],  params['num_filt2'], params['num_filt3']
        ]

    return params

class ObjectiveFntr:
    def __init__(
        self, base_dir, rotation, constant_params, tqdm, single_task="", do_match_train_windows=True
    ):
        self.base_dir = base_dir
        self.rotation = rotation
        self.constant_params = copy.copy(constant_params)
        self.single_task = single_task
        self.do_match_train_windows = do_match_train_windows
        if self.single_task: assert self.single_task in ABLATION_GROUPS

        self.tqdm = tqdm

    @staticmethod
    def perf_metrics_to_trial_result(perf_metrics):
        if len(perf_metrics) == 2:
            perf_metrics_all_time, perf_metrics_25 = perf_metrics
            perf_metrics_till_discharge = None
        else:
            perf_metrics_all_time, perf_metrics_25, perf_metrics_till_discharge = perf_metrics

        assert not (
            (perf_metrics_all_time is None) and (perf_metrics_25 is None) and
            (perf_metrics_till_discharge is None)
        ), "Can't all be none!"

        # TODO(mmd): operationalize
        per_task_scores = {}
        rolling_multilabel_aucs = [
            'mort_24h', 'mort_48h', 'dnr_24h', 'dnr_48h', 'cmo_24h', 'cmo_48h',
        ]
        for task in rolling_multilabel_aucs:
            if not perf_metrics_all_time: continue
            per_task_scores[task] = perf_metrics_all_time[('tasks_binary_multilabel', 'AUROC (all)')][task]

        rolling_multiclass_aucs = ['disch_24h', 'disch_48h', 'rolling_fts']
        for task in rolling_multiclass_aucs:
            if not perf_metrics_all_time: continue
            per_task_scores.update({
                '%s - %s' % (task, k): v for k, v in perf_metrics_all_time[
                    (task, 'AUROC (ovr; all)')
                ].to_dict().items()
            })

        static_multilabel_aucs = [
            'icd_infection', 'icd_neoplasms', 'icd_endocrine', 'icd_blood', 'icd_mental', 'icd_nervous',
            'icd_circulatory', 'icd_respiratory', 'icd_digestive', 'icd_genitourinary', 'icd_pregnancy',
            'icd_skin', 'icd_musculoskeletal', 'icd_congenital', 'icd_perinatal', 'icd_ill_defined',
            'icd_injury', 'icd_unknown',
        ]
        for task in static_multilabel_aucs:
            if not perf_metrics_25: continue

            # TODO: Why this conditional?
            if task in per_task_scores.keys():
                per_task_scores[task] = perf_metrics_25[('tasks_binary_multilabel', 'AUROC (all)')][task]
            else:
                per_task_scores[task]=0.5

        evaluate_till_discharge_multilabel_aucs = ['Readmission 30']
        for task in evaluate_till_discharge_multilabel_aucs:
            if not perf_metrics_till_discharge: continue
            if task in per_task_scores.keys():
                per_task_scores[task] = perf_metrics_till_discharge[
                    ('tasks_binary_multilabel', 'AUROC (all)')
                ][task]
            else:
                per_task_scores[task]=0.5

        static_multiclass_aucs = ['Final Acuity Outcome']
        for task in static_multiclass_aucs:
            if not perf_metrics_25: continue
            per_task_scores.update({
                '%s - %s' % (task, k): v for k, v in perf_metrics_25[
                    (task, 'AUROC (ovr; all)')
                ].to_dict().items()
            })

        if perf_metrics_all_time:
            per_task_scores.update({
                'next_timepoint_was_measured - %s' % k[0]: v for k, v in \
                    perf_metrics_all_time[('next_timepoint_was_measured', 'AUROC (all)')].to_dict().items()
            })

            per_task_scores.update({
                'next_timepoint - %s' % k[0]: v for k, v in \
                    (2**(perf_metrics_all_time[('next_timepoint', 'R2 (all)')] - 1)).to_dict().items()
            })

        return per_task_scores

    def __call__(self, params):
        base_dir = self.base_dir
        args_dict = copy.copy(self.constant_params)
        rotation = self.rotation
        tqdm = self.tqdm

        args_dict.update(resolve(copy.deepcopy(params)))
        args_hash = hash_dict(args_dict)

        run_dir = os.path.join(base_dir, args_hash)
        if not os.path.isdir(run_dir): os.makedirs(run_dir)
        else: raise NotImplementedError("Shouldn't be colliding!")
        with open(os.path.join(run_dir, PARAMS_FILENAME), mode='wb') as f: pickle.dump(params, f)

        args_dict['run_dir'] = run_dir
        args_dict['do_overwrite'] = True
        args_dict['rotation'] = rotation

        if self.single_task:
            if 'regression_task_weight' in args_dict: del(args_dict['regression_task_weight'])
            if 'task_weights_filepath' in args_dict: del(args_dict['task_weights_filepath'])
            args_dict['ablate'] = [k for k in ABLATION_GROUPS.keys() if k != self.single_task]

            if self.do_match_train_windows:
                args_dict['set_to_eval_mode'] = EVAL_MODES_BY_ABLATION_GROUPS[self.single_task]

        args = Args(**args_dict)

        try:
            trained_meta_model = run_model.main(args, tqdm)

            with open(
                os.path.join(ROTATIONS_DIR, args.notes, str(rotation), 'tuning_dataset.pkl'), mode='rb'
            ) as f:
                tuning_dataset = pickle.load(f)

            _, task_info, perf_metrics = evaluate_multi(
                tuning_dataset, model=trained_meta_model, n_gpu=trained_meta_model.n_gpu,
                get_all_reprs = False, batch_size=2*args.batch_size, tqdm=tqdm, num_workers=28,
                args=args
            )

            with open(os.path.join(args.run_dir, 'tuning_task_info.pkl'), mode='wb') as f:
                pickle.dump(task_info, f)
            with open(os.path.join(args.run_dir, 'tuning_perf_metrics.pkl'), mode='wb') as f:
                pickle.dump(perf_metrics, f)

            with open(
                os.path.join(ROTATIONS_DIR, args.notes, str(rotation), 'test_dataset.pkl'), mode='rb'
            ) as f:
                test_dataset = pickle.load(f)

            _, test_task_info, test_perf_metrics = evaluate_multi(
                test_dataset, model=trained_meta_model, n_gpu=trained_meta_model.n_gpu,
                get_all_reprs = False, batch_size=2*args.batch_size, tqdm=tqdm, num_workers=28,
                args=args,
            )

            with open(os.path.join(args.run_dir, 'test_task_info.pkl'), mode='wb') as f:
                pickle.dump(test_task_info, f)
            with open(os.path.join(args.run_dir, 'test_perf_metrics.pkl'), mode='wb') as f:
                pickle.dump(test_perf_metrics, f)

            tuning_scores = -pd.Series(ObjectiveFntr.perf_metrics_to_trial_result(perf_metrics))
            test_scores = -pd.Series(ObjectiveFntr.perf_metrics_to_trial_result(test_perf_metrics))

            loss = tuning_scores.mean()
            loss_variance = tuning_scores.std()**2
            test_loss = test_scores.mean()
            test_loss_variance = test_scores.std()**2
            status = STATUS_OK
        except Exception as e:
            loss, test_loss = np.NaN, np.NaN
            loss_variance, test_loss_variance = np.NaN, np.NaN
            status = STATUS_FAIL
            with open(os.path.join(args.run_dir, 'error.pkl'), mode='wb') as f: pickle.dump(e, f)

            print("Errored on %s: %s" % (args.run_dir, e))

        return {
            'loss': loss,
            'loss_variance': loss_variance,
            'true_loss': test_loss,
            'true_loss_variance': test_loss_variance,
            'status': status,
        }

def main(hyperparameter_search_args, tqdm=tqdm, fmin_kwargs=None):
    if fmin_kwargs is None: fmin_kwargs = {}

    search_dir = hyperparameter_search_args.search_dir
    hyperparameter_search_args.to_json_file(os.path.join(search_dir, "hyperopt_args.json"))

    hyperopt_space, constant_params = read_config(search_dir)

    rotation = hyperparameter_search_args.rotation
    base_dir = os.path.join(search_dir, str(rotation))
    already_existed = os.path.exists(base_dir) and len(os.listdir(base_dir)) > 1
    if not os.path.isdir(base_dir): os.makedirs(base_dir)

    objective = ObjectiveFntr(
        base_dir, rotation, constant_params, tqdm,
        single_task=hyperparameter_search_args.single_task_search,
        do_match_train_windows=hyperparameter_search_args.do_match_train_windows
    )

    algo = HP_ALGS[hyperparameter_search_args.algo]

    if hyperparameter_search_args.do_use_mongo:
        mongo_addr = '{base}/{db}/jobs'.format(
            base=hyperparameter_search_args.mongo_addr, db=hyperparameter_search_args.mongo_db
        )
        print("Parallelizing search via Mongo DB: %s" % mongo_addr)
        trials = MongoTrials(mongo_addr, exp_key=hyperparameter_search_args.mongo_exp_key)
    elif already_existed:
        _, _, _, _, trials = read_or_recreate_trials(search_dir, tqdm=tqdm)
        trials = trials[str(rotation)]
    else:
        trials = Trials()

    best = fmin(
        objective,
        space=hyperopt_space,
        algo=algo,
        max_evals=hyperparameter_search_args.max_evals,
        trials=trials,
        **fmin_kwargs
    )

    return trials
