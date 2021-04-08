from typing import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, set_detect_anomaly
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.init as init
import torch.optim

from copy import deepcopy
import os, traceback, numpy as np, pandas as pd
idx = pd.IndexSlice

from . import run_model

from ..utils import *
from ..constants import *
from ..data_utils import *
from ..BERT.model import *
from ..BERT.constants import *

from .args import *
from .meta_model import *

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score, explained_variance_score, r2_score
from sklearn.preprocessing import LabelBinarizer

# TODO: be better
STATIC_TASKS  = ['Final Acuity Outcome']
ROLLING_TASKS = [
    'next_timepoint', 'disch_48h', 'disch_24h', 'next_timepoint_was_measured', 'rolling_ftseq',
    'tasks_binary_multilabel'
]
def get_manuscript_metrics_via_args(perf_metrics, args, **kwargs):
    return get_manuscript_metrics(
        perf_metrics, do_eicu=args.do_eicu, do_masked_imputation=args.do_masked_imputation, **kwargs
    )

def get_manuscript_metrics(
    perf_metrics, breakdown=None, do_eicu=False, do_masked_imputation=False, report_raw_r2=False,
    include_all_metrics=True, metric='AUROC'
):
    assert metric in ('AUROC', 'AUPRC'), f"{metric} invalid! Must be either AUROC or AUPRC"
    if breakdown is None:
        breakdown = EICU_MANUSCRIPT_BREAKDOWN if do_eicu else MANUSCRIPT_BREAKDOWN
        if do_masked_imputation:
            if include_all_metrics:
                breakdown = {
                    **breakdown,
                    **MASKED_IMPUTATION_BREAKDOWN,
                }
            else:
                breakdown = MASKED_IMPUTATION_BREAKDOWN

    if len(perf_metrics) == 2:
        all_time, first_24 = perf_metrics
        ends_at_discharge = None
    else: all_time, first_24, ends_at_discharge = perf_metrics
    vals = {}
    for out_k, k in breakdown.items():
        if type(k) is tuple:
            source = first_24 if k[1] == 'first_24' else all_time if k[1] == 'all_time' else ends_at_discharge
            if source is None:
                vals[out_k] = np.NaN
                continue

            matcher = (lambda s: s == k[2]) if type(k[2]) is str else k[2]

            vals_to_avg = source[(k[0], f'{metric} (all)')][[
                matcher(idx) for idx in source[(k[0], f'{metric} (all)')].index
            ]]
            vals[out_k] = vals_to_avg.mean()
        elif type(k) is list:
            try:
                vals[out_k] = np.mean([all_time[(k2, f'{metric} (ovr; macro)')] for k2 in k])
            except:
                vals[out_k] = np.NaN
        elif k in ('next_timepoint_was_measured', 'masked_imputation_classification'):
            vals[out_k] = all_time[(k, f'{metric} (mean)')]
        elif k in ('next_timepoint', 'masked_imputation_regression'):
            r2 = all_time[(k, 'R2 (mean)')]
            vals[out_k] = r2 if report_raw_r2 else 2**(r2 - 1)
        else:
            source = first_24 if k == 'Final Acuity Outcome' else all_time
            if k == 'rolling_ftseq' and (k, f'{metric} (ovr; macro)') not in source and \
                ('rolling_fts', f'{metric} (ovr; macro)') in source:
                    # Backwards compatability for old ftseq name (fts).
                    k = 'rolling_fts'
            try:
                vals[out_k] = source[(k, f'{metric} (ovr; macro)')]
            except:
                vals[out_k] = np.NaN
    return pd.Series(vals)

def nan_safe_regression_r2(true, score, multioutput):
    assert multioutput in ('raw_values', 'uniform_average', 'variance_weighted')
    assert true.shape == score.shape, "Shapes mismatched!"

    try:
        N_samples, N_columns = true.shape
    except:
        print(true, score)
        print(true.shape, score.shape)
        raise

    missing_idx = np.isnan(true).all(axis=0)
    output = np.zeros(N_columns)
    output[missing_idx] = np.NaN

    for col in range(N_columns):
        if missing_idx[col]: continue

        true_col, score_col = true[:, col], score[:, col]
        present_idx = ~np.isnan(true_col) & ~np.isnan(score_col)
        true_present, score_present = true_col[present_idx], score_col[present_idx]

        output[col] = r2_score(true_present, score_present)

    if multioutput == 'raw_values': return output
    elif multioutput == 'uniform_average': return np.average(output)
    elif multioutput == 'variance_weighted':
        variances = np.nanvar(true, axis=0)
        return np.average(output, weights=variances)

def get_performance_metrics(performance_dict, all_vocabs=None):
    output = {}

    print('perf: ', list(performance_dict.keys()))

    assert len(performance_dict) > 0, "Cannot find metrics for an empty performance dict."
    assert set(performance_dict.keys()).issubset({
        'disch_48h', 'next_timepoint', 'Final Acuity Outcome', 'disch_24h', 'tasks_binary_multilabel',
        'next_timepoint_was_measured', 'rolling_ftseq', 'masked_imputation',
        'masked_imputation', 'masked_imputation_regression', 'masked_imputation_classification'
    }), "We only handle these tasks at the moment."

    total_loss = np.array(list(loss for _, _, loss in performance_dict.values()))
    total_loss.sum(axis=1)
    output[("Sum", "Loss (mean)")] = total_loss.mean()
    output[("Sum", "Loss (std)")]  = total_loss.std()

    CONTINUOUS_TASKS = ('next_timepoint', 'masked_imputation_regression')
    MULTILABEL_TASKS = (
        'tasks_binary_multilabel', 'next_timepoint_was_measured', 'masked_imputation_classification'
    )
    MULTICLASS_TASKS = ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_ftseq')

    for task, (score, true, loss) in performance_dict.items():
        output[(task, 'Loss (mean)')] = loss.mean()
        output[(task, 'Loss (std)')] = loss.std()
        if task =='masked_imputation':
            continue

        if task in CONTINUOUS_TASKS:
            #output[(task, 'Explained Variance (mean)')] = explained_variance_score(true, score)
            #output[(task, 'Explained Variance (variance-weighted mean)')] = explained_variance_score(
            #    true, score, multioutput='variance_weighted'
            #)
            #output[(task, 'Explained Variance (all)')] = explained_variance_score(
            #    true, score, multioutput='raw_values'
            #)
            # handle no samples case (for validation set without missing tokens
            if score.shape[0]==0:
                output[(task, 'R2 (mean)')] = np.nan
                output[(task, 'R2 (variance-weighted mean)')] = np.nan
                output[(task, 'R2 (all)')] = np.nan
                continue

            output[(task, 'R2 (mean)')] = nan_safe_regression_r2(true, score, multioutput='uniform_average')
            output[(task, 'R2 (variance-weighted mean)')] = nan_safe_regression_r2(
                true, score, multioutput='variance_weighted'
            )
            output[(task, 'R2 (all)')] = nan_safe_regression_r2(true, score, multioutput='raw_values')
            continue
        elif task in MULTICLASS_TASKS:
            if task == 'rolling_ftseq':
                valid = np.expand_dims(true != 0, 1)
                valid_flat = np.reshape(valid, (-1,))
                # Score - (32, 9, 20)
                preds = np.argmax(score, axis=1)
                # preds - (32, 20)
                preds = preds[true != 0].flatten()
                # preds - (81,)
                true = true[true != 0].flatten()
                # true - (81, )
                preds = preds[~np.isnan(preds)]
                # preds - (81,)
                true  = true[~np.isnan(true)]
                # true - (81, )

                score_flat = score.swapaxes(1, 2).reshape((-1, 9))
                score_flat = score_flat[valid_flat, :]
                score = score_flat
            else:
                if score.shape[0]==0:
                    #for masked imputation when the validation set doesn't have missingness.
                    output[(task, "AUROC (ovr; all)")] = pd.Series(
                    [np.NaN] * len(labels_human_readable), index=labels_human_readable
                )
                    output[(task, "AUROC (ovr; micro)")] = np.NaN
                    output[(task, "AUROC (ovr; macro)")] = np.NaN
                preds = np.argmax(score, axis=1)

            if task == 'Final Acuity Outcome':
                vocab = all_vocabs['static_tasks']['Final Acuity Event']
            elif task == 'rolling_ftseq':
                #vocab = [0] + sorted(list(set(true))) # TODO(mmd)...
                vocab = list(range(score.shape[1]))
            else:
                # TODO(mmd): as a result of a misconfigured adapted model...
                assert true.max() < 18
                score = score[:, :18]
                vocab = all_vocabs['rolling_tasks']['Imminent Acuity Event'][task]


            output[(task, "Balanced Accuracy")] = balanced_accuracy_score(true, preds)
            output[(task, "Cohen's Kappa")] = cohen_kappa_score(true, preds)


            try:
                lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
                lb.fit(list(range(score.shape[1])))
                true_binarized = lb.transform(true)

            except Exception as e:
                print('Failed on %s' % task, len(vocab), len(valid_labels), vocab, valid_labels)
                print(e)
                traceback.print_exc()
                print(task)
                print('score shape in get_performance_metrics', score.shape, task)

                output[(task, "AUROC (ovr; all)")] = pd.Series(
                    [np.NaN] * len(labels_human_readable), index=labels_human_readable
                )
                output[(task, "AUROC (ovr; micro)")] = np.NaN
                output[(task, "AUROC (ovr; macro)")] = np.NaN
                continue

            label_cnts = np.array([len(set(true_binarized[:, i])) for i in range(true_binarized.shape[1])])
            valid_labels = (label_cnts > 1)

            assert len(valid_labels) == len(vocab), (
                "Vocab / valid labels mismatch! \n"
                "Task: %s\n"
                "Vocab: %s\n"
                "Valid Labels: %s"
                "" % (task, ', '.join(str(x) for x in vocab), ', '.join(str(x) for x in valid_labels))
            )

            try:
                true_binarized = true_binarized[:, valid_labels]
                score_binarized = score[:, valid_labels]
                labels_human_readable = list(np.array(vocab)[valid_labels])

                output[(task, "AUROC (ovr; all)")] = pd.Series(
                    roc_auc_score(true_binarized, score_binarized, average=None), index=labels_human_readable
                )
                output[(task, "AUROC (ovr; micro)")] = roc_auc_score(
                    true_binarized, score_binarized, average='micro'
                )
                output[(task, "AUROC (ovr; macro)")] = roc_auc_score(
                    true_binarized, score_binarized, average='macro'
                )

            except Exception as e:
                print('Failed on %s' % task, len(vocab), len(valid_labels), vocab, valid_labels)
                print(e)
#                 traceback.print_exc()
                output[(task, "AUROC (ovr; all)")] = pd.Series(
                    [np.NaN] * len(labels_human_readable), index=labels_human_readable
                )
                output[(task, "AUROC (ovr; micro)")] = np.NaN
                output[(task, "AUROC (ovr; macro)")] = np.NaN

            continue

        preds = np.where(score > 0, 1, 0)
        accuracy, cohens_kappa, auroc, auprc = [], [], [], []

        assert len(score.shape) >= 2, f"Score is malformed for {task}:\n{score.shape}\n{score}"
        for col in range(score.shape[1]):
            score_col, preds_col, true_col = score[:, col], preds[:, col], true[:, col]
            score_col, preds_col, true_col = [
                arr[~np.isnan(true_col)] for arr in (score_col, preds_col, true_col)
            ]
            score_col, preds_col, true_col = [
                arr[~np.isnan(score_col)] for arr in (score_col, preds_col, true_col)
            ]

            if len(true_col) == 0:
                for m in accuracy, cohens_kappa, auroc, auprc: m.append(np.NaN)
                continue

            try:
                true_vals = set(true_col)
                preds_dtype = preds_col.dtype
                preds_vals = set(preds_col)

                accuracy.append(accuracy_score(true_col, preds_col))
                cohens_kappa.append(cohen_kappa_score(true_col, preds_col))
            except ValueError as e:
                print(
                        f"Errored ({e}) on {task}, {col}.\n"
                        f"Shapes: score {score_col.shape}, preds {preds_col.shape}, true {true_col.shape}\n"
                        f"Vals: preds {preds_vals}, true {true_vals}."
                )
                raise

            if len(set(true_col)) == 1:
                for m in auroc, auprc: m.append(np.NaN)
                continue
            try:
                auroc.append(roc_auc_score(true_col, score_col))
                auprc.append(average_precision_score(true_col, score_col))
            except ValueError as e:
                print(
                        f"Errored ({e}) on {task}, {col}.\n"
                        f"Shapes: score {score_col.shape}, preds {preds_col.shape}, true {true_col.shape}\n"
                        f"Vals: preds {preds_vals}, true {true_vals}."
                )
                raise

        output[(task, 'Accuracy (all)')] = accuracy
        output[(task, "Cohen's Kappa (all)")] = cohens_kappa
        output[(task, 'AUROC (all)')] = auroc
        output[(task, 'AUPRC (all)')] = auprc

        output[(task, 'Accuracy (mean)')] = np.mean(accuracy)
        output[(task, "Cohen's Kappa (mean)")] = np.mean(cohens_kappa)
        output[(task, 'AUROC (mean)')] = np.mean(auroc)
        output[(task, 'AUPRC (mean)')] = np.mean(auprc)

        output[(task, 'Accuracy (std)')] = np.std(accuracy)
        output[(task, "Cohen's Kappa (std)")] = np.std(cohens_kappa)
        output[(task, 'AUROC (std)')] = np.std(auroc)
        output[(task, 'AUPRC (std)')] = np.std(auprc)

    return output

def get_model(dataset, model=None, model_rundir=None, n_gpu=0, epoch='latest', args=None):
    if model is None:
        assert model_rundir is not None and os.path.exists(model_rundir)
        args = Args.from_json_file(os.path.join(model_rundir, ARGS_FILENAME))


        dataset.set_to_eval_mode('all_time', 10)
        sample_datum = dataset[0]

        model = MetaModel(
            args, sample_datum,
            class_names = {'tasks_binary_multilabel': dataset.get_binary_multilabel_keys()}
        )


        loaded, epoch = model.load(epoch=epoch)
        assert loaded, f"Can't reload model from {model_rundir}"

    elif args is None and model_rundir is not None and os.path.isfile(os.path.join(model_rundir, ARGS_FILENAME)):
        args = Args.from_json_file(os.path.join(model_rundir, ARGS_FILENAME))
    model.ablate([], post_init=True) # ablate no tasks, but this is necessary to add 'mask_imputation' to the tasks by default in evaluation when args.do_masked_imputation==True
    return model, args

def evaluate_multi(
    dataset,
    model=None, model_rundir=None, epoch='latest', n_gpu=torch.cuda.device_count(),
    num_random_endpoints=10, evaluate_on_25=True, evaluate_till_discharge=True,
    batch_size=32, num_workers=4,
    get_all_reprs=False, tqdm=tqdm,
    args=None, do_debug_run=False,
):
    assert num_random_endpoints or evaluate_on_25, "Must do something..."

    if epoch!= 'latest':
        try: epoch=int(epoch)
        except: raise Exception("epoch must be an integer or 'latest'")

    model, args = get_model(
        dataset, model=model, model_rundir=model_rundir, n_gpu=n_gpu, epoch=epoch, args=args
    )
    model.eval()

    dataset.max_seq_len = args.max_seq_len

    # TODO: DRY up

    task_performances = []
    all_reprs = []

    def do_eval(dataset):
        all_reprs_in_fn, task_performance_in_fn = [], {}

        sampler = SubsetRandomSampler(list(range(200))) if do_debug_run else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
        drange = tqdm(dataloader, total=len(dataloader))
        batch_num = -1
        for batch in drange:
            batch_num += 1
            # raise NotImplementedError("Stopped here.")
            with torch.no_grad():
                hidden_states, pooled_output, all_outputs, _ = model.forward(batch)
                if get_all_reprs: all_reprs_in_fn.append(pooled_output.cpu().data.numpy())

                for task, (pred, true, loss) in all_outputs.items():
                    if task not in task_performance_in_fn:
                        task_performance_in_fn[task] = ([], [], [])
                    if task=='masked_imputation':
                        continue

                    pred_np, true_np = pred.cpu().data.numpy(), true.cpu().data.numpy()

                    task_performance_in_fn[task][0].append(pred_np)
                    task_performance_in_fn[task][1].append(true_np)
                    task_performance_in_fn[task][2].append(loss.mean().cpu().data.numpy())

                # add exception for continuous bert task
                if 'masked_imputation' in all_outputs.keys():
                    if 'masked_imputation' not in task_performance_in_fn:
                        task_performance_in_fn['masked_imputation'] = ([], [], [])
                    if 'masked_imputation_classification' not in task_performance_in_fn:
                        task_performance_in_fn['masked_imputation_classification'] = ([], [], [])
                    if 'masked_imputation_regression' not in task_performance_in_fn:
                        task_performance_in_fn['masked_imputation_regression'] = ([], [], [])

                    pred_shape=all_outputs['masked_imputation'][0][1].shape

                    pred_regression = all_outputs['masked_imputation'][0][1].cpu().data.numpy()

                    true_regression = all_outputs['masked_imputation'][1][1].cpu().data.numpy()
                    true_regression = np.where(
                        (
                            (
                                all_outputs['masked_imputation'][1][2].expand(-1, -1, pred_shape[2]) == 1
                            ).cpu().data.numpy()
                        ) & (
                            (all_outputs['masked_imputation'][1][0] == 1).cpu().data.numpy()
                        ),
                        true_regression,
                        np.NaN * np.ones_like(true_regression)
                    )

                    pred_timepoints, true_timepoints = [], []
                    for i in range(pred_regression.shape[1]):
                        pred_timepoints.append(pred_regression[:, i, :])
                        true_timepoints.append(true_regression[:, i, :])

                    pred_regression = np.concatenate(pred_timepoints)
                    true_regression = np.concatenate(true_timepoints)

                    task_performance_in_fn['masked_imputation_regression'][0].append(pred_regression)
                    task_performance_in_fn['masked_imputation_regression'][1].append(true_regression)


                    #pred_regression = all_outputs['masked_imputation'][0][1].reshape(-1)[
                    #    all_outputs['masked_imputation'][1][2].expand(-1, -1, pred_shape[2]).reshape(-1)==1
                    #]
                    #true_regression = all_outputs['masked_imputation'][1][1].reshape(-1)[
                    #    all_outputs['masked_imputation'][1][2].expand(-1, -1, pred_shape[2]).reshape(-1)==1
                    #]
                    loss_regression = all_outputs['masked_imputation'][2][0]

                    # regression

                    #task_performance_in_fn['masked_imputation_regression'][0].append(
                    #    pred_regression.cpu().data.numpy().reshape(-1)
                    #)
                    #task_performance_in_fn['masked_imputation_regression'][1].append(
                    #    true_regression.cpu().data.numpy().reshape(-1)
                    #)
                    task_performance_in_fn['masked_imputation_regression'][2].append(
                        loss_regression.mean().cpu().data.numpy()
                    )

                    # classification

                    pred_classification = all_outputs['masked_imputation'][0][0].reshape(-1, pred_shape[2])[
                        all_outputs['masked_imputation'][1][2].reshape(-1)==1
                    ]
                    true_classification = all_outputs['masked_imputation'][1][0].reshape(-1, pred_shape[2])[
                        all_outputs['masked_imputation'][1][2].reshape(-1)==1
                    ]

                    true_classification_np = true_classification.cpu().data.numpy()
                    bad_R, bad_C = np.where((true_classification_np != 0) & (true_classification_np != 1))
                    if len(bad_R) > 0:
                        print("\n\n\n", f"BAD IDX FOUND AT BATCH {batch_num}, batch size {batch_size}")
                        print("\n\n\n")

                    loss_classification = all_outputs['masked_imputation'][2][1]

                    task_performance_in_fn['masked_imputation_classification'][0].append(
                        pred_classification.cpu().data.numpy()
                    )
                    task_performance_in_fn['masked_imputation_classification'][1].append(
                        true_classification_np
                    )
                    task_performance_in_fn['masked_imputation_classification'][2].append(
                        loss_classification.mean().cpu().data.numpy()
                    )

                    # joined loss
                    loss = all_outputs['masked_imputation'][2][2]
                    task_performance_in_fn['masked_imputation'][0].append(
                        pred_classification.cpu().data.numpy().reshape(-1)
                    )
                    task_performance_in_fn['masked_imputation'][1].append(
                        true_classification.cpu().data.numpy().reshape(-1)
                    ) #this is just needed to make code work, but isn't used in evaluation
                    task_performance_in_fn['masked_imputation'][2].append(loss.cpu().data.numpy())

        all_reprs.append((all_reprs_in_fn, dataset.index))
        task_performances.append(task_performance_in_fn)

    if num_random_endpoints > 0:
        print("Evaluating over %d random time-points per patient." % num_random_endpoints)

        dataset.set_to_eval_mode('all_time', num_random_endpoints)
        do_eval(dataset)

    if evaluate_on_25:
        print("Evaluating over the first <= 24 hours.")

        dataset.set_to_eval_mode('first_24')
        do_eval(dataset)

    if evaluate_till_discharge:
        print("Evaluating over the final %d hours." % args.max_seq_len)

        dataset.set_to_eval_mode('extend_till_discharge')
        do_eval(dataset)

    out_perf_metrics = []
    out_task_performances = []

    for task_performance in task_performances:
#         print('task_performances: ',list(task_performance.keys()))
        task_performance = {
            k: (
                np.concatenate(p), np.concatenate(t), np.hstack(l)
            ) for k, (p, t, l) in task_performance.items()
        }

        #for k, (p, t, l) in task_performance.items(): print(k, p.shape, t.shape, l.shape)

        binary_multilabel_tasks = dataset.get_binary_multilabel_keys()
        next_timepoint_was_measured_tasks = dataset.dfs['next_timepoint_was_measured'].columns
        next_timepoint_tasks = [c for c in dataset.dfs['next_timepoint'].columns if c[1] == 'mean']

        try:
            perf_metrics = get_performance_metrics(task_performance, all_vocabs=dataset.all_vocabs)

            for k, v in perf_metrics.items():
                if k[0] == 'next_timepoint' and k[1].endswith('all)'):
                    perf_metrics[k] = pd.Series(perf_metrics[k], index=next_timepoint_tasks)
                elif k[0] == 'next_timepoint_was_measured' and k[1].endswith('all)'):
                    perf_metrics[k] = pd.Series(v, index=next_timepoint_was_measured_tasks)
                elif k[0] == 'tasks_binary_multilabel' and k[1].endswith('all)'):
                    perf_metrics[k] = pd.Series(v, index=binary_multilabel_tasks)
            out_perf_metrics.append(perf_metrics)
            out_task_performances.append(task_performance)
        except Exception as e:
            print("Couldn't make perf_dict:", e)
            print(binary_multilabel_tasks, next_timepoint_was_measured_tasks, next_timepoint_tasks)
            traceback.print_exc()

            out_task_performances.append(task_performance)
            out_perf_metrics.append(None)

    return all_reprs, out_task_performances, out_perf_metrics

def main(args, tqdm=None, datasets=None, model=None):
    assert os.path.isdir(args.run_dir), "Run dir %s not found" % args.run_dir
    args.to_json_file(os.path.join(args.run_dir, EVAL_ARGS_FILENAME))

    if datasets is None: datasets = run_model.load_datasets(args, use_dataset_shells=False)
    else:
        for do, split in (
            (args.do_eval_train, 'train'), (args.do_eval_tuning, 'tuning'), (args.do_eval_test, 'test')
        ):
            if do: assert split in datasets, f"Missing {split} dataset %s!"

    templates = deepcopy(EVAL_FILE_TEMPLATES)
    if not args.do_save_all_reprs: templates = templates[1:]

    if not(hasattr(args, 'eval_type')): args.eval_type=None

    out = {}
    for name, dataset in datasets.items():
        if dataset is None: continue

        if args.eval_type=='female':
            fps = [os.path.join(args.run_dir, 'female_'+t % name) for t in templates]
        elif args.eval_type == 'male':
            fps = [os.path.join(args.run_dir, 'male_'+t % name) for t in templates]
        elif args.eval_type=='white':
            fps = [os.path.join(args.run_dir, 'white_'+t % name) for t in templates]
        elif args.eval_type == 'black':
            fps = [os.path.join(args.run_dir, 'black_'+t % name) for t in templates]
        else:
            fps = [os.path.join(args.run_dir, t % name) for t in templates]

        if args.eval_epoch != 'latest':
            fps = [fp.replace('.pkl', f'_{args.eval_epoch}.pkl') for fp in fps]

        if not args.do_overwrite and all(os.path.isfile(fp) for fp in fps):
            out[name] = [depickle(fp) for fp in fps]
            continue

        print("Processing %s for %s" % (name, args.run_dir))

        # evaluate just for females/males or all
        if args.eval_type in ['female', 'male', 'white', 'black']:
            # get female and male participants.
            subjects = dataset.orig_subjects
            if args.eval_type == 'male':
                males = dataset.dfs['statics'].loc[idx[subjects], 'gender_2'].values
                assert len(set(subjects))==len(males)
                male_subjects = [item[0] for item in zip(subjects, males) if item[1]==1]
                subjects=male_subjects
            elif args.eval_type =='female':
                females = dataset.dfs['statics'].loc[idx[subjects], 'gender_1'].values
                assert len(set(subjects))==len(females)
                female_subjects = [item[0] for item in zip(subjects, females) if item[1]==1]
                subjects=female_subjects
            elif args.eval_type =='white':
                white = dataset.dfs['statics'].loc[idx[subjects], 'ethnicity_4'].values
                assert len(set(subjects))==len(white)
                white_subjects = [item[0] for item in zip(subjects, white) if item[1]==1]
                subjects=white_subjects
            elif args.eval_type =='black':
                black = dataset.dfs['statics'].loc[idx[subjects], 'ethnicity_2'].values
                assert len(set(subjects))==len(black)
                black_subjects = [item[0] for item in zip(subjects, black) if item[1]==1]
                subjects=black_subjects

            orig_len=len(dataset)
            subjects_hours = list(zip(dataset.orig_subjects, dataset.orig_max_hours))
            subjects_hours = [item for item in subjects_hours if item[0] in subjects]
            subjects, hours =zip(*subjects_hours)
            dataset.orig_subjects = subjects
            dataset.orig_max_hours = hours
            dataset.reset_sequence_len(dataset.sequence_len, reset_index=False)
            dataset.reset_index()
            assert len(dataset) < orig_len, f"Failed to assert that {len(datasets['train'])} < {orig_len}"
            print('Successfully restricted dataset to evaluate on only '+args.eval_type)



        out[name] = evaluate_multi(
            dataset, model=model, model_rundir=args.run_dir, num_workers=args.num_dataloader_workers,
            get_all_reprs=args.do_save_all_reprs, tqdm=tqdm, do_debug_run=args.do_debug_run,
            epoch=args.eval_epoch, num_random_endpoints=args.num_random_endpoints,
        )

        to_save_l = out[name] if args.do_save_all_reprs else out[name][1:]
        for obj, fp in zip(to_save_l, fps):
            with open(fp, mode='wb') as f: pickle.dump(obj, f)

    return out
