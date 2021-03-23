"""
fine_tune.py
Fine tunes a pre-trained model on a specific (single) task
"""

import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

import json, os, pickle, random
from copy import deepcopy
from tqdm import tqdm
import glob


# TODO: check these imports.
from ..utils import *
from ..constants import *
from ..data_utils import *
from ..BERT.model import *
from ..BERT.constants import *

from .fts_decoder import *
from .evaluator import *
from .meta_model import *
from .run_model import setup_datasets_and_dataloaders, args_run_setup, train_meta_model

def fine_tune_model(
    fine_tune_args, meta_model_args, sample_datum, binary_multilabel_keys, train_dataloaders_by_data_frac,
    tqdm=None, meta_model=None, tuning_dataloader=None
):
    print('in fine tune model')
    reloaded = (meta_model is not None)

    verbose = False
    if hasattr(fine_tune_args, 'verbose'):
        verbose = fine_tune_args.verbose

    if meta_model is None:
        meta_model = MetaModel(
            meta_model_args, sample_datum,
            class_names = {'tasks_binary_multilabel': binary_multilabel_keys},
            verbose = verbose,
        )

    if not(reloaded):
        reloaded, epoch = meta_model.load()

        if fine_tune_args.do_single_task: assert not reloaded, "Shouldn't be reloading a ST fine-tuning run!"
        else: assert reloaded, "Can't fine-tune a not-yet-trained model!"

        epoch=0
        reloaded=False

    # For fine-tuning, we want to ablate away everything *but* the target task.
    ablate = [k for k in ABLATION_GROUPS.keys() if k != fine_tune_args.fine_tune_task]
    meta_model.ablate(ablate, post_init=True)

    outputs = [meta_model]
    for data_frac, train_dataloader in train_dataloaders_by_data_frac.items():
        fine_tune_dir_name = fine_tune_args.fine_tune_task
        if data_frac != 1: fine_tune_dir_name += f"_{str(data_frac).replace('.', '-')}"

        fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)
        assert os.path.isdir(fine_tune_run_dir), f"{fine_tune_run_dir} must exist!"

        if fine_tune_args.do_frozen_representation:
            meta_model_FTD = deepcopy(meta_model)
            meta_model_FTD_args = deepcopy(meta_model_args)

            meta_model_FTD.run_dir = os.path.join(fine_tune_run_dir, "FTD")
            meta_model_FTD.freeze_representation()
            meta_model_FTD_args.run_dir = meta_model_FTD.run_dir

            if not os.path.isdir(meta_model_FTD.run_dir): os.makedirs(meta_model_FTD.run_dir)

            # Train it from scractch with the representation frozen and task_weights appropriately set.
            train_meta_model(
                meta_model_FTD, train_dataloader, meta_model_FTD_args, reloaded=reloaded, epoch=epoch,
                tuning_dataloader=tuning_dataloader,
                train_embedding_after=fine_tune_args.train_embedding_after
            )
            outputs.append(meta_model_FTD)
        if fine_tune_args.do_free_representation:
            meta_model_FTF = deepcopy(meta_model)
            meta_model_FTF_args = deepcopy(meta_model_args)

            meta_model_FTF.run_dir = os.path.join(fine_tune_run_dir, "FTF")
            meta_model_FTF_args.run_dir = meta_model_FTF.run_dir

            if not os.path.isdir(meta_model_FTF.run_dir): os.makedirs(meta_model_FTF.run_dir)

            # Train it from scractch with the representation frozen and task_weights appropriately set.
            train_meta_model(
                meta_model_FTF, train_dataloader, meta_model_FTF_args, reloaded=reloaded, epoch=epoch,
                tuning_dataloader=tuning_dataloader,
                train_embedding_after=fine_tune_args.train_embedding_after
            )
            outputs.append(meta_model_FTF)
    return outputs

def main(fine_tune_args, tqdm):
    assert os.path.isdir(fine_tune_args.run_dir), "Run dir must exist!"
    assert (
        fine_tune_args.do_frozen_representation or
        fine_tune_args.do_free_representation
    ), "Need to do either FTF or FTD!"

    fine_tune_args.to_json_file(os.path.join(fine_tune_args.run_dir, FINE_TUNE_ARGS_FILENAME))

    ablation_groups = EICU_ABLATION_GROUPS if fine_tune_args.do_eicu else ABLATION_GROUPS

    assert fine_tune_args.fine_tune_task in ablation_groups,\
        f"Invalid fine tune task: {fine_tune_args.fine_tune_task}"
    assert ((fine_tune_args.frac_fine_tune_data >0) and (fine_tune_args.frac_fine_tune_data<=1)),\
        "frac_fine_tune_data must be in the range(0, 1]"

    if fine_tune_args.do_masked_imputation_PT:
        meta_model_dir = os.path.dirname(fine_tune_args.run_dir)
        meta_model_args = Args.from_json_file(os.path.join(meta_model_dir, ARGS_FILENAME))
        assert meta_model_args.do_masked_imputation, "Expected PT to do masked imputation!"
        assert meta_model_args.imputation_mask_rate > 0, "Expected PT to do masked imputation!"

        meta_model_args.do_fake_masked_imputation_shape = True
        meta_model_args.do_masked_imputation = False
        meta_model_args.imputation_mask_rate = 0
    else:
        meta_model_args = Args.from_json_file(os.path.join(fine_tune_args.run_dir, ARGS_FILENAME))
    meta_model_args.set_to_eval_mode=""

    set_to_eval_mode = None
    if fine_tune_args.do_match_train_windows:
        meta_model_args.set_to_eval_mode = EVAL_MODES_BY_ABLATION_GROUPS[fine_tune_args.fine_tune_task]

    ablate = [k for k in ablation_groups if k != fine_tune_args.fine_tune_task]

    assert fine_tune_args.frac_fine_tune_data == 1, "frac_fine_tune_data is deprecated!"
    assert fine_tune_args.frac_female == 1, "frac_female is deprecated!"
    assert fine_tune_args.frac_black == 1, "frac_balck is deprecated!"

    data_fracs = [1]
    if fine_tune_args.do_small_data: data_fracs.extend(SMALL_DATA_FRACS)

    datasets, train_dataloader = setup_datasets_and_dataloaders(meta_model_args)

    orig_len=len(datasets['train'])

    # Just to be safe, here, we'll take copies of everything.
    orig_subjects = deepcopy(datasets['train'].orig_subjects)
    orig_max_hours = deepcopy(datasets['train'].orig_max_hours)
    orig_index = deepcopy(datasets['train'].index)
    subjects_hours = deepcopy(list(zip(orig_subjects, orig_max_hours)))

    assert len(set(subjects_hours))==len(subjects_hours)

    assert datasets['train'].max_seq_len == meta_model_args.max_seq_len
    assert train_dataloader.dataset.max_seq_len == meta_model_args.max_seq_len

    sample_datum = datasets['train'][0]
    binary_multilabel_keys = datasets['train'].get_binary_multilabel_keys()
    train_dataloaders_by_data_frac = {1: train_dataloader}

    random.seed(fine_tune_args.frac_fine_tune_data_seed)

    for frac in data_fracs:
        fine_tune_dir_name = fine_tune_args.fine_tune_task
        if frac != 1: fine_tune_dir_name += f"_{str(frac).replace('.', '-')}"

        fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)

        if not os.path.exists(fine_tune_run_dir): os.makedirs(fine_tune_run_dir)

        data_frac_seed = random.randint(0, int(1e10))
        with open(os.path.join(fine_tune_run_dir, 'data_frac_seed.txt'), mode='w') as f:
            f.write(str(data_frac_seed))

        random.seed(data_frac_seed)
        if frac != 1:
            frac_subjects_hours = random.choices(subjects_hours, k=int(frac * orig_len))
            frac_subjects, frac_hours = zip(*frac_subjects_hours)

            frac_dataset = deepcopy(datasets['train'])

            frac_dataset.orig_subjects = frac_subjects
            frac_dataset.orig_max_hours = frac_hours

            frac_dataset.reset_sequence_len(frac_dataset.sequence_len)

            new_index = frac_dataset.index
            frac_dataset.item_cache_remap = {
                i: next(j for j, ov in enumerate(orig_index) if ov == nv) for i, nv in enumerate(new_index)
            }
            frac_len = len(frac_dataset)
            assert frac_len < orig_len, f"{len(frac_dataset)} !< {orig_len}"

            with open(os.path.join(fine_tune_run_dir, 'item_cache_remap.json'), mode='w') as f:
                f.write(json.dumps(frac_dataset.item_cache_remap))
            with open(os.path.join(fine_tune_run_dir, 'frac_dataset.pkl'), mode='wb') as f:
                pickle.dump(frac_dataset, f)
            with open(os.path.join(fine_tune_run_dir, 'len_stats.json'), mode='w') as f:
                f.write(json.dumps({
                    'orig_len': orig_len,
                    'frac': frac,
                    'frac_len': frac_len,
                }))

            sampler = RandomSampler(frac_dataset)

            train_dataloader = DataLoader(
                frac_dataset, sampler=sampler,
                batch_size=train_dataloader.batch_size, num_workers=train_dataloader.num_workers
            )

            assert train_dataloader.dataset.max_seq_len == meta_model_args.max_seq_len
            train_dataloaders_by_data_frac[frac] = train_dataloader

        for (do, suffix) in [
            (fine_tune_args.do_frozen_representation, "FTD"), (fine_tune_args.do_free_representation, "FTF"),
        ]:
            if not do: continue

            # Really the data subsetting should go in these args, for reproducibility.
            fine_tune_meta_model_args = deepcopy(meta_model_args)
            fine_tune_meta_model_args.run_dir = os.path.join(fine_tune_run_dir, suffix)

            fine_tune_meta_model_args.ablate = ablate

            args_run_setup(fine_tune_meta_model_args)

    return fine_tune_model(
        fine_tune_args, meta_model_args, sample_datum, binary_multilabel_keys, train_dataloaders_by_data_frac,
        tqdm=tqdm, tuning_dataloader=None,
    )
