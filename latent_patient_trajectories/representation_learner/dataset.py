"""
Representation Learner Dataset
"""
import pdb
import collections, copy, os, random, sys, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
from typing import Callable, Dict, Sequence, ClassVar, Set
idx=pd.IndexSlice

from ..constants import *
from .utils import *
from ..utils import *
from .extractors import *
from ..data_utils import prepare_continuous_labels, tokenize_notes
from ..BERT.constants import *
from ..BERT.data_processor import *
from pytorch_pretrained_bert.tokenization import BertTokenizer

import time
import random


class ImputationTypes(enum.Enum):
    ZERO   = enum.auto()
    BFILL  = enum.auto()
    FFILL  = enum.auto()
    LINEAR = enum.auto()

def bfill(df): return df.fillna(method='bfill')
def ffill(df): return df.fillna(method='ffill')
def zero(df): return df.fillna(value=0)
def interpolate(df): return df.interpolate(method='linear')
IMPUTATION_FUNCTIONS = {
    ImputationTypes.BFILL: bfill,
    ImputationTypes.FFILL: ffill,
    ImputationTypes.ZERO: zero,
    ImputationTypes.LINEAR: interpolate,
}

# TODO: Need a notes extractor.
class DatasetMaker():
    FEATURES_TO_EMBED = (
        FeatureTypes.CATEGORICAL, FeatureTypes.ORDINAL, FeatureTypes.ORDINAL_CYCLICAL
    )

    """ TODO: Store the incremental parts of this to disk. To do this, define a hashable subclass and make
        this and all extractors subclass from it.
        Just running all the extractors takes 39.8 s ± 1.14 s for 1 fold.
    """
    def __init__(
        self,
        rolling_tasks = {
            'Imminent Acuity Event':     RollingAcuityEventsExtractor(),
        },
        rolling_ftseq = RollingFTSExtractor(), # TODO(mmd): Breaking abstraction here!
        static_tasks  = {
            'Final Acuity Event': StaticAcuityOutcomeExtractor(),
            'Long LOS':           StaticLongLOSExtractor(),
            'Codes':              CodesExtractor(),
            'Readmission 30':              ReadmissionExtractor(),
        },
        timeseries_featurizers = {
            'DNR/CMO labels': EOLCareTSFeaturizer(),
            'Labs':           LabsFeaturizer(),
            'Treatments':     TreatmentStatusExtractor(),
        },
        static_featurizers = {
            'Demographics': DemographicFeaturizer(),
        },
        seed           = 1,
        dataset_params = {},
        integrate_note_bert = False,
        notes_file = os.path.join(DATA_DIR, 'topics.hdf'),
        notes_key = ''
    ):
        self.is_fit = False
        self.all_extractors = [] if rolling_ftseq is None else [rolling_ftseq]

        for d in (rolling_tasks, static_tasks, timeseries_featurizers, static_featurizers):
            self.all_extractors.extend(d.values())

        self.seed = seed
        self.__seed()

        self.rolling_tasks          = rolling_tasks
        self.rolling_ftseq          = rolling_ftseq
        self.static_tasks           = static_tasks
        self.timeseries_featurizers = timeseries_featurizers
        self.static_featurizers     = static_featurizers
        self.dataset_params         = dataset_params
        self.integrate_note_bert    = integrate_note_bert
        self.notes_file             = notes_file
        self.notes_key              = notes_key
        self.save_data_only         = False

    def __seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        for extractor in self.all_extractors: extractor.set_seed(self.seed)

    def __filter_to_folds(self, dfs, folds={}):
        assert len(folds) > 0, "Must process something!"
        return [
            df[df.index.get_level_values(FOLD_IDX_LVL).isin(folds)] if df is not None else df for df in dfs
        ]

    def fit(self, *dfs, folds={}):
        """ gets means and stds for scaling and fits extractors."""
        self.__seed()
        dfs = self.__filter_to_folds(dfs, folds)

        # print(dfs.keys())

        for e in self.all_extractors:
            try:
                if not e.is_fit: e.fit(*dfs)
            except:
                print("Failed on %s" % str(e))
                raise


        self.is_fit = True

    def __join_extractors(self, dfs, extractors, extract_continuous=True, extract_embeddables=True):
        out_continuous, out_to_embed = [], []
        for key, extractor in extractors.items():
            #print(key)
            to_embed_cols = [c for c, t in extractor.columns.items() if t in self.FEATURES_TO_EMBED]
            continuous_cols = [c for c, t in extractor.columns.items() if t not in self.FEATURES_TO_EMBED]
            # print(to_embed_cols)
            # print(continuous_cols)
            output = extractor.extract(*dfs)

            try:
                if extract_continuous: out_continuous.append(output[continuous_cols])
                if extract_embeddables: out_to_embed.append(output[to_embed_cols])
            except KeyError as e:
                print("Can't extract column that should be present!", e)
                print("output schema:")
                print(output.head())
                print(output.columns)
                print(output.index.names)
                raise

        # print(len(out_continuous))

        # if len(out_continuous)>0:
        #     for item in out_continuous:
        #         print(item.head(5))

        out_continuous = out_continuous[0].join(out_continuous[1:]) if len(out_continuous) > 0 else None
        out_to_embed = out_to_embed[0].join(out_to_embed[1:]) if len(out_to_embed) > 0 else None

        return out_continuous, out_to_embed

    def __scale(self, df, mean, std):
        if df.shape[1] > 0: return (df - mean)/std

    def process(self, *dfs, folds={}, eicu=False):
        """
        Runtime is about 1 minute
        Returns
            (dict) containing:
                rolling_ftseq (): The Future treatment sequence for each patient at each timepoint
                rolling_tasks_continuous (): The continuous tasks that change throughout the patients' stays
                rolling_tasks_to_embed ():
                static_tasks_continuous ():
                static_tasks_to_embed ():
                ts_continuous ():
                ts_to_embed ():
                statics_continuous ():
                statics_to_embed ():
                notes ():
                max_hours_map ():
                vocab ():
                max_time_since_measured ():
                seed (int):
        """
        assert self.is_fit, "Must be fit!"
        dfs = self.__filter_to_folds(dfs, folds)

        rolling_tasks_continuous, rolling_tasks_to_embed = self.__join_extractors(dfs, self.rolling_tasks)
        static_tasks_continuous, static_tasks_to_embed = self.__join_extractors(dfs, self.static_tasks)
        ts_continuous, ts_to_embed = self.__join_extractors(dfs, self.timeseries_featurizers)
        statics_continuous, statics_to_embed = self.__join_extractors(dfs, self.static_featurizers) # problem here
        rolling_ftseq = None if self.rolling_ftseq is None else self.rolling_ftseq.extract(*dfs)

        max_hours_df = ts_continuous[[]].reset_index()[['hours_in', 'icustay_id']] # TODO(mmd): constantify
        max_hours_df.set_index('icustay_id', inplace=True)
        max_hours_df = max_hours_df.groupby('icustay_id').max()
        max_hours_dict = max_hours_df['hours_in'].to_dict()

        notes = None

        def gv(v):
            try: return v.vocab
            except AttributeError as e:
                return {}

        if eicu:
            vocab={
                **self.static_featurizers['Demographics'].vocab,
            }
        else:
            vocab={
                **self.static_featurizers['Demographics'].vocab,
                **self.timeseries_featurizers['DNR/CMO labels'].vocab,
            }

        return dict(#PatientDataset
            rolling_ftseq=rolling_ftseq,
            rolling_tasks_continuous = rolling_tasks_continuous,
            rolling_tasks_to_embed = rolling_tasks_to_embed,
            static_tasks_continuous = static_tasks_continuous,
            static_tasks_to_embed = static_tasks_to_embed,
            ts_continuous = ts_continuous,
            ts_to_embed = ts_to_embed,
            statics_continuous = statics_continuous,
            statics_to_embed = statics_to_embed,
            notes=notes,
            max_hours_map=max_hours_dict,
            vocab = vocab,
            max_time_since_measured=self.timeseries_featurizers['Labs'].max_time_since_measured,
            all_vocabs={
                n: {k: gv(v) for k, v in vv.items()}
                for n, vv in [
                    ('static_featurizers', self.static_featurizers),
                    ('timeseries_featurizers', self.timeseries_featurizers),
                    ('static_tasks', self.static_tasks),
                    ('rolling_tasks', self.rolling_tasks),
                ]
            },
            seed=self.seed,
            **self.dataset_params,
        )

class PatientDataset(Dataset):
    def __init__(
        self,
        # These params are constructed explicitly by the dataset maker.
        rolling_ftseq:              pd.DataFrame,
        rolling_tasks_continuous: pd.DataFrame,
        rolling_tasks_to_embed:   pd.DataFrame,
        static_tasks_continuous:  pd.DataFrame,
        static_tasks_to_embed:    pd.DataFrame,
        ts_continuous:            pd.DataFrame,
        ts_to_embed:              pd.DataFrame,
        statics_continuous:       pd.DataFrame,
        statics_to_embed:         pd.DataFrame,
        notes:                    pd.DataFrame,
        max_hours_map:            Dict[int, int],
        seed:                     int = 1,
        vocab:                    Dict = {},
        all_vocabs: Dict={},
        max_time_since_measured:  int = 8,

        # These params can be provided by the DatasetMaker through dataset_params, but are not constructed
        # explicitly:

        # Pad all sequences to this length. This enables batching without a custom collate_fn.
        # TODO: write a custom collate_fn.
        max_seq_len:   int = 480, # 20 days.
        max_note_len: int = 512,

        # The minimum possible sequence length. Patients with fewer than this many continuous features are
        # omitted.
        min_seq_len:   int = 24,

        # How to impute the continuous timeseries features.
        imputation_method: ImputationTypes = ImputationTypes.LINEAR,

        # Evaluation-time Params:
        # Use a specific sequence length (rather than a random selection).
        sequence_length:        int = None,
        do_all_timepoints:     bool = False,
        num_random_endpoints:   int = 0,
        extend_till_discharge: bool = False,

        imputation_mask_rate: float = 0,

        reload_self_dir: str = "",
    ):
        """
        dataset maker is in
        """
        self.all_vocabs = all_vocabs
        self.max_note_len=max_note_len
        # Input Validation:
        assert not (do_all_timepoints and sequence_length is not None), \
            "Can't do all timepoints (passed %s) and a specific sequence length (passed %s)" % (
                str(do_all_timepoints), str(sequence_length)
            )
        assert not (do_all_timepoints and num_random_endpoints > 0), \
            "Can't do all timepoints and only a subset."
        assert not (sequence_length is not None and num_random_endpoints > 0), \
            "Can't do a specific sequence and a subset of sequences."
        assert not ts_to_embed.isnull().any().any(), \
            "Embeddable timeseries features should have no missingness!"
        assert not statics_to_embed.isnull().any().any(), \
            "Embeddable static features should have no missingness!"
        if rolling_ftseq is not None:
            assert not rolling_ftseq.isnull().any().any(), "Rolling FTS should have no missingness!"

        ts_to_embed = ts_to_embed.astype(np.int32)
        self.ts_continuous_cols = ts_continuous.columns

        ts_to_embed_vocab = {
            k: v for k, v in vocab.items() if k in ('DNR Ordered', 'Comfort Measures Ordered')
        }
        for c in ts_to_embed.columns:
            if type(c) is tuple and c[1] == 'time_since_measured':
                ts_to_embed_vocab[c] = ['%d hours' % x for x in range(max_time_since_measured+1)]

        # important columns
        important_columns = [
            f"{col}_{i}" for i in range(max_time_since_measured+1) for col in ts_to_embed if not(
                (col in ['DNR Ordered', 'Comfort Measures Ordered']) and (i>1)
            )
        ]
        # one hot encode using pandas
        ts_to_embed_one_hot=pd.get_dummies(ts_to_embed.astype(str))
        #assert len(important_columns) == len(ts_to_embed_one_hot.columns.tolist()), (
        #    f"Columns mismatch!\n"
        #    f"IC - TSTE: {set(important_columns) - set(ts_to_embed_one_hot.columns.tolist())}\n"
        #    f"TSTE - IC: {set(ts_to_embed_one_hot.columns.tolist()) - set(important_columns)}"
        #)

        # ts_to_embed_one_hot = one_hot_encode(ts_to_embed.columns, ts_to_embed, ts_to_embed_vocab)

        ts = pd.concat((ts_continuous, ts_to_embed_one_hot), axis=1)

        statics_to_embed = statics_to_embed.astype(np.int32)
        statics_to_embed_one_hot=pd.get_dummies(statics_to_embed.astype(str))
        # make assertions about statics_to_embed_one_hot

        statics = pd.concat((statics_continuous, statics_to_embed_one_hot), axis=1)

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_LOCATION, do_lower_case=False)
        notes = tokenize_notes(notes, self.tokenizer)

        # Do stuff...
        # TODO(mmd): consistent naming.
        dfs = [
            ('rolling_ftseq', rolling_ftseq),
            ('rolling_tasks_binary_multilabel', rolling_tasks_continuous),
            ('rolling_tasks_multiclass', rolling_tasks_to_embed),
            ('static_tasks_binary_multilabel', static_tasks_continuous),
            ('static_tasks_multiclass', static_tasks_to_embed),
            ('ts', ts),
            ('statics', statics),
        ]

        self.using_integrated_notes = False
        self.using_pretrained_notes = False

        # if not_none(notes): # this is v confusing
        if notes is not None:
            dfs.append(('notes', notes))
            if 'text' in notes:
                self.using_integrated_notes = True
            else:
                self.using_pretrained_notes = True
                self.notes_cols = notes.columns

        dfs  = [(k, df) for k, df in dfs if df is not None and df.shape[1] > 0]

        # We use this awkward, roundabout construction to preserve key ordering.
        self.dfs  = {k: df for k, df in dfs}
        self.keys = [k for k, df in dfs]

        # TODO(mmd): This is entirely antithetical to the whole design principle here... Fix it.
        self.multiclass_sizes = {}
        for k in ('rolling_tasks_multiclass', 'static_tasks_multiclass'):
            df = self.dfs[k]
            for c in df.columns:
                assert c not in self.multiclass_sizes, "Collision!"
                self.multiclass_sizes[c] = df[c].max()

        # So now we're going to modify a bunch of these dataframes ad-hoc even though ostensibly we'd like
        # this to be more general, but...

        # We also add some keys during processing:
#         drop_cols=[col for col in sorted(self.dfs['ts'].columns.tolist()) if 'measured' in col]
#         print(drop_cols)
#         self.dfs['ts'].drop(drop_cols, inplace=True)
        #self.dfs['ts'].drop([col for col in set(self.dfs['ts'].columns.tolist()) if 'measured' in col], inplace=True)
        self.dfs['next_timepoint'] = self.dfs['ts'].copy()
        # drop cols that have measured and time_since
        self.keys.append('next_timepoint')

        self.dfs['next_timepoint_was_measured'] = (ts_to_embed == 0).astype(float)
        if 'DNR Ordered' in self.dfs['next_timepoint_was_measured'].columns:
            self.dfs['next_timepoint_was_measured'].drop(
                columns=['DNR Ordered', 'Comfort Measures Ordered'], inplace=True
            )
        self.keys.append('next_timepoint_was_measured')


        self.keys.append('ts_mask')

        self.orig_max_seq_len  = min(max_seq_len, max(max_hours_map.values()))
        self.orig_min_seq_len  = min_seq_len
        self.orig_subjects = sorted(subj for subj, hrs_in in max_hours_map.items())
        self.orig_max_hours = [max_hours_map[subj] for subj in self.orig_subjects]

        self.subjects = [s for s in self.orig_subjects]

        self.seed        = seed
        self.impute_fn   = IMPUTATION_FUNCTIONS[imputation_method]
        self.max_seq_len = min(max_seq_len, max(max_hours_map.values()))

        self.min_seq_len = min_seq_len

        self.do_all_timepoints     = do_all_timepoints
        self.num_random_endpoints  = num_random_endpoints
        self.extend_till_discharge = extend_till_discharge

        self.imputation_mask_rate = imputation_mask_rate

        self.reset_sequence_len(sequence_length)

        self.binary_multilabel_task_concat_order = [
            'rolling_tasks_binary_multilabel', 'static_tasks_binary_multilabel'
        ]

        self.__seed()

        self.train_tune_test='train'
        self.epoch=0
        self.save_place=''
        self.reload_self_dir=reload_self_dir

    def reset_sequence_len(self, new_sequence_len, reset_index=True):
        self.sequence_len = new_sequence_len
        if self.sequence_len:
            assert not self.do_all_timepoints
            assert not self.extend_till_discharge
            assert self.sequence_len > self.orig_min_seq_len and self.sequence_len < self.orig_max_seq_len
            self.min_seq_len = self.sequence_len - 1
            self.max_seq_len = self.sequence_len

        max_hours_map = {subj: hrs_in for subj, hrs_in in zip(self.orig_subjects, self.orig_max_hours)}
        self.subjects = sorted(subj for subj, hrs_in in max_hours_map.items() if hrs_in > self.min_seq_len)
        self.max_hours = [max_hours_map[subj] for subj in self.subjects]

        if reset_index: self.reset_index()

    def reset_index(self):
        self.index = []
        if self.do_all_timepoints:
            for subject, max_hour in zip(self.subjects, self.max_hours):
                self.index.extend([(subject, hr) for hr in range(self.orig_min_seq_len, max_hour)])
        elif self.num_random_endpoints:
            for subject, max_hour in zip(self.subjects, self.max_hours):
                possible_hours = list(range(self.orig_min_seq_len, max_hour))
                if len(possible_hours) >= self.num_random_endpoints:
                    random_endpoints = np.random.choice(
                        possible_hours, self.num_random_endpoints, replace=False
                    )
                else: random_endpoints = possible_hours

                self.index.extend([(subject, hr) for hr in random_endpoints])
        elif self.extend_till_discharge:
            self.index.extend(list(zip(self.subjects, self.max_hours)))
        else:
            self.index = self.subjects

    def set_to_eval_mode(self, eval_mode, num_random_endpoints=1):
        """
        Sets the dataset to operate in one of the foundational evaluation modes--either, first 24 hours,
        all time (epitomized through N random selections per patients, usually 1 for training and 10 for
        eval), or extend_till_discharge.
        """
        assert eval_mode in EVAL_MODES, \
            "Invalid eval_mode: %s. Must be in %s" % (eval_mode, ', '.join(EVAL_MODES))

        # Only used for tracking
        self.eval_mode = eval_mode

        # Constant changes regardless of eval_mode
        self.do_all_timepoints = False
        self.num_random_endpoints = 0
        self.sequence_len = None
        self.extend_till_discharge = False

        if eval_mode == 'all_time': self.num_random_endpoints = num_random_endpoints
        elif eval_mode == 'first_24': self.sequence_len = 25 # One extra to get full first day
        elif eval_mode == 'extend_till_discharge': self.extend_till_discharge = True

        self.reset_index()

    def assert_has_attr(self, attr):
        assert hasattr(self, attr), f"Must initialize dataset with {attr}"

    def get_save_path(self, epoch=None, item=None):
        if hasattr(self, 'skip_cache') and self.skip_cache: return ""
        if item is not None and hasattr(self, 'item_cache_remap') and self.item_cache_remap:
            # We use this in the context of small_data_runs, to ensure that we're actually pulling the
            # records corresponding to the small data samples we select.
            item = self.item_cache_remap[item]

        for attr in ("max_seq_len", "train_tune_test", "save_place"): self.assert_has_attr(attr)
        if epoch is None:
            self.assert_has_attr('epoch')
            epoch = self.epoch

        save_dir = os.path.join(self.save_place, f'msl_{self.max_seq_len}_{self.train_tune_test}')
        if item is None:
            filename = f"epoch-{epoch}.pkl"
        else:
            filename = f"item-{item}.pt"
            save_dir = os.path.join(save_dir, f'epoch-{self.epoch}')

        if hasattr(self, 'eval_mode'):
            save_dir = os.path.join(save_dir, self.eval_mode)
        else:
            assert self.train_tune_test == 'train',\
                "If the dataset is in tuning or test mode, it should have an eval_mode!"

        return os.path.join(save_dir, filename)

    def load_save_path(self, epoch=None, item=None):
        if hasattr(self, 'skip_cache') and self.skip_cache: return False, None

        save_path = self.get_save_path(epoch=epoch, item=item)
        if os.path.isfile(save_path):
            try: return True, torch.load(save_path)
            except RuntimeError as e:
                print(save_path, e)
            except EOFError as e:
                print(save_path, e)

        return False, None

    def save_item(self, tensors, item, n_attempts=10):
        if hasattr(self, 'skip_cache') and self.skip_cache: return

        attempt = 0
        while attempt < n_attempts:
            save_path = self.get_save_path(item=item)
            save_dir = os.path.dirname(save_path)
            try:
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                torch.save({k: v.half() if v.dtype==torch.float32 else v for k, v in tensors.items()}, save_path)
                return
            except FileExistsError as e:
                print(f"Couldn't save {save_path}: {e}. Trying again.")
                attempt += 1

    def set_epoch(self, epoch, load_cache=True):
        assert epoch >= 0 and type(epoch) is int, f"{epoch} is invalid, must be non-negative integer!"

        self.epoch_cached = False
        self.epoch = epoch

        if load_cache: self.epoch_cached, self.cached_epoch = self.load_save_path()
        else: self.epoch_cached, self.cached_epoch = False, None

        self.__seed()

    def set_binary_multilabel_keys(self):
        self.binary_multilabel_keys = self.get_binary_multilabel_keys()

    def get_binary_multilabel_keys(self):
        if hasattr(self, "binary_multilabel_keys"): return self.binary_multilabel_keys

        out = []
        for key in self.binary_multilabel_task_concat_order: out.extend(list(self.dfs[key].columns))
        return out

    def __seed(self):
        try: seed = self.seed + self.epoch
        except: seed = self.seed

        random.seed(self.seed)
        np.random.seed(self.seed)

    def __len__(self): return len(self.index)

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', [batch_size, 119]
                'ts', [batch_size, 239, 184]
                'statics', [batch_size, 54]
                'next_timepoint', [batch_size, 56]
                'next_timepoint_was_measured', [batch_size, 56]
                'disch_24h', [batch_size, 1]
                'disch_48h', [batch_size, 1]
                'Final Acuity Outcome', [batch_size, 1]
                'ts_mask', [batch_size, 239]
                'tasks_binary_multilabel', [batch_size, 7]
                'note_ids', [batch_size, 239, 512]
                'note_masks', [batch_size, 239, 512]
                'note_segment_ids', [batch_size, 239, 512]
                'note_hours_idx', [batch_size, 239]
                'note_hours_num', [batch_size])
        """
        # We'll use these for a bit of special processing surrounding our masked imputation task, so we
        # define them now.
        ts_vals_key, ts_is_measured_key, imputation_mask_key = 'ts_vals', 'ts_is_measured', 'ts_mask'

        # Loading data
        try:
            tensors = None
            if self.epoch_cached:
                tensors = self.cached_epoch[item]
            else:
                loaded, cached_item = self.load_save_path(item=item)
                if loaded: tensors = cached_item

            if tensors is not None:
                if self.save_data_only: return {'null': torch.zeros((1, 1))}
                tensors.update({k:v.float() for k,v in tensors.items() if v.dtype==torch.float16})

                if 'rolling_fts' in tensors:
                    tensors['rolling_ftseq'] = tensors.pop('rolling_fts')

                # Now adding the mask key.
                if self.imputation_mask_rate > 0:
                    any_masked = False
                    while not any_masked:
                        mask_prob = np.random.uniform(size=(self.max_seq_len, 1))
                        any_masked = ((mask_prob < self.imputation_mask_rate).sum() > 0)
                    tensors[imputation_mask_key] = torch.Tensor(np.where(
                        mask_prob < self.imputation_mask_rate,
                        np.ones_like(mask_prob), np.zeros_like(mask_prob)
                    ))
                elif 'ts_mask' in tensors: del tensors['ts_mask']
                    #assert 'ts_mask' not in tensors, (
                    #    f"ts_mask shouldn't be in tensors {self.epoch}, {item}, {self.epoch_cached}, "
                    #    f"{loaded}, {self.get_save_path(item=item)}"
                    #)
                return tensors
        except:
            print(f"Failed to load item {item}")
            print(f"Save path: {self.get_save_path(item=item)}")
            raise

        # Now we actually need to create the item, but we may not have bothered to lad the dataframes yet. If
        # not, we'll do that now.
        try:
            self.dfs
            print_shapes = False
        except AttributeError as e:
            print(f"Failed to load item from {self.get_save_path(item=item)}. Reloading dfs and creating it.")
            assert hasattr(self, 'reload_self_dir'), f"Can't build items as lacks dfs or reload_self_dir!"

            full_self_path = os.path.join(self.reload_self_dir, f"{self.train_tune_test}_dataset.pkl")
            assert os.path.isfile(full_self_path), f"{full_self_path} doesn't exist! Can't reload dfs."

            full_dataset = depickle(full_self_path)
            self.dfs = full_dataset.dfs
            self.subjects = full_dataset.subjects
            self.orig_subjects = full_dataset.orig_subjects

            self.reset_index()
            print_shapes = True
            print("Reloaded dfs. Continuing.")

        # Icustay id is always first.
        idx = self.index[item]
        if type(idx) is tuple:
            icustay_id, end_time = idx
            start_time = max(end_time - self.max_seq_len, 0)
            seq_len = end_time - start_time
        else:
            icustay_id = idx
            if self.sequence_len:
                end_time   = self.sequence_len
                start_time = max(end_time - self.max_seq_len, 0)
                seq_len    = end_time - start_time
            else:
                max_seq_len = min(self.max_hours[item], self.max_seq_len)
                end_time    = random.randint(self.min_seq_len, self.max_hours[item]) # the end time for this patient
                start_time  = max(end_time - max_seq_len, 0) # the start time corresponding to the random_end_time
                seq_len     = end_time - start_time

        assert seq_len <= self.max_seq_len, f"seq_len is {seq_len}, which is not less than or equal to max seq_length=={max_seq_len}"

        correction_attempts = 0
        while 'rolling_fts' in self.dfs and 'rolling_ftseq' not in self.dfs:
            try:
                print("Amending dfs to include rolling_ftseq")
                self.dfs['rolling_ftseq'] = self.dfs['rolling_fts']
                self.dfs.pop('rolling_fts', None)
            except: pass
            correction_attempts += 1
            if correction_attempts > 10:
                raise ValueError(f"Failed to correct dataframes fts v. ftseq bug!")


        # collect the indices for the patient
        idxs = {k: (df.index.get_level_values('icustay_id') == icustay_id) for k, df in self.dfs.items()}
        # We'll piggy back on our "next_timepoint" task for this imputation task. A more elegant solution
        # would be to just store the measurement indicators and use them for both this task and the
        # next timepoint prediction, but that's not how things are implemented for now.
        idxs[ts_is_measured_key] = idxs['next_timepoint_was_measured'].copy()

        # get the indices for each df between start_time and end_time
        # Note for our special case of `ts_is_measured_key` & `next_timepoint_was_measured`, we still have it
        # the case that the input features end at *<* end_time, whereas the target extractions are
        # *==* end_time, so this should be valid.
        for idxs_k, dfs_k in (
            ('ts', 'ts'), ('notes', 'notes'), (ts_is_measured_key, 'next_timepoint_was_measured'),
        ):
            if idxs_k in idxs:
                hours_in = self.dfs[dfs_k].index.get_level_values('hours_in')
                idxs[idxs_k] &= ((hours_in >= start_time) & (hours_in < end_time))


        # get the next task for predictions
        for k in [
            'rolling_tasks_binary_multilabel', 'rolling_tasks_multiclass', 'rolling_ftseq', 'next_timepoint',
            'next_timepoint_was_measured',
        ]:
            if k not in self.dfs or self.dfs[k] is None: continue
            if k in idxs: idxs[k] &= (self.dfs[k].index.get_level_values('hours_in') == end_time)

        # get the correct subset of the dfs
        dfs = {k: df.loc[idxs[k]].copy() for k, df in self.dfs.items() if df is not None}
        dfs[ts_is_measured_key] = self.dfs['next_timepoint_was_measured'].loc[idxs[ts_is_measured_key]].copy()

        # break up all of these dataframes that were processed as one into individual dfs
        for k in ('rolling_tasks_multiclass', 'static_tasks_multiclass'):
            df = dfs[k]
            for c in df.columns:
                dfs[c] = df[[c]]

            del dfs[k]

        if seq_len != len(dfs['ts']):
            print(idx, start_time, end_time, self.sequence_len)
            raise AssertionError("Length mismatch! %d v %d" % (seq_len, len(dfs['ts'])))

        # For the next timepoint, we only want the means of measured labs.
        # TODO(mmd): Is this the right place for this logic? Or should it go earlier?
        cols = dfs['next_timepoint'].columns
        # print(cols)
        mean_labs_cols = [c for c in cols if type(c) is tuple and c[1] == 'mean']
        dfs['next_timepoint'] = dfs['next_timepoint'][mean_labs_cols].fillna(value=-1)

        # Here, we pull out data for a masked imputation task. We want to store separately the continuous TS
        # values (not imputed, as we don't want to predict imputed values), indicators of whether TS vals were
        # measured (to mask out values we don't want to include in our imputation value and to predict what
        # values should be imputed at any masked timepoint), and a mask key for the entire timeseries to
        # indicate which timepoints are actually masked.

        dfs[ts_vals_key] = dfs['ts'].loc[:, mean_labs_cols].copy().fillna(0)
        # dfs[ts_is_measured_key] is already defined, based on the logic above.
        # dfs[imputation_mask_key] we'll actually construct later, in the numpy arrays directly, as it doesn't have the
        # same structure (e.g., column names) as the real dfs, we just need to match shape.


        # TS continuous ais the only remaining actual timeseries feature.
        # It needs to be imputed, padded, and reshaped.
        dfs['ts'].loc[:, self.ts_continuous_cols] = self.impute_fn(
            dfs['ts'].loc[:, self.ts_continuous_cols]
        ).fillna(0) # First impute, then fill w/ 0.


        if self.using_pretrained_notes:
            dfs['notes'].loc[:, self.notes_cols] = self.impute_fn(
                dfs['notes'].loc[:, self.notes_cols]
            ).fillna(0) # First impute, then fill w/ 0.  # this is producing nans

        np_arrays = {k: df.values for k, df in dfs.items()}
        # We will deal with notes separately if we are integrating them instead of simply using pretrained embeddings
        if not self.using_pretrained_notes:
            np_arrays.pop('notes', None)

        # Now adding the mask key.
        if self.imputation_mask_rate > 0:
            any_masked = False
            while not any_masked:
                mask_prob = np.random.uniform(size=(self.max_seq_len, 1))
                any_masked = ((mask_prob < self.imputation_mask_rate).sum() > 0)
            np_arrays[imputation_mask_key] = np.where(
                mask_prob < self.imputation_mask_rate, np.ones_like(mask_prob), np.zeros_like(mask_prob)
            )

        # Padding
        for k in ('ts', ts_vals_key, ts_is_measured_key, 'notes'):
            if k in np_arrays:
                num_features = np_arrays[k].shape[1]
                if np_arrays[k].shape[0] != self.max_seq_len:
                    if self.max_seq_len > seq_len:
                        pad = np.zeros((self.max_seq_len - seq_len, num_features))
                        np_arrays[k] = np.expand_dims(np.concatenate((np_arrays[k], pad)), 0)
                elif self.max_seq_len == seq_len:
                    np_arrays[k] = np.expand_dims(np_arrays[k], 0)

        try:
            np_arrays['tasks_binary_multilabel'] = np.concatenate(
                [np_arrays[k] for k in self.binary_multilabel_task_concat_order], axis=1
            )
            del np_arrays['rolling_tasks_binary_multilabel']
            del np_arrays['static_tasks_binary_multilabel']
        except ValueError as e:
            print(idx, start_time, end_time, self.sequence_len)
            for k in self.binary_multilabel_task_concat_order:
                print(f"{k}: {np_arrays[k].shape}")
            raise

        # Notes
        if self.using_integrated_notes:
            raise NotImplementedError("Doesn't support notes at present.")

        tensors = {}
        for k, arr in np_arrays.items():
            #assert arr.shape[0] == 1, f"Must only have one first dimension for {k}! Got {arr.shape}"
            # print(k, arr.shape)
            if arr.shape[0] == 1: tensors[k] = torch.tensor(arr[0])
            else: tensors[k] = torch.tensor(arr)

        if self.imputation_mask_rate == 0:
            assert 'ts_mask' not in tensors, f"{item}, {idx}, {k: t.shape for k, t in tensors.items()}"
        else:
            assert 'ts_mask' in tensors, f"{item}, {idx}, {k: t.shape for k, t in tensors.items()}"

        if print_shapes: print({k: t.shape for k, t in tensors.items()})

        # we don't want to overwrite normal evaluation scripts for masked eval
        if self.imputation_mask_rate == 0: self.save_item(tensors, item=item)

        return tensors


class PatientDatasetSample(Dataset):
    def __init__(self,seed: int = 1, vocab:Dict={}, max_time_since_measured: int=8, max_seq_len: int=480, max_note_len: int = 512, min_seq_len:   int = 24, sequence_length: int = None):
        self.num_subjects=20000
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.sequence_length = sequence_length

    def __seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __len__(self): return self.num_subjects

    def get_binary_multilabel_keys(self):
        out = []
        for key in self.binary_multilabel_task_concat_order: out.extend(list(self.dfs[key].columns))
        return out

    def __getitem__(self, item):
        """
        Returns:
            tensors (dict)
                'rolling_ftseq', [batch_size, 119]
                'ts', [batch_size, 239, 184]
                'statics', [batch_size, 54]
                'next_timepoint', [batch_size, 56]
                'next_timepoint_was_measured', [batch_size, 56]
                'disch_24h', [batch_size, 1]
                'disch_48h', [batch_size, 1]
                'Final Acuity Outcome', [batch_size, 1]
                'ts_mask', [batch_size, 239, 1]
                'tasks_binary_multilabel', [batch_size, 7]
                'note_ids', [batch_size, 239, 512]
                'note_masks', [batch_size, 239, 512]
                'note_segment_ids', [batch_size, 239, 512]
                'note_hours_idx', [batch_size, 239]
                'note_hours_num', [batch_size])
        """
        note_hours_num = np.random.randint(0, self.max_seq_len)
        # sample num notes between 0 and max_seq_length
        notes=list(range(self.max_seq_len))
        rand_notes = sorted(random.sample(notes, note_hours_num))
        note_hour_index = np.ones(self.max_seq_len)*-1
        note_hour_index[:note_hours_num]=np.asarray(rand_notes)

        a=np.zeros(self.max_seq_len*512)
        a[rand_notes]=1


        np_arrays={'rolling_ftseq': np.random.rand(119).reshape(1, -1),
                'ts': np.random.rand(self.max_seq_len*184).reshape(1, self.max_seq_len, 184),
                'statics': np.random.rand(54).reshape(1, -1),
                'next_timepoint': np.random.rand(56).reshape(1, -1),
                'next_timepoint_was_measured': np.random.randint(0, 2, 56).reshape(1, -1),
                'disch_24h': np.random.rand(1).reshape(1, -1),
                'disch_48h': np.random.rand(1).reshape(1, -1),
                'Final Acuity Outcome': np.random.rand(1).reshape(1, -1),
                'ts_mask': np.random.rand(self.max_seq_len).reshape(1, -1),
                'tasks_binary_multilabel': np.random.randint(0, 2, 7).reshape(1, -1),
                'note_ids': (a*np.random.rand(self.max_seq_len*512)).reshape(1, self.max_seq_len, 512),
                'note_masks': a.reshape(1, self.max_seq_len, 512),
                'note_segment_ids': np.random.rand(self.max_seq_len*512).reshape(1, self.max_seq_len, 512),
                'note_hours_idx': note_hour_index.reshape(1, -1),
                'note_hours_num': np.asarray(note_hours_num).reshape(1, -1)}

        tensors = {}
        for k, arr in np_arrays.items():
            assert arr.shape[0] == 1, "Must only have one first dimension!"
            # print(k, arr.shape)
            tensors[k] = torch.tensor(arr[0])

        return tensors
