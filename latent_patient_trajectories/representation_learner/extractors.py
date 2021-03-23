import abc, enum, itertools, random, numpy as np, pandas as pd
from collections import Counter
from typing import Callable, Dict, Sequence, ClassVar, Set
idx=pd.IndexSlice

from ..constants import *
from ..utils import *
from .utils import *

# TODO(mmd): Add type hints everywhere.
# Helper Functions
def hrs(x): return pd.to_timedelta(x, unit='hour')
def days(x): return pd.to_timedelta(x, unit='day')

### Base Classes:
#

# proper enums break in jupyter autoreload....
class FeatureTypes():
    CONTINUOUS       = 'continuous' # A continuous float.
    # A categorical variable, represented in pandas as a categorical type or str, represented in output by an
    # integer index into the extractor class's vocab map.
    CATEGORICAL      = 'categorical'
    BINARY           = 'binary' # A binary variable 0/1 encoded.
    # An ordinal variable represented by a numerical type. Distance proportional to euclidean distance.
    ORDINAL          = 'ordinal'
    # An ordinal variable represented by an integer. Cycles around max -- e.g., all are equivalent modulo the
    # max integer present. Distances proportional to modular distance.
    ORDINAL_CYCLICAL = 'ordinal_cyclical'

class Extractor(abc.ABC):
    """ A base class for all extractors, which ingest dataframes as produced according to data_utils and
        yield various labels/features.
    """

    def __init__(self, seed=1):
        self.is_fit  = False
        self.seed    = seed
        self.columns = {} # Dict[str, FeatureTypes]

    def set_seed(self, seed, reseed_immediately=True):
        self.seed = seed
        if reseed_immediately: self.__seed()

    def __seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    @abc.abstractmethod
    def _fit(
        self,
        dfs: Sequence[pd.DataFrame],
    ):
        """ TODO: Overwrite this in base classes. """
        # All inputs of the form (statics, numerics, treatments, codes, notes, treatment_sequences)

    @abc.abstractmethod
    def _extract(
        self,
        dfs: Sequence[pd.DataFrame],
    ):
        """ TODO: Overwrite this in base classes. """
        # All inputs of the form (statics, numerics, treatments, codes, notes, treatment_sequences)

    def fit(self, *dfs):
        self.__seed()
        self._fit(dfs)
        self.is_fit = True

    def extract(self, *dfs):
        assert self.is_fit, "Must fit the extractor first on train data!"
        self.__seed()
        return self._extract(dfs)

class RollingTimedStaticEventsExtractor(Extractor):
    START_TIME_COL = 'START_TIME'

    @classmethod
    def __column_output_type(v):
        if v[0] == 0 and v[-1] == 1: return FeatureTypes.BINARY
        elif type(v[0]) is str or type(v[1]) is str: return FeatureTypes.CATEGORICAL

    # TODO: Need to fit vocab
    def __init__(
        self,
        tasks = {
            # TODO: Better names.
            # TASK:      (INIT_COL, TIME_COL, WINDOW, GAP, EXCLUSION_FN, VALUE_NUM_OR_COL)
        },
        start_time_cols = [],
        err_early       = True, # whether or not to err on the early side or the late side. 
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        assert len(tasks) > 0, "Must extract some tasks!"

        self.tasks           = tasks
        self.start_time_cols = start_time_cols
        self.err_early       = err_early

    def __get_time_vals(self, df, col, is_begin):
        if type(col) is str:             vals = df[col]
        elif is_begin == self.err_early: vals = df[list(col)].max(axis=1)
        else:                            vals = df[list(col)].min(axis=1)

        return vals

    def _fit(self, dfs):
        statics = dfs[0]

        self.vocab, self.columns = {}, {}
        for task, (init_val_or_col,time_col,window,gap_time,exclusion_fn,val_or_col) in self.tasks.items():
            task_vocab = Counter()
            for col in (init_val_or_col, val_or_col):
                if type(col) is not str or col not in statics:
                    if col == init_val_or_col: task_vocab[col] = len(statics)
                    else: task_vocab.update([col, col])
                    continue

                vals = statics[col].values

                try:
                    any_null = np.isnan(vals).any()
                    vals = vals[~np.isnan(vals)]
                except TypeError:
                    any_null = False
                    pass
                task_vocab.update(vals)
                if any_null and np.NaN not in task_vocab: task_vocab.update([np.NaN])
            if set(task_vocab.keys()) == {0, 1}: self.columns[task] = FeatureTypes.BINARY
            else:
                self.columns[task] = FeatureTypes.CATEGORICAL
                self.vocab[task] = [k for k, cnt in task_vocab.most_common()]

        self.vocab_map = {t: {k: i for i, k in enumerate(v)} for t, v in self.vocab.items()}
        self.cols_to_transform = list(self.vocab_map.keys())
        self.is_fit    = True

    def _extract(self, dfs):
        statics = dfs[0].copy()

        # numerics has the right schema, but we don't need any of its data.
        out_df = dfs[1][[]].copy()

        # I compute the start time pre-join to avoid doing unnecessary work.
        statics[self.START_TIME_COL] = self.__get_time_vals(
            statics, self.start_time_cols, is_begin=True
        )

        out_df  = out_df.join(statics, on=PATIENT_ID_COLS + [FOLD_IDX_LVL], how='left')
        start_time = out_df[self.START_TIME_COL]
        time_in = pd.to_timedelta(out_df.index.get_level_values(DURATIONS_COL), unit='hour')

        duration_cols = {}

        for task, (init_val_or_col,time_col,window,gap_time,exclusion_fn,val_or_col) in self.tasks.items():
            if time_col in duration_cols: at_duration_col = duration_cols[time_col]
            else:
                at_duration_col = self.__get_time_vals(out_df, time_col, is_begin=False) - start_time

                if at_duration_col.dtype is not pd.to_timedelta([3,4], unit='hour').dtype:
                    at_duration_col = pd.to_timedelta(at_duration_col, unit='hour')

                duration_cols[time_col] = at_duration_col

            if window is None:
                if self.err_early: at_duration_col = at_duration_col.dt.floor('H')
                else:              at_duration_col = at_duration_col.dt.ceil('H')
                in_window  = (time_in == at_duration_col)
                gap_window = None
            else:
                if gap_time is None: gap_time = hrs(0)
                if type(gap_time) is tuple:
                    gap_front, gap_back = gap_time
                    gap_window = (
                        (time_in < at_duration_col + gap_back) &
                        (time_in >= at_duration_col - gap_front)
                    )
                    in_window = (time_in < at_duration_col - gap_front) & (time_in >= at_duration_col - window)
                else:
                    gap_window = (time_in < at_duration_col) & (time_in >= at_duration_col - gap_time)
                    in_window = (time_in < at_duration_col - gap_time) & (time_in >= at_duration_col - window)

            if exclusion_fn is not None:
                exclusion_window = exclusion_fn(out_df)
                in_window = in_window & (~exclusion_window)
            else: exclusion_window = None

            values = out_df.loc[in_window, val_or_col] if type(val_or_col) is str else val_or_col

            out_df[task] = out_df[init_val_or_col] if init_val_or_col in out_df else init_val_or_col
            out_df.loc[in_window, task] = values

            if gap_window is not None: out_df.loc[gap_window, task] = np.NaN
            if exclusion_window is not None: out_df.loc[exclusion_window, task] = np.NaN

        out_df = out_df[list(self.columns.keys())]



        out_df.loc[:, self.cols_to_transform] = out_df.transform(
            {c: lambda x: m[x] if x in m else np.NaN for c, m in self.vocab_map.items()}
        )[self.cols_to_transform]

        return out_df

### Input Featurizers
#
##  Static Featurizers
#
class DemographicFeaturizer(Extractor):
    MEDIAN_OF_CORRECTED_AGE = 91.4 # source https://mimic.physionet.org/mimictables/patients/

    def __init__(
        self,
        columns = {
            'age':            FeatureTypes.CONTINUOUS,
            'gender':         FeatureTypes.CATEGORICAL,
            'ethnicity':      FeatureTypes.CATEGORICAL,
            'insurance':      FeatureTypes.CATEGORICAL,
            'admission_type': FeatureTypes.CATEGORICAL,
            'first_careunit': FeatureTypes.CATEGORICAL,
        },
        continuous_features_bucket_sizes = None,
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.columns = columns
        if type(continuous_features_bucket_sizes) is dict:
            self.continuous_features_bucket_sizes = continuous_features_bucket_sizes
        elif continuous_features_bucket_sizes is not None:
            self.continuous_features_bucket_sizes = {
                k: continuous_features_bucket_sizes for k,v in columns.items() if v == FeatureTypes.CONTINUOUS
            }
        else: self.continuous_features_bucket_sizes = {}

        for k in self.continuous_features_bucket_sizes: self.columns[k] = FeatureTypes.ORDINAL

        self.categorical_columns = {k for k, v in columns.items() if v == FeatureTypes.CATEGORICAL}
        self.continuous_columns = {k for k, v in columns.items() if v == FeatureTypes.CONTINUOUS}

    def __correct(self, df):
        if 'age' not in df: return df

        # Age correction.
        df.loc[df['age'] > 300, 'age'] = self.MEDIAN_OF_CORRECTED_AGE
        return df

    def _fit(self, dfs):
        statics = self.__correct(dfs[0].copy())
        self.means = statics[self.continuous_columns].mean(axis=0)

        self.stds = statics[self.continuous_columns].std(axis=0)

        self.vocab = {c: [UNK] + list(sorted(set(statics[c]) - {UNK,})) for c in self.categorical_columns}
        self.vocab_map = {c: {k: i for i, k in enumerate(v)} for c, v in self.vocab.items()}

        # # narrow down ethnicities to 5 bins
        def simple_ethnicities(item):
            if 'asian' in item.lower(): return 1
            elif 'black' in item.lower(): return 2
            elif 'hisp' in item.lower(): return 3
            elif 'white' in item.lower(): return 4
            else: return 0

        update_ethnicity={k: simple_ethnicities(k) for k in self.vocab['ethnicity']}
        self.vocab_map['ethnicity']=update_ethnicity


        self.continuous_features_buckets, self.ordinal_labels_means = {}, {}
        for c, N in self.continuous_features_bucket_sizes.items():
            vals      = statics[c].dropna().values
            quantiles = list(np.quantile(vals, np.linspace(0, 1, N))) + [np.Inf]
            self.continuous_features_buckets[c] = quantiles
            self.ordinal_labels_means[c] = [
                np.mean(vals[quantiles[i] <= vals and vals < quantiles[i+1]]) for i in range(N)
            ]

        self.cols_to_transform = list(self.categorical_columns)+list(self.continuous_features_buckets.keys())

    def __transform_fntr(self, c):
        if c in self.categorical_columns:
            m = self.vocab_map[c]
            return lambda x: m[x] if x in m else m[UNK]
        elif c in self.continuous_features_buckets:
            return lambda x: next(i for i, v in enumerate(self.continuous_features_buckets[c]) if x < v)
        else: raise NotImplementedError("Don't have a transformation here!")

    def _extract(self, dfs):
        out_df = self.__correct(dfs[0][list(self.columns.keys())].copy())
        out_df.loc[:, self.cols_to_transform] = out_df.transform(
            {c: self.__transform_fntr(c) for c in self.cols_to_transform}
        )[self.cols_to_transform]
        out_df.loc[:, self.continuous_columns] = (out_df[self.continuous_columns] - self.means)/self.stds

        return out_df


##  Rolling TS Featurizers
#
class EOLCareTSFeaturizer(RollingTimedStaticEventsExtractor):
    """ Basically just defaults for RTPE Extractor. """
    START_TIME_COL = 'intime' # Here we just use intime as we want something that is maximally late.

    def __init__(
        self,
        columns = {
            # TASK NAME                  INITIAL_VAL, TIME_AT_ON
            'DNR Ordered':              ('dnr_first', 'dnr_first_charttime'),
            'Comfort Measures Ordered': ('cmo_first', 'timecmo_chart'),
        },
        **base_class_kwargs,
    ):
        # We use a negative window to ensure that in all subsequent cells, the EOL care signal will be on.
        tasks = {c: tuple(list(v) + [hrs(0), days(-1e3), None, 1]) for c, v in columns.items()}

        # We want to err late here, so as to not give the model signals about a patient as input before they
        # could actually happen.
        super().__init__(tasks, self.START_TIME_COL, err_early=False, **base_class_kwargs)

    def _extract(self, dfs):
        return super()._extract(dfs).fillna(0) # Stupid, but I don't want NaNs in my output, to avoid leakage.


class LabsFeaturizer(Extractor):
    TIME_OF_DAY = 'time_in_day'

    def __init__(
        self,
        labs:                     Sequence[str] = 'ALL',
        agg_fns:                  Sequence[str] = ['mean', 'std'],
        max_time_since_measured:            int = 8,
        impute_here:                       bool = False,
        include_time_of_day:               bool = False,
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.labs, self.agg_fns, self.impute_here = labs, agg_fns, impute_here
        self.init_time_since_measured = max_time_since_measured
        self.max_time_since_measured  = max_time_since_measured
        self.include_time_of_day      = include_time_of_day

        assert not include_time_of_day, "Can't actually extract this yet..."
        if include_time_of_day: self.columns[(self.TIME_OF_DAY, None)] = FeatureTypes.ORDINAL_CYCLICAL

    def _fit(self, dfs):
        if self.labs == 'ALL': self.labs = list(set(dfs[1].columns.get_level_values(0)))

        # TODO(mmd): Constant-ify below
        self.columns = {
            (c, a): FeatureTypes.CONTINUOUS for c, a in itertools.product(self.labs, self.agg_fns)
        }

        self.columns.update({(c, 'time_since_measured'): FeatureTypes.ORDINAL for c in self.labs})
        self.vocab = list(range(self.max_time_since_measured)) # TODO: will this be helpful?

        numerics = dfs[1].loc[:, idx[self.labs, self.agg_fns]]
        self.means, self.stds = numerics.mean(axis=0), numerics.std(axis=0)

    def _extract(self, dfs):
        # Don't impute, just extract time_since_measured and 
        numerics = dfs[1].loc[:, idx[self.labs, :]].copy()
        numerics.loc[:, idx[:, self.agg_fns]] = (numerics.loc[:, idx[:, self.agg_fns]]-self.means) / self.stds

        numerics = add_time_since_measured(
            numerics,
            init_time_since_measured = self.init_time_since_measured,
            max_time_since_measured  = self.max_time_since_measured,
            hour_aggregation = 1,
        )
        numerics = numerics.loc[:, idx[:, self.agg_fns + ['time_since_measured']]]
        return numerics

class TreatmentStatusExtractor(Extractor):
    def __init__(
        self,
        treatments: Sequence[str] = 'ALL',
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.treatments = treatments

    def _extract(self, dfs): return dfs[2][self.treatments].copy()
    def _fit(self, dfs):
        if self.treatments == 'ALL': self.treatments = sorted(list(set(dfs[2].columns)))

        self.columns = {t: FeatureTypes.BINARY for t in self.treatments} # TODO: this is wrong!
        self.is_fit = True

# TODO:
# class NotesExtractor(Extractor):

### Output Extractors
#
##  Static Extractors
#   Here we produce several static extractors, including one for a final Acuity Extractor (mort++) and
#   one for long LOS.
#

class DeathLabels(enum.Enum):
    IN_ICU                     = enum.auto()
    IN_HOSPITAL                = enum.auto()
    WITHIN_30_DAYS             = enum.auto()
    WITHIN_1_YEAR              = enum.auto()

class StaticAcuityOutcomeExtractor(Extractor):
    DISCHARGE_LOCATIONS_TO_EXCLUDE = {'DEAD/EXPIRED', 'Death'}
    # TODO: constant
    def __init__(
        self,
        name='Final Acuity Outcome',
        eicu=False,
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.name = name
        self.columns = {name: FeatureTypes.CATEGORICAL}
        self.eicu = eicu

    def _fit(self, dfs):
        statics = dfs[0]
        print(statics.columns.tolist())
        #print(dfs[6].columns.tolist())
        self.vocab = [dl for dl, cnt in Counter(statics.discharge_location).most_common()]
        self.vocab = [dl for dl in self.vocab if dl not in self.DISCHARGE_LOCATIONS_TO_EXCLUDE]
        self.vocab += [x.name for x in DeathLabels]
        self.vocab_map = {k: i for i, k in enumerate(self.vocab)}
        self.vocab_map['unknown'] = len(self.vocab_map)

    def _extract_row(self, r):
        if r.discharge_location in self.vocab_map: return self.vocab_map[r.discharge_location]
        elif r.mort_icu  == 1:                     return self.vocab_map[DeathLabels.IN_ICU.name]
        elif self.eicu and r.mort_hosp == 1:       return self.vocab_map[DeathLabels.IN_HOSPITAL.name]
        elif (not self.eicu) and r.hospital_expire_flag == 1: return self.vocab_map[DeathLabels.IN_HOSPITAL.name]
        else: return np.NaN

    def _extract(self, dfs):
        statics = dfs[0].copy()

        acuity_outcomes = statics.apply(self._extract_row, axis=1)
        acuity_outcomes.name = self.name
        return pd.DataFrame(acuity_outcomes)

class ReadmissionExtractor(Extractor):
    # TODO: constant
    def __init__(
        self,
        name='Readmission 30',
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.name = name
        self.columns = {name: FeatureTypes.BINARY}

    def _fit(self, dfs):
        """ Do nothing. Extractor is natively fit. """
        pass

    def _extract(self, dfs):
        statics = dfs[0].copy()
        readmission_df = statics['readmission_30']
        readmission_df.name = self.name
        return pd.DataFrame(readmission_df)


class StaticLongLOSExtractor(Extractor):
    NAME = 'Long LOS'
    def __init__(
        self,
        los_boundaries = (days(3),),
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.los_boundaries = sorted(los_boundaries, reverse=False)
        self.columns        = {self.NAME: FeatureTypes.BINARY}

        prev_start = None
        self.vocab = []
        for boundary in self.los_boundaries:
            self.vocab.append((prev_start, boundary))
            prev_start = boundary
        self.vocab.append((prev_start, None))

        self.is_fit = True # This extractor doesn't need to be fit.

    def __convert_los(self, l):
        for i, boundary in enumerate(self.los_boundaries):
            if boundary > l: return i
        return len(self.los_boundaries)

    def _extract(self, dfs):
        los = pd.to_timedelta(dfs[0].copy().los_icu, unit='day').apply(self.__convert_los)
        los.name = self.NAME
        return pd.DataFrame(los)

    def _fit(self, dfs):
        """ Do nothing. Extractor is natively fit. """

## Rolling Extractors
#  Here we produce several rolling time varying extractors, including a rolling acuity event extractor and a
#  rolling FTS extractor.
#
#

def e_fn(x):
    print('woo')
    print(x.columns)
    print(x.head())
    return x.dnr_first == 1

class RollingAcuityEventsExtractor(RollingTimedStaticEventsExtractor):
    """ Basically just defaults for RollingTimedStaticEventsExtractor """
    START_TIME_COLS = 'intime'

    def __init__(
        self,
        tasks = {
            # TODO: Better names.
            # TASK:      (INIT, TIME_COL,        WINDOW,  GAP,                 EXCLUSION_FN,   VALUE_NUM/COL
            'mort_24h':  (0,    'deathtime',     days(1), hrs(2),              None,                       1),
            'mort_48h':  (0,    'deathtime',     days(2), hrs(6),              None,                       1),
            'dnr_24h':   (0,    'dnr_first_charttime', days(1), (hrs(2), days(1e3)), lambda x: x.dnr_first == 1, 1),
            'dnr_48h':   (0,    'dnr_first_charttime', days(2), (hrs(6), days(1e3)), lambda x: x.dnr_first == 1, 1),
            'cmo_24h':   (0,    'timecmo_chart', days(1), hrs(2), None, 1),
            'cmo_48h':   (0,    'timecmo_chart', days(2), hrs(6), None, 1),
            'disch_24h': (
                'NO_DISCH', ('outtime', 'dischtime'), days(1), hrs(2), lambda x: x.hospital_expire_flag == 1,
                'discharge_location',
            ),
            'disch_48h': (
                'NO_DISCH', ('outtime', 'dischtime'), days(2), hrs(6), lambda x: x.hospital_expire_flag == 1,
                'discharge_location',
            ),
        },
        **base_class_kwargs,
    ):
        # These are outputs, so we want to err early (as predicting early is better than predicting late and
        # acquiring label leakage.
        super().__init__(tasks, self.START_TIME_COLS, err_early=True, **base_class_kwargs)

class RollingFTSExtractor(Extractor):
    # TODO(mmd): Remove outcome_acuity_extractor... It isn't a good idea to have it all as one.
    def __init__(
        self,
        gap_time                 = hrs(6),
        multilabel               = False,
        #outcome_acuity_extractor = StaticAcuityOutcomeExtractor(),
        name                     = 'Future Treatment Sequence',
        max_seq_len_cap          = 200,
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        assert not multilabel, "Doesn't actually support Multilabel yet."

        self.gap_time                 = gap_time
        self.multilabel               = multilabel
        #self.outcome_acuity_extractor = outcome_acuity_extractor
        self.name                     = name
        self.columns                  = {name: FeatureTypes.CATEGORICAL}
        self.max_seq_len_cap          = max_seq_len_cap

    def _fit(self, dfs):
        #if not self.outcome_acuity_extractor.is_fit: self.outcome_acuity_extractor.fit(*dfs)

        treatment_sequences = dfs[-1]['Treatment Sequence']
        self.vocab = ['PAD'] + sorted(
            list(set(frozenset(x) for x in treatment_sequences))
        )# + self.outcome_acuity_extractor.vocab
        self.vocab_map = {k: i for i, k in enumerate(self.vocab)}

        self.max_len = min(self.max_seq_len_cap, treatment_sequences.groupby(PATIENT_ID_COLS).count().max())

    def _extract(self, dfs):
        #outcomes = self.outcome_acuity_extractor.extract(*dfs)

        as_indices = dfs[-1].copy()
        as_indices['Treatment Sequence'] = as_indices['Treatment Sequence'].apply(
            lambda t: self.vocab_map[frozenset(t)]
        )

        # This way, when we join, we'll get the future treatment sequence as of gap time hours ahead:
        adj_hours_in = pd.to_timedelta(
            as_indices.index.get_level_values(DURATIONS_COL), unit='hour'
        ) - self.gap_time
        adj_hours_in = (adj_hours_in / pd.Timedelta('1 hour')).astype(int)
        as_indices.index = as_indices.index.droplevel([DURATIONS_COL])
        as_indices[DURATIONS_COL] = adj_hours_in
        as_indices.set_index(DURATIONS_COL, append=True, inplace=True)

        as_indices['sequence_element'] = as_indices.groupby(PATIENT_ID_COLS).cumcount()+1
        expanded = as_indices.pivot_table(
            columns=['sequence_element'], values=['Treatment Sequence'], index=as_indices.index.names
        ).groupby(PATIENT_ID_COLS).fillna(method='bfill')

        future_sequences = expanded.apply(
            lambda x: pad(x.values[~x.isnull().values][:self.max_len], self.max_len), axis=1
        )
        future_sequences.name = self.name

        # treatments has the right schema, but we don't need any of its data.
        out_df = dfs[2][[]].copy()
        out_df = out_df.join(
            future_sequences, on=PATIENT_ID_COLS + [FOLD_IDX_LVL, DURATIONS_COL], how='left'
        )

        out_df_filled = out_df.groupby(PATIENT_ID_COLS + [FOLD_IDX_LVL]).fillna(method='bfill')
        out_df_filled['Future Treatment Sequence'] = out_df_filled['Future Treatment Sequence'].apply(
            lambda x: [0] * self.max_len if type(x) is float and np.isnan(x) else x
        )

        # TODO(mmd): For some reason the last several (e.g., 2, 3) of this dataframe is exclusively 0s.
        out_df_seq = pd.DataFrame(
            out_df_filled['Future Treatment Sequence'].values.tolist(), index=out_df_filled.index
        )
        out_df_seq.columns.names = ['Future Treatment Sequence Element']

        return out_df_seq

        #out_df_with_outcomes = out_df_filled.join(outcomes, on=PATIENT_ID_COLS + [FOLD_IDX_LVL], how='left')
        #return out_df_with_outcomes

        #or_empty_list = lambda x: x if type(x) is list else []
        #seqs_with_outcome = out_df_with_outcomes.apply(
        #    lambda r: or_empty_list(r['Future Treatment Sequence']) + [r[self.outcome_acuity_extractor.name]],
        #    axis=1
        #)
        #seqs_with_outcome.name = self.name

        #return seqs_with_outcome

def rolling_treatment_events(*dfs):
    """ TODO """

def rolling_los(*dfs):
    """ TODO """

class CodesExtractor(Extractor):
    def __init__(
        self,
        name='Codes',
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        self.name = name
        # self.columns = {name: FeatureTypes.CATEGORICAL}
        cols=['infection','neoplasms', 'endocrine', 'blood',  'mental', 'nervous', 'circulatory', 'respiratory', 'digestive',
          'genitourinary', 'pregnancy', 'skin', 'musculoskeletal', 'congenital', 'perinatal', 'ill_defined', 'injury', 'unknown']        
        self.col_names={val:'icd_'+item for val, item in enumerate(cols) }

        self.columns = {v: FeatureTypes.BINARY for v in self.col_names.values()}

        #end init
    def __lookup_codes(self, row):
        icd9_codes=row['icd9_codes']
        result = [0]*18
        for code in icd9_codes:
            if 'v' in code.lower():
                continue
            if 'e' in code.lower():
                continue
            if len(code)>3:
                code = code[:3]+'.'+code[4:]
            code = float(code)
            if (0<=code)&(code<140):
                #infection
                result[0]=1
            if (140<=code)&(code<240):
                #neoplasms
                result[1]=1
            if (240<=code)&(code<280):
                #endocrine/nutritional/metabolic
                result[2]=1
            if (280<=code)&(code<290):
                #diseases of blood and blood forming organs
                result[3]=1
            if (290<=code)&(code<319):
                #Mental disorders
                result[4]=1
            if (320<=code)&(code<390):
                #Diseases of Nervous System and Sense Organs (PDF, 52KB)
                result[5]=1
            if (390<=code)&(code<460):
                #Diseases of the Circulatory System (PDF, 23KB)
                result[6]=1
            if (460<=code)&(code<520):
                #Diseases of the Respiratory System (PDF, 16KB)
                result[7]=1
            if (520<=code)&(code<580):
                #Diseases of the Digestive System (PDF, 24KB)
                result[8]=1
            if (580<=code)&(code<630):
                #Diseases of the Genitourinary System (PDF, 26KB)
                result[9]=1
            if (630<=code)&(code<680):
                #Complications of Pregnancy, Childbirth and the Puerperium (PDF, 30KB)
                result[10]=1
            if (680<=code)&(code<710):
                #Diseases of the Skin and Subcutaneous Tissue (PDF, 13KB)
                result[11]=1
            if (710<=code)&(code<740):
                #Diseases of the Musculoskeletal System and Connective Tissue (PDF, 25KB)
                result[12]=1
            if (740<=code)&(code<760):
                #Congenital Anomalies (PDF, 208KB)
                result[13]=1
            if (760<=code)&(code<780):
                #Certain Conditions Originating in the Perinatal Period (PDF, 201KB)
                result[14]=1
            if (780<=code)&(code<800):
                #Symptoms, Signs and Ill-defined Conditions (PDF, 209KB)
                result[15]=1
            if (800<=code)&(code<1000):
                #Injury and Poisoning (PDF, 1.2MB)
                result[16]=1
        if sum(result)==0:
            result[17]=1 #unknown or other col
        
        return result

    def _extract(self, dfs):
        # los.name = self.NAME

        

        icd9 = dfs[3].copy()
        icd9 = icd9.apply(self.__lookup_codes, axis=1, result_type='expand').rename(columns=self.col_names)
        self.name=icd9.columns.tolist()
        icd9 = icd9.droplevel('hours_in', axis=0) #drop hours in
        icd9.drop_duplicates(inplace=True)
        return icd9

    def _fit(self, dfs):
        """ Do nothing. Extractor is natively fit. """
        # warn that there is one index per hours_in. which is wrong



