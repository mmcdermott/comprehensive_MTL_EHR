import pickle, os, pandas as pd
import time
from tqdm import tqdm
import numpy as np
# import ml_toolkit.pandas_constructions as pdc
import pdb

from .utils import *
from .constants import *
from .BERT.constants import *
from .BERT.data_processor import convert_tokens_to_features
from .BERT.load_and_yield_embeddings import *
from pytorch_pretrained_bert.tokenization import BertTokenizer

import time

import psutil
import multiprocessing


def and_idxs(idxs):
    assert len(idxs) > 0, "Must provide a non-empty list of indices."
    running = idxs[0]
    for idx in idxs[1:]: running = (running & idx)
    return running

def filter_data(statics_df, *other_dfs, criteria=EXCLUSION_CRITERIA):
    static_filtered = statics_df[
        and_idxs([
            (statics_df[col] >= mn) for col, (mn, mx) in criteria.items() if mn is not None
        ] + [
            (statics_df[col] <= mx) for col, (mn, mx) in criteria.items() if mx is not None
        ])
    ]

    ids = set(static_filtered.index.get_level_values(ICUSTAY_ID))
    num_patients = len(static_filtered)
    assert len(ids) == num_patients

    max_times = (
        static_filtered[['deathtime', 'dischtime', 'outtime']].min(axis=1) -
        static_filtered[['intime', 'admittime']].max(axis=1)
    )

    #death_deltas = static_filtered.deathtime - static_filtered.intime
    #disch_deltas = static_filtered[['dischtime', 'outtime']].min(axis=1) - static_filtered.intime
    #max_times = pd.concat([death_deltas, disch_deltas], axis=1).min(axis=1)
    orig_times = pd.to_timedelta(static_filtered.los_icu, 'day')
    bad_ids = max_times[max_times <= pd.to_timedelta(static_filtered.los_icu, 'day').dt.ceil('H')]

    bad_ids_set = set(bad_ids.index.get_level_values(ICUSTAY_ID))
    true_max_hours = np.floor(bad_ids / np.timedelta64(1, 'h'))

    for l in true_max_hours.index.names:
        if l != ICUSTAY_ID: true_max_hours.index = true_max_hours.index.droplevel(l)
    true_max_hours = true_max_hours.to_dict()

    if len(true_max_hours) > 100:
        print(
            "Found an excessive (%d/%d) number of ids needing correction... \n"
            "%s..." % (
                len(true_max_hours),
                num_patients,
                ', '.join(':'.join(str(x) for x in xs) for xs in list(true_max_hours.items())[:25])
            )
        )

    ids_to_remove = []
    for k, v in true_max_hours.items():
        if v <= 0: ids_to_remove.append(k)

    if ids_to_remove:
        ids -= set(ids_to_remove)
        static_filtered = static_filtered[static_filtered.index.get_level_values(ICUSTAY_ID).isin(ids)]

        if len(ids_to_remove) > 100:
            print(
                "Found an excessive (%d/%d/%d) number of ids with negative adjusted max_len... "
                "%s..." % (
                    len(ids_to_remove),
                    len(true_max_hours),
                    num_patients,
                    ', '.join(str(x) for x in list(ids_to_remove)[:25])
                )
            )

    def should_keep_hours_in(x):
        icustay_id, hours_in = x.name[2], x.name[3] # This is going to break for notes b/c index malordered
        return not (icustay_id in bad_ids_set and hours_in >= true_max_hours[icustay_id])

    other_dfs_filtered = [df[df.index.get_level_values(ICUSTAY_ID).isin(ids)] if df is not None else None for df in other_dfs]

    # TODO(mmd): this is just ... terrible.
    for i in (0, 1, 2, 3, 5, 6):
        if len(other_dfs_filtered) > i:
            if other_dfs_filtered[i] is not None:
                other_dfs_filtered[i] = other_dfs_filtered[i][
                    other_dfs_filtered[i].apply(should_keep_hours_in, axis=1)
                ]
        else: assert i == 6, "Shouldn't be missing more than 1 df here as this function is terrible"
    # TODO(mmd): Notes!
    # def should_keep_chartdate(x):

    return [static_filtered] + other_dfs_filtered

def filter_notes(notes_df):
    return notes_df

def load_data(
    datapath           = os.path.join(DATA_DIR, DATA_FILENAME),
    fts_path           = os.path.join(DATA_DIR, FTS_FILENAME),
    folds_path         = os.path.join(DATA_DIR, FOLDS_FILENAME),
    notes_path         = os.path.join(DATA_DIR, NOTES_FILENAME),
    load_numerics      = True, load_treatments = True, load_codes = True,
    criteria           = EXCLUSION_CRITERIA,
    note_categories    = [],
    load_first_n_notes = -1,
    test_run           = False,
    eICU               = False,
    return_immediately = True,
):
    if eICU:
        assert fts_path is None
        assert notes_path is None

        numerics_treatments = pd.read_hdf(
            os.path.join(DATA_DIR, EICU_DATA_FILENAME), 'labs_vitals_treatments'
        )
        statics_df = pd.read_hdf(os.path.join(DATA_DIR, EICU_DATA_FILENAME), 'data_df')
        if return_immediately: return statics_df, numerics_treatments

        assert FOLD_IDX_LVL in numerics_treatments.index.names
        assert FOLD_IDX_LVL in statics_df.index.names

        if len(statics_df.index.names) < 2:
            print("Correcting Index!")
            statics_df.set_index(
                [c for c in numerics_treatments.index.names if c != 'hours_in'], inplace=True
            )

        try:
            # TODO(mmd): port this to filter_data function.
            static_filtered = statics_df[
                and_idxs([
                    (statics_df[col] >= mn) for col, (mn, mx) in criteria.items() if mn is not None
                ] + [
                    (statics_df[col] <= mx) for col, (mn, mx) in criteria.items() if mx is not None
                ])
            ]
            numerics_treatments_filtered = numerics_treatments[
                numerics_treatments.index.get_level_values(ICUSTAY_ID).isin(
                    static_filtered.index.get_level_values(ICUSTAY_ID)
                )
            ]

            return static_filtered, numerics_treatments_filtered
        except Exception as e:
            print(e)
            return statics_df, numerics_treatments

    if test_run:
        datapath = os.path.join(DATA_DIR, TEST_DATA_FILENAME)
        notes_path = os.path.join(DATA_DIR, TEST_NOTES_FILENAME)
        if fts_path is not None: fts_path = os.path.join(DATA_DIR, TEST_FTS_FILENAME)

    time_old=time.time()

    #30s
    print(datapath)
    with pd.HDFStore(datapath) as hdf:
        print(hdf.keys())

    dfs = [pd.read_hdf(datapath, STATICS)]
    if load_numerics:   dfs.append(pd.read_hdf(datapath, NUMERICS))
    else: dfs.append(None)
    if load_treatments: dfs.append(pd.read_hdf(datapath, TREATMENTS))
    else: dfs.append(None)
    if load_codes:      dfs.append(pd.read_hdf(datapath, CODES))
    else: dfs.append(None)

    print("\t load numerics treatments code: {}".format(time.time()-time_old))
    time_old=time.time()

    # 17 s
    if notes_path is not None:
        notes = pd.read_hdf(notes_path)
        if note_categories:
            notes = notes[notes.index.get_level_values('category').isin(note_categories)]
        notes = notes[:load_first_n_notes]
        dfs.append(notes)
    else:
        dfs.append(None)

    print("\t load notes: {}".format(time.time()-time_old))
    time_old=time.time()

    # 0.5s
    if fts_path is not None: dfs.extend([pd.read_hdf(fts_path, 'general'), pd.read_hdf(fts_path, 'granular')])
    else: dfs.extend([None, None])

    print("\t load fts_path: {}".format(time.time()-time_old))
    time_old=time.time()

    #2.7s
    dfs = filter_data(*dfs, criteria=criteria)

    print("\t filter_data: {}".format(time.time()-time_old))
    time_old=time.time()

    #120 seconds
    cpus = psutil.cpu_count()# get a few available cpus



    if folds_path is not None:
        with open(folds_path, mode='rb') as f: folds = pickle.load(f)

        new_folds={}
        for i in range(len(folds)):
            new_folds.update({v: i for v in folds[i]}) # structure asserts that there are no repeats

        # test_time_old=time.time()
        for df in dfs:
            if df is None: continue
            if FOLD_IDX_LVL in df.index.names: df.reset_index(level=FOLD_IDX_LVL, inplace=True)

            df.loc[:, FOLD_IDX_LVL] = [new_folds[s_i] for s_i in df.index.get_level_values(SUBJECT_ID)]
            df.set_index(FOLD_IDX_LVL, append=True, inplace=True)
        # print("\t\t add_folds reindex: {}".format(time.time()-test_time_old))

    print("\t add_folds: {}".format(time.time()-time_old))


    return dfs

def add_folds_to_df(df, folds, idx_col=SUBJECT_ID):
    any_overlap = False
    s_prev = set(folds[0])
    for s in folds[1:]:
        if len(s_prev.intersection(s)) > 0:
            any_overlap = True
            break
        s_prev.update(s)
    assert not any_overlap, "Folds should be non-overlapping!"


    total_set = set(df.index.get_level_values(idx_col))
    assert total_set.issubset(s_prev), "Folds should cover entirety of dataframe"

    fold_num = [next(i for i, s in enumerate(folds) if v in s) for v in df.index.get_level_values(idx_col)]
    df.loc[:, FOLD_IDX_LVL] = fold_num
    df.set_index(FOLD_IDX_LVL, append=True, inplace=True)

def to_seq(x):
    try: return list(x)
    except TypeError as e: return [x]

def get_splits(df, held_out_folds, tuning_evaluation_folds, K = 10):
    if type(held_out_folds) is not set: held_out_folds = set(to_seq(held_out_folds))
    if type(tuning_evaluation_folds) is not set: tuning_evaluation_folds = set(to_seq(tuning_evaluation_folds))

    train_folds = set(f for f in range(K) if f not in held_out_folds.union(tuning_evaluation_folds))

    folds = df.index.get_level_values(FOLD_IDX_LVL)
    fold_sets = {'train': train_folds, 'tuning': tuning_evaluation_folds, 'held_out': held_out_folds}
    return {l: df[folds.isin(s)] for l, s in fold_sets.items()}

def prepare_continuous_labels(numerics, statics):
    """
    Replace with Matthews code
    """
    timeseries_outcomes=numerics.copy()
    idx=pd.IndexSlice

    timeseries_outcomes.columns=[col[0]+"_"+col[1] for col in timeseries_outcomes.columns.tolist()]

    cols=['mort_24h', 'mort_48h', 'disch_24h', 'disch_48h']

    try:
        statics.drop(columns =['max_hours'])
    except:
        pass

    if 'max_hours' not in statics.columns.tolist():
        intime=statics.loc[:, ['intime', 'admittime']].max(axis=1)
        outtime=statics.loc[:, ['outtime', 'dischtime']].min(axis=1)
        statics.loc[:, 'max_hours']=(outtime-intime)/pd.Timedelta(hours=1)

    allowed_patients = set(statics[statics.max_hours > 5].index.get_level_values(SUBJECT_ID))

    for col in cols: timeseries_outcomes[col]=np.nan

    #delete the patients and other columns that we don't want
    timeseries_outcomes=timeseries_outcomes.loc[
        timeseries_outcomes.index.get_level_values(SUBJECT_ID).isin(allowed_patients), cols
    ]

    statics_temp=statics.copy()
    # join statics_temp and numerics so that statics variables are available at every hour
    timeseries_outcomes=timeseries_outcomes.join(statics_temp, on=['subject_id', 'hadm_id', 'icustay_id', 'Fold'])

    # patients who survive
    timeseries_outcomes.loc[timeseries_outcomes.loc[:, 'mort_icu']==0, 'mort_24h']=0
    timeseries_outcomes.loc[timeseries_outcomes.loc[:, 'mort_icu']==0, 'mort_48h']=0

    # fill start times with 0
    timeseries_outcomes.loc[idx[:,:,:,0,:],cols]=0

    # for 24 hour outcomes:
    # only keep labels that are greater than or equal to max_hours-24
    cond_1=(timeseries_outcomes.loc[:, 'max_hours']-timeseries_outcomes.index.get_level_values('hours_in'))<=24
    cond_2=timeseries_outcomes.loc[:, 'mort_icu']==1
    #if patient dies in hospital within 24 hours of icu discharge
    cond_3=(pd.to_datetime(timeseries_outcomes.loc[:, 'dischtime'], format='%Y-%m-%d %H:%M:%S')-pd.to_datetime(timeseries_outcomes.loc[:, 'outtime'], format='%Y-%m-%d %H:%M:%S')) / np.timedelta64(1, 'h')<=24 # this raises class prob from 0.022 to 0.056


    timeseries_outcomes.loc[cond_1&(cond_2|cond_3), 'mort_24h']=1

    cond_2=timeseries_outcomes.loc[:, 'mort_icu']==0
    timeseries_outcomes.loc[cond_1&cond_2, 'disch_24h']=1


    # for 48 hour outcomes:
    # only keep labels that are greater than or equal to max_hours-48
    cond_1=(timeseries_outcomes.loc[:, 'max_hours']-timeseries_outcomes.index.get_level_values('hours_in'))<=48
    cond_2=timeseries_outcomes.loc[:, 'mort_icu']==1
    cond_3=(pd.to_datetime(timeseries_outcomes.loc[:, 'dischtime'], format='%Y-%m-%d %H:%M:%S')-pd.to_datetime(timeseries_outcomes.loc[:, 'outtime'], format='%Y-%m-%d %H:%M:%S')) / np.timedelta64(1, 'h')<=48 # this raises class prob from 0.04 to 0.18

    timeseries_outcomes.loc[cond_1&(cond_2|cond_3), 'mort_48h']=1

    cond_2=timeseries_outcomes.loc[:, 'mort_icu']==0
    timeseries_outcomes.loc[cond_1&cond_2, 'disch_48h']=1

    timeseries_outcomes.loc[:, cols]=timeseries_outcomes.loc[:, cols].fillna(value=0, axis=1)

    return timeseries_outcomes, allowed_patients

def simple_impute(df_input, tqdm=tqdm):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    df_in=df_input.copy()

    #masked data
    masked_df=pd.notna(df_in)
    masked_df=masked_df.apply(pd.to_numeric)

    #we can fill in the missing values with any number now (they get multiplied out to be zero)
    # df_in=df_in.fillna(0)

    #time since last measurement
    # time_old=time.time()
    is_absent = (1 - masked_df)
    hours_of_absence = is_absent.groupby(ID_COLS).cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_df = time_since_measured.fillna(0)

    # print(time.time()-time_old)
    # time_old=time.time()

    # time_index=None
    # if not(time_index): time_index='hours_in'
    # index_of_hours=list(df_in.index.names).index(time_index)
    # time_in=np.asarray([item[index_of_hours] for item in df_in.index.tolist()])
    # time_df=df_in.copy()

    # ##try multiprocessing
    # # this implementation is exact with one indexing since observation except where the patient enters the icu like in the original paper takes about 50 seconds per fold
    # with Pool(os.cpu_count()-2) as pool:
    #    par_func = partial(time_since_last, time_in)
    #    outs     = pool.map(par_func, [masked_df[col] for col in  time_df.columns.tolist()])
    # time_df=pd.concat(outs, axis=1)
    # print(time.time()-time_old)

    # print(time_df2.head(10))
    # print(time_df.head(10))

    #time_df=time_df.fillna(0)

    # for col in tqdm(time_df.columns.tolist()):
    #     mask=masked_df[col].values
    #     time_df[col]=time_since_last(time_in, mask)
    # time_df=time_df.fillna(0)

    #last observed value
    X_last_obsv_df=df_input.copy()

    # print(X_last_obsv_df.head(5))

    # do the mean imputation for only the first hour
    columns=X_last_obsv_df.columns.tolist()
    #Only do means where the column isn't the outcome
    subset_data=X_last_obsv_df.loc[(slice(None),slice(None), slice(None), 0), columns]

    # (not sure how original paper did it, possibly just fill with zeros???)
    subset_data=subset_data.fillna(0)

    #replace first hour data with the imputed first hour data
    replace_index=subset_data.index.tolist()

    # print(np.sum(np.sum(np.isnan(X_last_obsv_df.values))))
    X_last_obsv_df.loc[(slice(None),slice(None), slice(None), 0), columns] = subset_data.values
    # X_last_obsv_df=X_last_obsv_df.fillna(0)

    # now it is safe for forward fill
    #forward fill the rest of the sorted data
    # X_last_obsv_df=X_last_obsv_df.loc[subject_index,:]
    # print(np.sum(np.sum(np.isnan(X_last_obsv_df.values))))
    X_last_obsv_df=X_last_obsv_df.fillna(method='ffill')
    # print(np.sum(np.sum(np.isnan(X_last_obsv_df.values))))

    # diabetic spocglucresult is completely missing
    # for col in X_last_obsv_df.columns.tolist():
    #     print(col, np.sum(np.sum(np.isnan(X_last_obsv_df.loc[:, col].values))))

    X_last_obsv_df=X_last_obsv_df.fillna(0)
    # print(np.sum(np.sum(np.isnan(X_last_obsv_df.values))))
    # input()


    return df_in, X_last_obsv_df, masked_df, time_df

def convert_notes_to_features_eff(notes, input_seq_length=None, seq_len=None, tokenizer=None):
    note_hour_idx = pd.notnull(notes).any(axis=1).to_numpy().nonzero()[0]

    note_ids = np.empty((seq_len, input_seq_length,))
    note_ids[:] = np.nan

    note_masks = np.empty((seq_len, input_seq_length,))
    note_masks[:] = np.nan

    note_segment_ids = np.empty((seq_len, input_seq_length,))
    note_segment_ids[:] = np.nan

    for idx in note_hour_idx:
        tokens = notes.iloc[idx].tokens
        features = convert_tokens_to_features(tokens, None, tokenizer, input_seq_length)
        note_ids[idx] = features.input_ids
        note_masks[idx] = features.input_mask
        note_segment_ids[idx] = features.segment_ids

    note_hour_num = len(note_hour_idx)
    note_hour_idx = np.pad(note_hour_idx, (0, seq_len-note_hour_num), 'constant', constant_values=(-1))

    return (note_ids, note_masks, note_segment_ids, note_hour_idx, note_hour_num)


def convert_notes_to_features_bret(notes, input_seq_length=None, seq_len=None, tokenizer=None):
    """
    TODO: Add comments
    """

    note_hour_idx = pd.notnull(notes).any(axis=1).to_numpy().nonzero()[0]
    print('note_hour_index ', set(note_hour_idx))
    cumsum_notes=np.cumsum(note_hour_idx)*note_hour_idx

    note_ids=np.zeros((len(notes), input_seq_length))
    note_masks=np.zeros((len(notes), input_seq_length))
    note_segment_ids=np.zeros((len(notes), input_seq_length))


    def wrap(x):
        features = convert_tokens_to_features(x['tokens'], None, tokenizer, input_seq_length)
        return features.input_ids, features.input_mask, features.segment_ids


    res_df = notes.iloc[note_hour_idx].apply(wrap, axis=1, result_type='expand')
    print(res_df.values.shape)
    print(np.asarray(res_df.loc[:, 0].values).shape)
    print(np.asarray(res_df.loc[:, 1].values).shape)
    print(np.asarray(res_df.loc[:, 2].values).shape)

    note_ids[note_hour_idx, :] = np.asarray(res_df.loc[:, 0].values)
    note_masks[note_hour_idx, :] = np.asarray(res_df.loc[:, 1].values)
    note_segment_ids[note_hour_idx, :] = np.asarray(res_df.loc[:, 2].values)



    note_hour_num = len(note_hour_idx)
    note_hour_idx = np.pad(note_hour_idx, (0, seq_len-note_hour_num), 'constant', constant_values=(-1))

    return (note_ids, note_masks, note_segment_ids, note_hour_idx, note_hour_num)


def convert_notes_to_features(notes, input_seq_length=None, seq_len=None, tokenizer=None):
    note_ids = np.empty((0, input_seq_length), int)
    note_masks = np.empty((0, input_seq_length), int)
    note_segment_ids = np.empty((0, input_seq_length), int)

    for index, row in notes.iterrows():
        features = convert_tokens_to_features(row.tokens, None, tokenizer, input_seq_length)
        note_ids = np.append(note_ids, np.asarray([features.input_ids]), axis=0)
        note_masks = np.append(note_masks, np.asarray([features.input_mask]), axis=0)
        note_segment_ids = np.append(note_segment_ids, np.asarray([features.segment_ids]), axis=0)

    padding_number = seq_len - len(notes)

    padding_note_ids = np.tile(note_ids[0], (padding_number, 1))
    padding_note_masks = np.tile(note_masks[0], (padding_number, 1))
    padding_segment_ids = np.tile(note_segment_ids[0], (padding_number, 1))

    note_ids = np.concatenate((note_ids, padding_note_ids), axis=0)
    note_masks = np.concatenate((note_masks, padding_note_masks), axis=0)
    note_segment_ids = np.concatenate((note_segment_ids, padding_segment_ids), axis=0)

    return (note_ids, note_masks, note_segment_ids)

# TODO(mmd): Hook up max_seq_length here!
def process_and_tokenize(note, tokenizer, max_seq_length=512):
    note = preprocess_sentence(note)
    tokens = tokenizer.tokenize(note)
    
    seq_len = len(tokens)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    return tokens

def tokenize_and_impute_notes(notes, tokenizer):
    if notes is None: return None
    notes['tokens'] = np.nan
    notes.loc[~notes['text'].isna(), 'tokens']=notes.loc[~notes['text'].isna(), 'text'].apply(process_and_tokenize, tokenizer=tokenizer)
    _ , note_last_obsv_df, note_masked_df, note_time_df=simple_impute(notes, tqdm)
    note_last_obsv_df.tokens = note_last_obsv_df.tokens.replace({0: ['[CLS]']})
    return note_last_obsv_df

def tokenize_notes(notes, tokenizer):
    if notes is None or 'text' not in notes: return notes

    notes['tokens'] = np.nan
    notes.loc[~notes['text'].isna(), 'tokens'] = notes.loc[~notes['text'].isna(), 'text'].apply(
        process_and_tokenize, tokenizer=tokenizer
    )
    return notes

def generate_input_embeddings_for_batch(model, note_ids, note_masks, note_segment_ids, device):

    for i, pt_note_ids in enumerate(note_ids):
        pt_note_masks = note_masks[i]
        pt_note_segment_ids = note_segment_ids[i]

        pt_input_embeddings, _, _ = generate_input_embeddings(
            model, 
            pt_note_ids, pt_note_masks, pt_note_segment_ids,
            device, disable_tqdm = True
        )

        pt_input_embeddings = torch.tensor(pt_input_embeddings)

        if i == 0:
            all_pt_outputs = pt_input_embeddings[None, :]
        else:
            all_pt_outputs = torch.cat([all_pt_outputs, pt_input_embeddings[None, :]], dim=0)

    return all_pt_outputs

# TODO(mmd): eliminate this function (it is still used)
def fit_on_device(data, single_batch, device):
    data = data.squeeze()
    if single_batch:
        data = data.unsqueeze(0)
    return data.to(device).float()

def reshape_note_bert_input(
    note_ids, note_masks, note_segment_ids, note_hours_idx, note_hours_num
):
    combined_note_ids = None
    combined_note_masks = None
    combined_segment_ids = None
    
    for batch_idx, num_hours in enumerate(note_hours_num):
        for j in range(0, num_hours):
            hour_idx = note_hours_idx[batch_idx][j]
            
            note_ids_row = note_ids[batch_idx][hour_idx]
            note_masks_row = note_masks[batch_idx][hour_idx]
            note_segment_ids_row = note_segment_ids[batch_idx][hour_idx]
                                    
            if not combined_note_ids is not None:
                combined_note_ids = note_ids_row.unsqueeze(0)
                combined_note_masks = note_masks_row.unsqueeze(0)
                combined_segment_ids = note_segment_ids_row.unsqueeze(0)
            else:
                combined_note_ids = torch.cat(
                    [combined_note_ids, note_ids_row.unsqueeze(0)], dim=0
                )
                combined_note_masks = torch.cat(
                    [combined_note_masks, note_masks_row.unsqueeze(0)], dim=0
                )
                combined_segment_ids = torch.cat(
                    [combined_segment_ids, note_segment_ids_row.unsqueeze(0)], dim=0
                )

    return (combined_note_ids, combined_note_masks, combined_segment_ids)

def impute_interval(first_hour, last_hour, tensor):
    if int(last_hour) == 0:
        return torch.tensor([])
    else:
        return tensor.unsqueeze(0).repeat(last_hour - first_hour, 1)

def reshape_note_bert_output(
    pooled_output, note_hours_num, note_hours_idx, batch_size, max_seq_length, embedding_dim, impute=True
):

    if not not_none(pooled_output):
        return torch.zeros(batch_size, max_seq_length, embedding_dim)

    if impute:
        idx = 0
        input_embeddings = None
        zeros = torch.zeros(embedding_dim)

        for batch_idx, num_hours in enumerate(note_hours_num):
            hour_intervals = note_hours_idx[batch_idx][:num_hours]

            if num_hours == 0:
                batch_embeddings = torch.zeros(max_seq_length, embedding_dim)

            for pos, hour in enumerate(hour_intervals):
                if pos == 0:
                    batch_embeddings = impute_interval(0, hour, zeros)

                if pos+1 == len(hour_intervals):
                    next_interval = max_seq_length
                else:
                    next_interval = hour_intervals[pos+1]

                interval_embeddings = impute_interval(hour, next_interval, pooled_output[idx])
                batch_embeddings = torch.cat([batch_embeddings, interval_embeddings], dim=0)

                idx += 1

            if not input_embeddings is not None:
                input_embeddings = batch_embeddings.unsqueeze(0)
            else:
                input_embeddings = torch.cat([input_embeddings, batch_embeddings.unsqueeze(0)], dim=0)
    else:
        input_embeddings = torch.zeros(batch_size, max_seq_length, embedding_dim)

        idx = 0
        for batch_idx, num_hours in enumerate(note_hours_num):
            for j in range(0, num_hours):
                hour_idx = note_hours_idx[batch_idx][j]
                input_embeddings[batch_idx][hour_idx] = pooled_output[idx]
                idx += 1

    return input_embeddings

def preselect_notes(note_hours_idx, note_hours_num, n_gpu):
    # Currently about 7 BERT can be ran per gpu
    max_number_notes = n_gpu * 6

    if sum(note_hours_num) > max_number_notes:
        diff = sum(note_hours_num) - max_number_notes
        while diff > 0:
            max_batch = torch.max(note_hours_num, 0)[1]
            note_hours_num[max_batch] -= 1
            note_hours_idx[max_batch] = torch.cat([note_hours_idx[max_batch][1:], torch.tensor([-1])], dim=0)

            diff = diff - 1

    return (note_hours_idx, note_hours_num)
