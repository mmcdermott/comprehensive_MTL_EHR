import os, random, sys, torch, numpy as np, pandas as pd, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
idx=pd.IndexSlice

from ..constants import *

from tqdm import tqdm

def one_hot_encode(
    cols_to_encode,
    df,
    vocab,
    inplace=False,
):
    # TODO(mmd): Write unit tests for this function!
    if not inplace: df = df.copy()

    for col in cols_to_encode:
        max_idx = len(vocab[col])
        one_hot_c = np.zeros((len(df), max_idx))
        one_hot_c[list(range(len(df))), df[col]] = 1
        new_cols = [(col, t) for t in vocab[col]]
        df[new_cols] = pd.DataFrame(one_hot_c, index=df.index)

        # Now remove the old
        df.drop(columns=col, inplace=True)
    return df

#def one_hot_encode(cols_to_encode, df, vocab, inplace=False):
#    """
#    """
#    # TODO(mmd): Write unit tests for this function!
#    if not inplace: df = df.copy()
#
#
#    # i.e. comfort measures ordered
#
#    # for each encoded column
#    for col in tqdm(cols_to_encode):
#        print("\n", col)
#        print( vocab[col])
#        print(list(set(df[col].values.tolist())))
#        # find the number of words for that column
#        max_idx = len(vocab[col])
#        # create an empty df for the number of words in that column
#        one_hot_c = np.zeros((len(df), max_idx))
#        # why not just use [:, df[col]] this line is unintuitive
#        one_hot_c[list(range(len(df))), df[col]] = 1
#        # create a new column for eacht in vocab
#        new_cols = [(col, t) for t in vocab[col]]
#
#        new_df=pd.get_dummies(df[col])
#        new_df.index=df.index # set one hot df index to be same as original index
#        # df[new_cols]=new_df
#
#        print(one_hot_c[:5, :])
#
#        for i, col in enumerate(new_cols):
#            df[col]=one_hot_c[:, i].ravel()
#        # df.loc[:, new_cols] = one_hot_c # only assign with .loc to supres the assign on copy warning
#
#        # Now remove the old
#        df.drop(columns=col, inplace=True)
#    return df

def add_time_since_measured(
    df, init_time_since_measured=100, max_time_since_measured=100, hour_aggregation = 1,
):
    idx = pd.IndexSlice
    df = df.copy()

    is_absent = (df.loc[:, idx[:, 'count']] == 0).astype(int)
    hours_of_absence = is_absent.groupby(ID_COLS).cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].groupby(ID_COLS).fillna(method='ffill')
    time_since_measured.fillna(init_time_since_measured, inplace=True)
    time_since_measured[time_since_measured > max_time_since_measured] = max_time_since_measured

    # Somehow, prior to the rename above and here, the columns index lost its name, so we use level=1 here.
    # But note that this is more brittle.
    time_since_measured.rename(
        columns={'count': 'time_since_measured'}, level=1, inplace=True
    )

    if hour_aggregation !=1:
        time_since_measured.loc[:, :] = time_since_measured.values // hour_aggregation * int(hour_aggregation)

    df_out = pd.concat((df, time_since_measured), axis=1)
    df_out.sort_index(axis=1, inplace=True)
    return df_out

def FullyConnectedNet(
    in_dim, out_dim,
    hidden_sizes     = [],
    inner_activation = nn.ReLU,
    out_activation   = None,
    dropout_prob     = 0.1,
    dropout_layer    = nn.Dropout,
):
    layers = []
    if dropout is not None: layers.append(dropout_layer(p=dropout_prob))
    for hs in hidden_sizes: 
        layers.extend([nn.Linear(in_dim, hs), activation()])
        in_dim = hs
    layers.append(nn.Linear(in_dim, out_dim))
    if out_activation is not None: layers.append(out_activation())

    return nn.Sequential(*layers)

def get_loss_if_labeled(loss_fct):
    def f(out, labels, **kwargs): return (out, None) if labels is None else loss_fct(out, labels, **kwargs)
    return f

# TODO(mmd): This looks like it is doing all the same thing....
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)


# TODO(mmd): This is probably not needed--can just omit heads or data...
def ablate(all_outputs, ablations):
    """
   all_outputs (dict): {
                'will_be_measured': (will_be_measured_pred, will_be_measured, will_be_measured_loss),
                'next_timepoint':   (next_values_pred, next_values, reconstruction_loss),
                'mort_icu':         (mort_icu_pred, mort_icu, mort_icu_loss),
                'mort_24':          (mort_24_pred, mort_24, mort_24h_loss),
                'disch_24':         (disch_24_pred, disch_24, disch_24h_loss),
                'los_left':         (los_left_pred, los_left, LOS_left_loss),
                'fts_decoding':     (fts_logits, fts_labels, fts_loss),
            }

    args.ablate (list): a list containing they keys of the all_outputs of which to ablate
    """
    losses=all_outputs.keys()

    losses=[l for l in losses if l not in ablations]

    # total_loss=Variable(torch.tensor(0)).float()
    # for l in losses:
    #     # print([item.shape for item in all_outputs[l]])
    #     # label, target, loss=all_outputs[l]
    #     # print(label.shape)
    #     # print(target.shape)
    #     # print(loss.shape)
    #     # print(loss)
    #     total_loss+=all_outputs[l][-1].sum() # sum necessary for multi gpu jobs

    return torch.stack([all_outputs[l][-1].sum() for l in losses]).sum()
