import hashlib, pickle, numpy as np#, ml_toolkit.pandas_constructions as pdc

import pandas as pd
import re

def get_index_levels(df, levels, make_objects_categories=True):
    #todo: make this package a requirement so env doesn't break
    #from ml_toolkit
    df_2 = pd.DataFrame(index=df.index)
    for level in levels: df_2[level] = df_2.index.get_level_values(level)
    if make_objects_categories:
        for column in df_2.columns:
            if df_2[column].dtype == object: df_2[column] = df_2[column].astype('category')
    return df_2

def freq_or_count(x, type_at_1=int):
    #assert type(x) in (float, int), "x must be either a float or an integer."
    x = float(x)
    assert x >= 0, "x must be nonnegative"
    if x < 1: return x
    elif x > 1 and int(x) == x: return int(x)
    elif x == 1: return type_at_1(x)
    else: raise NotImplementedError("x cannot be coerced to frequency or count!")

def __nested_sorted_repr(c):
    if type(c) in (set, frozenset): return tuple(sorted(c))
    if type(c) is dict: return tuple(sorted([(k, __nested_sorted_repr(v)) for k, v in c.items()]))
    if type(c) in (tuple, list): return tuple([__nested_sorted_repr(e) for e in c])
    else: return c

def hash_dict(d): return hash_repr(__nested_sorted_repr(d))
def hash_repr(tup):
    m = hashlib.new('md5')
    m.update(repr(tup).encode('utf-8'))
    return m.hexdigest()

def pad(l, max_len, pad_value = 0):
    # only df is tested.
    if type(l) is list: return l + ([pad_value]*(max_len - len(l)))
    elif type(l) is np.ndarray:
        try: return l.resize((max_len - len(l), pad_value))
        except ValueError as e: pass

        return np.concatenate((l, pad_value * np.ones([max_len - l.shape[0]] + list(l.shape[1:]))))
    raise NotImplementedError("Only supports lists or numpy arrays at present.")

# TODO(mmd): Consider moving to pdc
def add_id_col(df, id_idxs, id_col_name):
    assert len(id_idxs) > 1, "Too few id idx columns."

    df[id_col_name] = [hash_repr(tuple(str(x) for x in a)) for a in get_index_levels(df, id_idxs).values]
    df.set_index(id_col_name, append=True, inplace=True)

def depickle(filepath):
    with open(filepath, mode='rb') as f: return pickle.load(f)
def read_txt(filepath):
    with open(filepath, mode='r') as f: return f.read()

# For debugging pandas apply/transform ops.
def print_and_raise(*args, **kwargs):
    print(args, kwargs)
    raise NotImplementedError

def zip_dicts_assert(*dcts):
    d0 = dcts[0]
    s0 = set(d0.keys())
    for d in dcts[1:]:
        s1 = set(d.keys())
        assert d0.keys() == d.keys(), f"Keys Disagree! d0 - d1 = {s0 - s1}, d1 - d0 = {s1 - s0}"

    for i in set(dcts[0]).intersection(*dcts[1:]): yield (i,) + tuple(d[i] for d in dcts)

def zip_dicts(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]): yield (i,) + tuple(d[i] for d in dcts)

def zip_dicts_union(*dcts):
    keys = set(dcts[0].keys())
    for d in dcts[1:]: keys.update(d.keys())

    for k in keys: yield (k,) + tuple(d[k] if k in d else np.NaN for d in dcts)

def tokenize_str(str_):
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()

def not_none(stuff):
    if not stuff is not None:
        return False
    else:
        return True

def is_none(stuff):
    not not_none
