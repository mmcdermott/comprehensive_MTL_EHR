# TODO: Use Dask--integrate processor and converter (maybe tensor dataset too). Do as Mapreduce proper.
import sys

import gc, random, numpy as np
from multiprocessing import Pool
from contextlib import closing


from latent_patient_trajectories.utils import *
from latent_patient_trajectories.data_utils import *
from latent_patient_trajectories.constants import *

from latent_patient_trajectories.BERT.constants import *
from latent_patient_trajectories.BERT.data_processor import *
from latent_patient_trajectories.BERT.model import *


def flatten(arr):
  if type(arr) is np.ndarray: return np.reshape(arr, [len(arr), -1])
  elif type(arr) is list:
    r = []
    for l in arr: r += l
    return r
  raise NotImplementedError

def I(x): return x

class ContinuousInputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_sequence_orig, input_sequence_masked, input_mask, segment_ids, el_was_masked, labels
    ):
        self.input_sequence_orig   = input_sequence_orig
        self.input_sequence_masked = input_sequence_masked
        self.input_mask            = input_mask
        self.segment_ids           = segment_ids
        self.el_was_masked         = el_was_masked
        self.whole_sequence_labels = labels

def _mask_seq(
    seq, random_token_seq, can_mask, mask_prob, random_token_mask_prob, random_token_replace_prob,
):
    if seq is None: return None, None, None

    mask_ps = np.random.uniform(low=0, high=1, size=(len(seq), ))
    mask_vector = CONTROL_VECTOR_PREFIXES[MASK](np.zeros((1, seq.shape[1])))

    was_masked = [0] * len(seq)
    seq_orig   = seq.copy()

    random_seq_i = 0 # we track this separately because we don't have enough random seqs given ctrl tokens.
    for i, can_mask in enumerate(can_mask):
        if not can_mask: continue

        random_seq_i += 1 # we increment here and subtract below so we don't need to increment in the if.
        p = mask_ps[i]
        if p >= mask_prob: continue

        was_masked[i] = 1

        p /= mask_prob

        if p < random_token_mask_prob: seq[i, :] = mask_vector
        elif p < random_token_mask_prob + random_token_replace_prob:
            seq[i, :] = random_token_seq[random_seq_i - 1, :]

    return seq_orig, seq, was_masked

def _truncate_seq_pair(seq_a, seq_b, max_length):
    """Truncates a sequence pair in place to the maximum length. You don't seed this function as otherwise it
    would always favor one sequence."""

    total_length = len(seq_a) + len(seq_b)
    if total_length <= max_length: return seq_a, seq_b

    elif len(seq_b) <= max_length // 2:   seq_a = seq_a[:max_length - len(seq_b)]
    elif len(seq_a) <= max_length // 2: seq_b = seq_b[:max_length - len(seq_a)]
    elif max_length % 2 == 0:
        seq_a = seq_a[:max_length // 2]
        seq_b = seq_b[:max_length // 2]
    else:
        if random.random() > 0.5:
            seq_a = seq_a[:(max_length // 2) + 1]
            seq_b = seq_b[:max_length // 2]
        else:
            seq_a = seq_a[:max_length // 2]
            seq_b = seq_b[:(max_length // 2) + 1]
    return seq_a, seq_b

def __convert_example(args):
    # TODO(mmd): modify to work with just values.
    example, max_seq_length, random_seq, mask_select_prob, token_mask_prob, token_replace_prob = args
    mask = lambda seq, random_seq, can_mask: _mask_seq(
        seq, random_seq, can_mask, mask_select_prob, token_mask_prob, token_replace_prob
    )

    seq_a = example.seq_a
    seq_dim = seq_a.shape[1]

    seq_b = None
    if example.seq_b is not None:
        seq_b = example.seq_b
        seq_len = len(seq_a) + len(seq_b)

        # Modifies `seq_a` and `seq_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        seq_a, seq_b = _truncate_seq_pair(seq_a, seq_b, max_seq_length - 3)
    else:
        seq_len = len(seq_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(seq_a) > max_seq_length - 2: seq_a = seq_a[:(max_seq_length - 2)]

    # TODO: Update comment.
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    # First, we pad the sequence vectors with zeros so we can introduce our auxiliary tokens. Then, we add
    # the CLS and SEP tokens into the sequence.

    augmented_seq_dim = seq_dim + NUM_CONTROL_TOKENS

    can_mask    = [False] + [True] * len(seq_a) + [False]
    seq_a = np.concatenate((np.zeros((len(seq_a), NUM_CONTROL_TOKENS)), seq_a), axis=1)
    cls_vector = CONTROL_VECTOR_PREFIXES[CLS](np.zeros((1, augmented_seq_dim)))
    sep_vector = CONTROL_VECTOR_PREFIXES[SEP](np.zeros((1, augmented_seq_dim)))

    sequence = np.concatenate((cls_vector, seq_a, sep_vector), axis=0)
    segment_ids = [0] * sequence.shape[0]

    if seq_b is not None:
        # Adding the other [SEP] token.
        can_mask += [True] * len(seq_b) + [False]
        seq_b = np.concatenate((np.zeros((len(seq_b), NUM_CONTROL_TOKENS)), seq_b), axis=1)
        sequence = np.concatenate((sequence, seq_b, sep_vector), axis=0)
        segment_ids += [1] * (seq_b.shape[0] + 1)

    # Masking for masked reconstruction loss.

    seq_orig, seq_masked, el_was_masked = mask(sequence, random_seq, can_mask)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(sequence)

    # Zero-pad up to the sequence length.

    padding = [0] * (max_seq_length - len(sequence))
    input_mask += padding
    segment_ids += padding
    el_was_masked += padding

    seq_orig   = np.concatenate((seq_orig, np.zeros((len(padding), augmented_seq_dim))), axis=0)
    seq_masked = np.concatenate((seq_masked, np.zeros((len(padding), augmented_seq_dim))), axis=0)

    # TODO(mmd): Continue from here.

    assert len(seq_orig) == max_seq_length
    assert len(seq_masked) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(el_was_masked) == max_seq_length

    return ContinuousInputFeatures(
        input_sequence_orig=seq_orig,
        input_sequence_masked=seq_masked,
        input_mask=input_mask,
        segment_ids=segment_ids,
        el_was_masked=el_was_masked,
        labels=example.whole_sequence_labels,
    )

def convert_examples_to_features(
    examples, max_seq_length, seed=1, mask_select_prob=0.15, token_mask_prob=0.8,
    token_replace_prob=0.1, tqdm=None, multiprocessing_pool_size=1, shuffle=False
):
    """Loads a data file into a list of `InputBatch`s."""
    random.seed(seed)
    np.random.seed(seed)

    all_seqs = []
    for example in examples:
        all_seqs.append(example.seq_a)
        if example.seq_b is not None: all_seqs.append(example.seq_b)

    all_seqs = np.concatenate(all_seqs, axis=0)
    if shuffle: all_seqs = np.random.permutation(all_seqs)
    all_seqs = np.concatenate((np.zeros((len(all_seqs), NUM_CONTROL_TOKENS)), all_seqs), axis=1)

    seen_so_far = 0
    random_seqs = []
    for example in examples:
        total_seq_len = len(example.seq_a) + (0 if example.seq_b is None else len(example.seq_b))
        random_seqs.append(all_seqs[seen_so_far : seen_so_far + total_seq_len])
        seen_so_far += total_seq_len

    zipped = zip(examples, random_seqs)
    inputs = (
        (e, max_seq_length, rs, mask_select_prob, token_mask_prob, token_replace_prob) for e, rs in zipped
    )

    if multiprocessing_pool_size > 1:
        with Pool(multiprocessing_pool_size) as p:
            if tqdm is not None: features = list(tqdm(p.imap(__convert_example, inputs), total=len(examples)))
            else: features = p.map(__convert_example, inputs)
    else:
        if tqdm is not None: inputs = tqdm(inputs, total=len(examples))
        features = [__convert_example(i) for i in inputs]

    return features

def to_set(x): return set(x) if type(x) in [list, tuple, set] else set([x])

class ContinuousInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, seq_a, seq_b, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            seq_a: ?. The first sequence.
            seq_b: ?. The second sequence.
            Only must be specified for sequence pair tasks.
        """
        self.guid = guid
        self.seq_a = seq_a
        self.seq_b = seq_b
        self.whole_sequence_labels = labels

def any_subsequent_indices_ordered(l):
    for i in range(len(l)-1):
        if l[i] == (l[i+1] - 1): return True
    return False

class ContinuousBertGetEmbeddingsProcessor():
    SEQUENCE_ID = 'sequence_id'
    STREAM_ID   = 'sequence_stream_id'

    def __init__(
        self, sequence_id_idxs,
        tuning_folds=[], held_out_folds=[],
        seed=1, tqdm = None, multiprocessing_pool_size=1,
        chunksize = None,
    ):
        self.tuning_folds, self.held_out_folds = to_set(tuning_folds), to_set(held_out_folds)
        self.train_folds = set([f for f in range(K) if f not in self.tuning_folds.union(self.held_out_folds)])

        self.seed                      = seed
        self.tqdm                      = tqdm
        self.multiprocessing_pool_size = multiprocessing_pool_size
        self.chunksize                 = chunksize

        self.sequence_id_idxs = list(to_set(sequence_id_idxs))
        if len(self.sequence_id_idxs) > 1:
            self.should_add_id = True
            self.sequence_id_idx = self.SEQUENCE_ID
        else:
            self.should_add_id = False
            self.sequence_id_idx = self.sequence_id_idxs[0]


    def get_train_examples(self, df, save_path=None):
        return self.get_examples(df, self.train_folds, save_path)
    def get_dev_examples(self, df, save_path=None):
        return self.get_examples(df, self.tuning_folds, save_path)
    def get_test_examples(self, df, save_path=None):
        return self.get_examples(df, self.held_out_folds, save_path)

    def get_examples(self, df, folds=None, save_path=None):
        """Creates examples for the training and dev sets."""
        print(df.shape, folds)

        fold_idx = df.index.get_level_values(FOLD_IDX_LVL)
        if folds is not None: df = df[fold_idx.isin(to_set(folds))]

        if self.should_add_id: add_id_col(df, self.sequence_id_idxs, self.SEQUENCE_ID)

        random.seed(self.seed)
        np.random.seed(self.seed)

        sequence_idx = df.index.get_level_values(self.sequence_id_idx)
        sequences = sorted(list(set(sequence_idx))) # Make this determinisitic for crying out loud.

        if self.chunksize is None: return self.process_sequences(df, sequences, save_path=save_path)

        sequences_chunks = np.array_split(list(sequences), len(sequences)//self.chunksize)

        for chunk_num, sequences_chunk in enumerate(sequences_chunks):
            if save_path is not None:
                save_path_chunk = '%s.chunk_%d' % (save_path, chunk_num)
                if os.path.isfile(save_path_chunk):
                    print("Already finished chunk %d at %s" % (chunk_num, save_path_chunk))
                    continue

            print("Processing Chunk %d/%d" % (chunk_num + 1, len(sequences_chunks)))
            examples = self.process_sequences(
                df[sequence_idx.isin(sequences_chunk)], sequences_chunk, save_path=save_path_chunk
            )

            del examples
            gc.collect()

    def process_sequences(self, df, sequences, save_path=None):
        sequence_idx = df.index.get_level_values(self.sequence_id_idx)
        examples_gen = (ContinuousInputExample(
            guid=seq, seq_a=df[sequence_idx == seq].values, seq_b=None, labels={}
        ) for seq in sequences)

        N = len(sequences)
        assert N > 0, "Must process some sequencs!"
        print("Processing %d sequences" % N)

        tqdm = lambda i: (i if self.tqdm is None else self.tqdm(i, total=N))

        if self.multiprocessing_pool_size > 1:
            with closing(Pool(self.multiprocessing_pool_size)) as p:
                if self.tqdm is None: examples = p.map(I, examples_gen)
                else: examples = list(tqdm(p.imap(I, examples_gen)))
        else: examples = list(tqdm(examples_gen))

        if save_path is not None:
            with open(save_path, mode='wb') as f: pickle.dump(examples, f)

        assert len(examples) == N, "this really ought not be necessary..."

        return examples

class ContinuousBertPretrainingDataProcessor():
    SEQUENCE_ID = 'sequence_id'
    STREAM_ID   = 'sequence_stream_id'

    def __init__(
        self, sequence_id_idxs, sequence_stream_idxs, stream_order_idxs,
        tuning_folds=[], held_out_folds=[],
        guid_sep = '-', seed=1,
        tqdm = None, multiprocessing_pool_size=1, chunksize=None,
    ):
        self.tuning_folds, self.held_out_folds = to_set(tuning_folds), to_set(held_out_folds)
        self.train_folds = set([f for f in range(K) if f not in self.tuning_folds.union(self.held_out_folds)])

        self.sequence_id_idxs          = list(to_set(sequence_id_idxs))
        self.sequence_stream_idxs      = list(to_set(sequence_stream_idxs))
        self.stream_order_idxs         = stream_order_idxs
        self.guid_sep                  = guid_sep
        self.seed                      = seed
        self.whole_sequence_tasks      = {SEQUENCES_ORDERED: len(LABEL_ENUMS[SEQUENCES_ORDERED])}
        self.tqdm                      = tqdm
        self.multiprocessing_pool_size = multiprocessing_pool_size
        self.chunksize                 = chunksize

    def get_train_examples(self, df, save_path=None):
        return self.get_examples(df, self.train_folds, save_path)
    def get_dev_examples(self, df, save_path=None):
        return self.get_examples(df, self.tuning_folds, save_path)
    def get_test_examples(self, df, save_path=None):
        return self.get_examples(df, self.held_out_folds, save_path)

    def process_stream(self, args):
        sequence_id_idx = self.SEQUENCE_ID if len(self.sequence_id_idxs) > 1 else self.sequence_id_idxs[0]

        # TODO(mmd): Make this not need whole df.
        s_df, non_stream_df = args

        examples = []
        seq_idx = s_df.index.get_level_values(sequence_id_idx)
        seqs = list(set(seq_idx))
        if len(seqs) == 1:
            examples.append(ContinuousInputExample(
                guid=seqs[0], seq_a=s_df.values, seq_b=None, labels={},
            ))
            return examples

        n_examples = len(seqs) - 1
        for i in range(n_examples):
            seq_a_id, seq_b_id = seqs[i], seqs[i+1]
            seq_a = s_df[seq_idx == seq_a_id].values
            seq_b = s_df[seq_idx == seq_b_id].values
            guid  = '%s%s%s' % (seq_a_id, self.guid_sep, seq_b_id)

            examples.append(ContinuousInputExample(
                guid=guid, seq_a=seq_a, seq_b=seq_b,
                labels={SEQUENCES_ORDERED: SequenceOrderType.ORDERED_AND_FROM_SAME_STREAM},
            ))

        # TODO: Set thresholds
        mismatched_subseqs = np.random.permutation(list(range(n_examples+1)))
        while any_subsequent_indices_ordered(mismatched_subseqs):
            mismatched_subseqs = np.random.permutation(list(range(n_examples+1)))

        for i in range(n_examples):
            seq_a_id, seq_b_id = seqs[mismatched_subseqs[i]], seqs[mismatched_subseqs[i+1]]
            seq_a = s_df[seq_idx == seq_a_id].values
            seq_b = s_df[seq_idx == seq_b_id].values
            guid  = '%s%s%s' % (seq_a_id, self.guid_sep, seq_b_id)

            examples.append(ContinuousInputExample(
                guid=guid, seq_a=seq_a, seq_b=seq_b,
                labels={SEQUENCES_ORDERED: SequenceOrderType.NOT_ORDERED_BUT_FROM_SAME_STREAM},
            ))

        # Do non-same-stream.
        non_stream_seqs_idx = non_stream_df.index.get_level_values(sequence_id_idx)
        non_stream_seqs = np.random.permutation(list(set(non_stream_seqs_idx)))[:n_examples+2]
        shuffled_seqs_idx = np.random.permutation(list(range(n_examples+1)))

        for i in range(min(len(non_stream_seqs), n_examples+1)):
            seq_a_id, seq_b_id = seqs[shuffled_seqs_idx[i]], non_stream_seqs[i]
            seq_a = s_df[seq_idx == seq_a_id].values
            seq_b = non_stream_df[non_stream_seqs_idx == seq_b_id].values
            guid  = '%s%s%s' % (seq_a_id, self.guid_sep, seq_b_id)
            examples.append(ContinuousInputExample(
                guid=guid, seq_a=seq_a, seq_b=seq_b,
                labels={SEQUENCES_ORDERED: SequenceOrderType.NOT_FROM_SAME_STREAM},
            ))

        return examples

    def get_examples(self, df, folds=None, save_path=None):
        """Creates examples for the training and dev sets."""
        fold_idx = df.index.get_level_values(FOLD_IDX_LVL)
        if folds is not None: df = df[fold_idx.isin(to_set(folds))]

        if len(self.sequence_id_idxs) > 1:
            add_id_col(df, self.sequence_id_idxs, self.SEQUENCE_ID)
            sequence_id_idx = self.SEQUENCE_ID
        else: sequence_id_idx = self.sequence_id_idxs[0]
        if len(self.sequence_stream_idxs) > 1:
            add_id_col(df, self.sequence_stream_idxs, self.SEQUENCE_ID)
            sequence_stream_idx = self.STREAM_ID
        else: sequence_stream_idx = self.sequence_stream_idxs[0]

        df = df.sort_index(level=[sequence_stream_idx] + self.stream_order_idxs, axis=0)

        random.seed(self.seed)
        np.random.seed(self.seed)

        stream_idx = df.index.get_level_values(sequence_stream_idx)
        sequence_idx = df.index.get_level_values(sequence_id_idx)
        streams = np.random.permutation(list(set(stream_idx)))
        sequences = set(sequence_idx)

        # This processing step is _very_ expensive. Running it with no bells and whistles takes ~70 hours.
        # So, we have the option to parallelize it via a multiprocessing pool, which requires serializing and
        # sending input data across the cpu pool. We can't do this if we're passing the whole dataframe
        # around, so instead we build these generators below which do the slicing in the main thread before
        # sending to the workers. At the end, the system looks like it will take on the order of 2 - 6.5 hours
        # with a pool of size 256 over 80 cpu nodes.

        N = len(streams)
        if self.chunksize is None: streams_chunks = [streams]
        else: streams_chunks = np.array_split(streams, N//self.chunksize)

        # TODO(mmd): Make all work...
        all_examples = []
        for chunk_num, streams_chunk in enumerate(streams_chunks):
            if save_path is not None:
                save_path_chunk = '%s.chunk_%d' % (save_path, chunk_num)
                if os.path.isfile(save_path_chunk):
                    print("Already finished chunk %d at %s" % (chunk_num, save_path_chunk))
                    continue

            print("Processing Chunk %d/%d" % (chunk_num, len(streams_chunks)))

            print(chunk_num, 'streams_chunk', sys.getsizeof(streams_chunk))

            stream_dfs = (df[stream_idx == stream] for stream in streams_chunk)

            print(chunk_num, 'streams_df', sys.getsizeof(stream_dfs))

            stream_sequences = (set(s_df.index.get_level_values(sequence_id_idx)) for s_df in stream_dfs)

            print(chunk_num, 'stream_sequences', sys.getsizeof(stream_sequences))

            nonstream_dfs = (df[
                sequence_idx.isin(np.random.choice(list(sequences-stream_seqs), len(stream_seqs)+2))
            ] for stream_seqs in stream_sequences)

            print(chunk_num, 'nonstream_dfs', sys.getsizeof(nonstream_dfs))

            inputs = zip(stream_dfs, nonstream_dfs)
            if self.multiprocessing_pool_size > 1:
                with closing(Pool(self.multiprocessing_pool_size)) as p:
                    if self.tqdm is not None: 
                        examples = list(self.tqdm(
                            p.imap(self.process_stream, inputs), total=len(streams_chunk)
                        ))
                    else: examples = p.map(self.process_stream, inputs)
                examples = flatten(examples)
            else:
                examples = []
                for i in (inputs if self.tqdm is None else self.tqdm(inputs, total=len(streams_chunk))):
                    examples.extend(self.process_stream(i))

            if save_path is not None:
                save_path_chunk = '%s.chunk_%d' % (save_path, chunk_num)
                print("Saving partial processor output to %s" % save_path_chunk)
                with open(save_path_chunk, mode='wb') as f: pickle.dump(examples, f)
            else: all_examples.extend(examples)

            del stream_dfs
            del stream_sequences
            del nonstream_dfs
            del inputs
            del examples
            gc.collect()

        return all_examples
