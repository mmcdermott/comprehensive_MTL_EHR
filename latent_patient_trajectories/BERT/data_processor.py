import sys
from datetime import datetime
from ..constants import *

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DfDataProcessor(object):
    def __init__(self, sentence_a_col, sentence_b_col = None):
        self.sentence_a_col = sentence_a_col
        self.sentence_b_col = sentence_b_col

    def get_examples(self, df, folds=None):
        fold_idx = df.index.get_level_values(FOLD_IDX_LVL)
        if folds is not None:
            df = df[fold_idx.isin(folds)]

        return [InputExample(
            guid = str(idx), text_a = r[self.sentence_a_col],
            text_b = None if self.sentence_b_col is None else r[self.sentence_b_col],
        ) for idx, r in df.iterrows()]

def convert_example_to_tokens(example, tokenizer, max_len, max_seq_length):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        seq_len = len(tokens_a) + len(tokens_b)

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        seq_len = len(tokens_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    if seq_len > max_len:
        max_len = seq_len
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
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
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

    return (tokens, tokens_b, max_len)

def convert_tokens_to_features(tokens, tokens_b, tokenizer, max_seq_length):
    assert not tokens_b, "Not supported."
    if isinstance(tokens, str):
        tokens = [tokens]

    if len(tokens) > max_seq_length:
        tokens = ["[CLS]"] + tokens[-max_seq_length + 1:]

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length, \
            "len(x) [%d] != max_seq_length [%d]" % (len(input_ids), max_seq_length)
    assert len(input_mask) == max_seq_length, \
            "len(x) [%d] != max_seq_length [%d]" % (len(input_mask), max_seq_length)
    assert len(segment_ids) == max_seq_length, \
            "len(x) [%d] != max_seq_length [%d]" % (len(segment_ids), max_seq_length)

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)


def convert_examples_to_features(examples, task_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    max_len = 0
    for (ex_index, example) in enumerate(examples):
        tokens_for_example, tokens_b, max_len = convert_example_to_tokens(
            example, tokenizer, max_len, max_seq_length
        )

        features_for_example = convert_tokens_to_features(
            tokens_for_example, tokens_b, tokenizer, max_seq_length
        )

        features.append(features_for_example)

    print('Max Sequence Length: %d' %max_len)

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
