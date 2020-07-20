import enum, os

from ..constants import *

#BERT_MODEL_LOCATION = os.path.join(
#    os.environ['ML4H_BASE'], 
#    'pretrained_models', 
#    'pretrained_bert_tf', 
#    'biobert_pretrain_output_all_notes_150000'
#)

BERT_MODEL_LOCATION_ALT = os.path.join(
    RUNS_DIR,
    'BERT',
    'biobert_pretrain_output_all_notes_150000'
)
BERT_MODEL_LOCATION = BERT_MODEL_LOCATION_ALT


PRETRAINED_BERT_HIDDEN_DIM = 768

NOTES_AS_SINGLE_SENTENCE_FILENAME = 'notes_single_sentence.hdf'
NOTES_AS_SENTENCE_SEQS_FILENAME = 'notes_split_sentences.hdf'
NOTES_AS_SENTENCE_VEC_SEQS_FILENAME = 'all_sentences_as_vecs.hdf'
BERT_RUNS_DIR = os.path.join(RUNS_DIR, 'BERT')

# TODO: why do we need PAD or UNK?
UNK, SEP, PAD, CLS, MASK = "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"
CONTROL_TOKENS = [UNK, SEP, PAD, CLS, MASK]

NUM_CONTROL_TOKENS = len(CONTROL_TOKENS)
def set_at(i, v=1):
    def f(arr):
        arr[0, i] = v # TODO: Maybe generalize?
        return arr
    return f
CONTROL_VECTOR_PREFIXES = {
    token: set_at(i, 1) for i, token in enumerate(CONTROL_TOKENS)
    #token: np.array(([0]*i) + [1] + ([0]*(len(CONTROL_TOKENS)-i))) for i, token in enumerate(CONTROL_TOKENS)
}

# For reading notes as sentence seqs:
ALL = 'all'

# Notes
NOTE_ORDER_COLS = ['chartdate', 'charttime']
NOTE_ID_COLS = [ICUSTAY_ID, HADM_ID, 'category'] + NOTE_ORDER_COLS
NOTE_ID = 'note_id'

# NoteBERT
STATIC_CLINICAL_BERT_RUNS_DIR = os.path.join(BERT_RUNS_DIR, 'static_clinical_BERT')
DATASET_FILENAME = 'train_dataset.torch'
PROCESSOR_FILENAME = 'processor.pkl'
EXAMPLES_FILENAME = 'examples.pkl'

SEQUENCES_ORDERED = 'Sequence Order'
class SequenceOrderType(enum.Enum):
    ORDERED_AND_FROM_SAME_STREAM = 0
    NOT_ORDERED_BUT_FROM_SAME_STREAM = 1
    NOT_FROM_SAME_STREAM = 2

LABEL_ENUMS = {
    SEQUENCES_ORDERED: SequenceOrderType,
}
