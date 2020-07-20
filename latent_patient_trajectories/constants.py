import os

PROJECT_NAME                = 'latent_patient_trajectories'
PROJECT_DIR                 = os.path.join(os.environ['PROJECTS_BASE'], PROJECT_NAME)
RUNS_DIR                    = os.path.join(PROJECT_DIR, 'runs')
HYPERPARAMETER_SEARCH_DIR   = os.path.join(RUNS_DIR, 'hyperparameter_search')
TASK_GENERALIZABILITY_DIR   = os.path.join(RUNS_DIR, 'task_generalizability')
DATA_DIR                    = os.path.join(PROJECT_DIR, 'processed_data')
# DATA_DIR                    = '/scratch/gobi2/bnestor/mimic_extraction_results'
ROTATIONS_DIR               = os.path.join(PROJECT_DIR, 'dataset', 'rotations')
DATA_FILENAME               = 'all_hourly_data.h5'
FOLDS_FILENAME              = 'subject_ids_per_fold.pkl'
FTS_FILENAME                = 'treatment_sequence.h5'
NOTES_FILENAME              = 'notes.hdf'
ARGS_FILENAME               = 'args.json'
PARAMS_FILENAME             = 'raw_params.json'
FINE_TUNE_ARGS_FILENAME     = 'fine_tune_args.json'
EVAL_ARGS_FILENAME          = 'eval_args.json'
CLUSTERING_ARGS_FILENAME    = 'clustering_args.json'
CLUSTERING_CONFIG_FILENAME  = 'clustering_config.json'
CONFIG_FILENAME             = 'bert_config.json'
HYP_CONFIG_FILENAME         = 'hyperparameter_search_config.json'
TASK_GEN_CFG_FILENAME       = 'task_generalizability_config.json'
TASK_GEN_BASE_ARGS_FILENAME = 'task_generalizability_model_base_args.json'
TASK_GEN_EXP_ARGS_FILENAME  = 'task_generalizability_exp_args.json'
GET_ALL_FLAT_REPR_ARGS_FILENAME = 'get_all_flat_repr_args.json'
GET_PCA_ARGS_FILENAME           = 'get_pca_args.json'

FLAT_DATA_FILENAME_TEMPLATE = '{split}_{type}_flat_data.h5'

TEST_DATA_FILENAME  = 'all_hourly_data_test.h5'
TEST_NOTES_FILENAME = 'notes_test.hdf'
TEST_FTS_FILENAME   = 'treatment_sequence_test.h5'

STATICS    = 'patients'
NUMERICS   = 'vitals_labs'
CODES      = 'codes'
TREATMENTS = 'interventions'

ICUSTAY_ID = 'icustay_id'
SUBJECT_ID = 'subject_id'
HADM_ID    = 'hadm_id'

ID_COLS = [ICUSTAY_ID, HADM_ID, SUBJECT_ID]

FOLD_IDX_LVL = 'Fold'
K = 10 # K-fold CV

EXCLUSION_CRITERIA = {
    # TODO(ANYONE WHO CHANGES): FTS were generated with exclusion criteria 1.5, None for LOS. Must be updated
    # if changed.
    'los_icu': (1.5, None),
}

PATIENT_ID_COLS = [SUBJECT_ID, HADM_ID, ICUSTAY_ID] # TODO(mmd): We need a separate one for some joins. Why?

ALL_TASKS = [
    'rolling_fts',
    'disch_24h',
    'disch_48h',
    'Final Acuity Outcome',
    'tasks_binary_multilabel',
    'next_timepoint',
    'next_timepoint_was_measured',
]
ABLATION_GROUPS = {
    'icd10': [
        'icd_infection', 'icd_neoplasms', 'icd_endocrine', 'icd_blood', 'icd_mental', 'icd_nervous',
        'icd_circulatory', 'icd_respiratory', 'icd_digestive', 'icd_genitourinary', 'icd_pregnancy',
        'icd_skin', 'icd_musculoskeletal', 'icd_congenital', 'icd_perinatal', 'icd_ill_defined','icd_injury',
        'icd_unknown'
    ],
    'discharge': ['disch_24h', 'disch_48h'],
    'mortality': ['mort_24h', 'mort_48h'],
    'los': ['Long LOS'],
    'readmission': ['Readmission 30'],
    'future_treatment_sequence': ['rolling_fts'],
    'acuity': ['Final Acuity Outcome'],
    'next_timepoint_info': ['next_timepoint', 'next_timepoint_was_measured'],
    'dnr': ['dnr_24h', 'dnr_48h'],
    'cmo': ['cmo_24h', 'cmo_48h'],
}

ALL_SPECIFIC_TREATMENTS = [
    'vent',
    'nivdurations',
    'adenosine',
    'dobutamine',
    'dopamine',
    'epinephrine',
    'isuprel',
    'milrinone',
    'norepinephrine',
    'phenylephrine',
    'vasopressin',
    'colloid_bolus',
    'crystalloid_bolus',
]

GENERALIZED_TREATMENTS = {
    'vent': ('vent', 'nivdurations'), 'vaso': ('vaso',), 'bolus': ('colloid_bolus', 'crystalloid_bolus'),
}

DURATIONS_COL = 'hours_in'

TRAIN, TUNING, HELD_OUT = 'train', 'tuning', 'held out'

UNK = 'Unknown'

ABBREVIATIONS = {
    'Imminent Mortality':              'MOR',
    'Comfort Measures':                'CMO',
    'DNR Ordered':                     'DNR',
    'ICD Code Prediction':             'ICD',
    'Long LOS':                        'LOS',
    '30-day Readmission':              'REA',
    'Imminent Discharge':              'DIS',
    'Final Acuity Outcome':            'ACU',
    'Next Hour Will-be-measured':      'WBM',
    'Future Treatment Sequence (FTS)': 'FTS',
}

MANUSCRIPT_BREAKDOWN = {
    ABBREVIATIONS['Imminent Mortality']:              ('tasks_binary_multilabel','all_time',lambda s: s.startswith('mort')),
    ABBREVIATIONS['Comfort Measures']:                ('tasks_binary_multilabel','all_time',lambda s: s.startswith('cmo')),
    ABBREVIATIONS['DNR Ordered']:                     ('tasks_binary_multilabel','all_time',lambda s: s.startswith('dnr')),
    ABBREVIATIONS['ICD Code Prediction']:             ('tasks_binary_multilabel','first_24',lambda s: s.startswith('icd')),
    ABBREVIATIONS['Long LOS']:                        ('tasks_binary_multilabel','first_24','Long LOS'),
    ABBREVIATIONS['30-day Readmission']:              ('tasks_binary_multilabel','extend_till_discharge','Readmission 30'),
    ABBREVIATIONS['Imminent Discharge']:              ['disch_24h', 'disch_48h'],
    ABBREVIATIONS['Final Acuity Outcome']:            'Final Acuity Outcome',
    ABBREVIATIONS['Next Hour Will-be-measured']:      'next_timepoint_was_measured',
    ABBREVIATIONS['Future Treatment Sequence (FTS)']: 'rolling_fts',
}
EVAL_MODES = ('all_time', 'first_24', 'extend_till_discharge')
EVAL_MODES_BY_ABLATION_GROUPS = {
    'icd10':                     'first_24',
    'discharge':                 'all_time',
    'mortality':                 'all_time',
    'los':                       'first_24',
    'readmission':               'extend_till_discharge',
    'future_treatment_sequence': 'all_time',
    'acuity':                    'first_24',
    'next_timepoint_info':       'all_time',
    'dnr':                       'all_time',
    'cmo':                       'all_time',
}

TASK_BINARY_MULTILABEL_ORDER = ['mort_24h', 'mort_48h', 'dnr_24h', 'dnr_48h', 'cmo_24h', 
                                'cmo_48h', 'Long LOS', 'icd_infection', 'icd_neoplasms', 
                                'icd_endocrine', 'icd_blood', 'icd_mental',  'icd_nervous', 
                                'icd_circulatory', 'icd_respiratory', 'icd_digestive', 
                                'icd_genitourinary', 'icd_pregnancy', 'icd_skin', 'icd_musculoskeletal', 
                                'icd_congenital', 'icd_perinatal', 'icd_ill_defined', 
                                'icd_injury', 'icd_unknown', 'Readmission 30']

TASK_HEAD_MAPPING = {
    'acuity': set(['task_heads.Final Acuity Outcome.weight',
                   'task_heads.Final Acuity Outcome.bias']),
    'discharge': set(['task_heads.disch_24h.weight',
                      'task_heads.disch_24h.bias',
                      'task_heads.disch_48h.weight',
                      'task_heads.disch_48h.bias']),
    'next_timepoint_info': set(['task_heads.next_timepoint.weight',
                                'task_heads.next_timepoint.bias',
                                'task_heads.next_timepoint_was_measured.weight',
                                'task_heads.next_timepoint_was_measured.bias']),
    'future_treatment_sequence': set(['treatment_embeddings.weight',
                                      'FTS_decoder.decoder.treatment_embeddings.weight',
                                      'FTS_decoder.decoder.C_proj.weight',
                                      'FTS_decoder.decoder.C_proj.bias',
                                      'FTS_decoder.decoder.H_proj.weight',
                                      'FTS_decoder.decoder.H_proj.bias',
                                      'FTS_decoder.decoder.X_proj.weight',
                                      'FTS_decoder.decoder.X_proj.bias',
                                      'FTS_decoder.decoder.LSTM.weight_ih_l0',
                                      'FTS_decoder.decoder.LSTM.weight_hh_l0',
                                      'FTS_decoder.decoder.LSTM.bias_ih_l0',
                                      'FTS_decoder.decoder.LSTM.bias_hh_l0',
                                      'FTS_decoder.predictor.classifier.weight',
                                      'FTS_decoder.predictor.classifier.bias']),
    'dnr': set(),
    'los': set(),
    'mortality': set(),
    'icd10': set(),
    'readmission': set(),
    'cmo': set(),
}

# TODO(mmd): The for tasks_binary_multilabel are augmented
TASK_SPECIFIC_WEIGHT_PREFIXES = {
    'acuity':                    set(['task_heads.Final Acuity Outcome']),
    'discharge':                 set(['task_heads.disch_24h', 'task_heads.disch_48h']),
    'next_timepoint_info':       set(['task_heads.next_timepoint', 'task_heads.next_timepoint_was_measured']),
    'future_treatment_sequence': set(['treatment_embeddings', 'FTS_decoder']),
    'dnr':                       set(['task_heads.tasks_binary_multilabel.dnr']),
    'los':                       set(['task_heads.tasks_binary_multilabel.Long LOS']),
    'mortality':                 set(['task_heads.tasks_binary_multilabel.mort']),
    'icd10':                     set(['task_heads.tasks_binary_multilabel.icd']),
    'readmission':               set(['task_heads.tasks_binary_multilabel.Readmission 30']),
    'cmo':                       set(['task_heads.tasks_binary_multilabel.cmo']),
}

ALWAYS_EQ_KEYS = [
    'task_losses.tasks_binary_multilabel.BCE_LL.pos_weight',
    'bert.encoder.layer.0.attention.self.key.bias',
]
