"""
SelfAttentionEncoder.py
"""

import torch, torch.optim, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from torch.autograd import Variable, set_detect_anomaly
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler

from math import floor

from ..utils import *
from ..constants import *
from ..data_utils import *
from ..representation_learner.fts_decoder import *
from ..BERT.model import *
from ..BERT.constants import *

from copy import deepcopy

class TaskBinaryMultilabelLoss(nn.Module):
    def __init__(self, binary_multilabel_loss_weight=None):
        super().__init__()
        self.weights = binary_multilabel_loss_weight
        self.weights.requires_grad_(False)
        params = {'pos_weight': binary_multilabel_loss_weight, 'reduction': 'none'}
        self.BCE_LL = nn.BCEWithLogitsLoss(**params)

    def forward(self, logits, labels):
        new_weights = self.weights.unsqueeze(0).expand_as(logits)
        out = self.BCE_LL(logits, labels)
        out = out * new_weights
        return out

def get_task_losses(task_class_weights):
    task_losses = {}
    for t in ('disch_24h', 'disch_48h'):
        # May have missingness.
        params = {'ignore_index': -1, 'reduction': 'none'}
        if t in task_class_weights: params['weight'] = task_class_weights[t]
        task_losses[t] = nn.CrossEntropyLoss(**params)
    for t in ('Final Acuity Outcome',):
        params = {'ignore_index': -1}
        if t in task_class_weights: params['weight'] = task_class_weights[t]
        task_losses[t] = nn.CrossEntropyLoss(**params)
    for t in ('tasks_binary_multilabel',):
        # May have missingness.
        # See:
        # https://discuss.pytorch.org/t/what-is-the-difference-between-bcewithlogitsloss-and-multilabelsoftmarginloss/14944/13
        # params = {'reduction': 'none'}
        # if t in task_class_weights: params['pos_weight'] = task_class_weights[t]
        # task_losses[t] = nn.BCEWithLogitsLoss(**params)
        if t in task_class_weights:
            task_losses[t] = TaskBinaryMultilabelLoss(task_class_weights[t])
        else:
            task_losses[t] = TaskBinaryMultilabelLoss()
    for t in ('next_timepoint_was_measured',):
        params = {}
        if t in task_class_weights: params['weight'] = task_class_weights[t]
        task_losses[t] = nn.MultiLabelSoftMarginLoss(**params)

    return nn.ModuleDict(task_losses)

# TODO(mmd): Update comment
# TODO(mmd): Strip unnecessary fields from config.
class SelfAttentionTimeseries(BertPreTrainedModel):
    """BERT model with continuous reconstruction pre-training heads.
    Also supports additional, user-specified auxiliary losses.

    This module comprises the BERT model followed by the two pre-training heads:
        - the masked continuous reconstruction modeling head, and
        - the next sequence classification head.
    as well as any user specified auxiliary loss prediction heads.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `sequence`: a torch.LongTensor of shape [batch_size, sequence_length, hidden_size]
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sequence A` and type 1 corresponds to
            a `sequence B` token (see BERT paper for more details).
        `ts_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sequences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sequence_label`: optional next sequence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sequence is the continuation, 1 => next sequence is a random sequence.

    Outputs:
        if `masked_lm_labels` and `next_sequence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sequence classification loss.
        if `masked_lm_labels` or `next_sequence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sequence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    sequence = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(sequence, token_type_ids, input_mask)
    ```
    """
    def __init__(
        self,
        config,
        n_gpu = 0,
        use_cuda=True,
        task_class_weights = None,
        task_weights = None,
    ):
        super().__init__(config)
        if task_class_weights is None: task_class_weights = {}

        # TODO(mmd): This should probably be a modification of the superclass.
        self.bert = ContinuousBertModel(config)
#         self.cls = ContinuousBertPreTrainingHeads(config) # modify this to get all of the necessary tasks
        self.apply(self.init_bert_weights)
        self.n_gpu = n_gpu

        # TODO: API!
        # TODO: disch dims are wrong (18)
        self.task_class_weights = task_class_weights
        self.task_dims = {
            'disch_24h': 20,
            'disch_48h': 20,
            'Final Acuity Outcome': 20,
            'tasks_binary_multilabel': 26, #7, # ICD10, los, readmission
            'next_timepoint': 56, # TODO put in config
            'next_timepoint_was_measured': 56,
        }

        # TODO(mmd): API?
        self.task_heads = nn.ModuleDict(
            {t: nn.Linear(config.hidden_size, d) for t, d in self.task_dims.items()}
        )
        if task_weights is None:
            self.task_weights = {t: 1 for t in self.task_heads.keys()}
            self.task_weights['rolling_fts'] = 1
        else: self.task_weights = task_weights

        # Setting these here enables to(device) / .cuda() to naturally affect them.

        self.task_permits_missingness = {'disch_24h', 'disch_48h', 'tasks_binary_multilabel'}
        self.task_losses = get_task_losses(self.task_class_weights)
        self.next_timepoint_reconstruction_loss = nn.MSELoss(reduction='none')

        # We do Rolling FTS separately
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )
        self.FTS_decoder = FutureTreatmentSequenceDecoder(
            decoder_module = LSTMDecoder(
                in_dim               = config.hidden_size,
                treatment_embeddings = self.treatment_embeddings,
            ),
            predictor_module = SingleTaskPredictor(
                in_dim      = 25,
                num_classes = 9,
            ),
        )

    def freeze_representation(self):
        for p in self.bert.parameters(): p.requires_grad = False
            
    def unfreeze_representation(self):
        for p in self.bert.parameters(): p.requires_grad = True

    def forward(
        self,
        dfs, # Should be a dict...
        #ts,
        #statics,
        #ts_mask=None,
        #next_timepoint=None,
        #next_timepoint_was_measured=None,
        #disch_24=None,
        #disch_48=None,
        #rolling_fts = None,
        #tasks_binary_multilabel = None,
    ):
        # TODO(mmd): Put type conversions in dataset.
        for k in (
            'ts', 'statics', 'ts_mask', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
        ):
            if k in dfs: dfs[k] = dfs[k].float()
        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_fts'):
            if k in dfs: dfs[k] = dfs[k].squeeze().long()

        # TODO(mmd): Moving to meta-model.
        #statics, ts = dfs['statics'], dfs['ts']

        #batch_size, seq_len, ts_feat_dim = list(ts.shape)
        #batch_size, statics_feat_dim = list(statics.shape)

        #statics = statics.unsqueeze(1).expand([batch_size, seq_len, statics_feat_dim])

        #input_sequence = torch.cat((ts, statics), dim=2)
        input_sequence = dfs['input_sequence']
        batch_size, seq_len, feat_dim = list(input_sequence.shape)
        sequence_output, pooled_output = self.bert(
            input_sequence, None, dfs['ts_mask'].squeeze(), output_all_encoded_layers=False)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output is batch_size, hidden_dim

        # insert all the prediction tasks here
        task_labels = {
            k: df for k, df in dfs.items() if \
                    k not in ('statics', 'ts', 'rolling_fts', 'ts_mask') and df is not None
        }
        tasks = list(set(task_labels.keys()).intersection(self.task_heads.keys()))
        assert 'rolling_fts' in dfs or tasks, "Must have some tasks!"

        task_logits = {t: self.task_heads[t](pooled_output) for t in tasks}

        #for t in tasks:
        #    if t not in self.task_losses: continue
        #    print(t, 'labels', dfs[t].dtype, dfs[t].shape, 'logits', task_logits[t].dtype, task_logits[t].shape)

        task_losses = {}
        for task, loss_fn, logits, labels, weight in zip_dicts(
            self.task_losses, task_logits, dfs, self.task_weights
        ):
            if task in self.task_permits_missingness:
                isnan = torch.isnan(labels)
                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels)
                
                try:
                    if task in ('disch_24h', 'disch_48h'):
                        loss = torch.where(
                            isnan, torch.zeros_like(labels, dtype=torch.float32),
                            loss_fn(logits, labels_smoothed)
                        )
                    elif task in ('tasks_binary_multilabel',):
                        loss = (self.task_class_weights[task]!=0).float() * torch.where(isnan, torch.zeros_like(logits), loss_fn(logits, labels_smoothed))
                    else:
                        raise
                except:
                    print(task)
                    print(self.task_class_weights.keys())
                    print(self.task_class_weights[task])
                    print(self.task_class_weights[task]!=0)
                    print(task, self.task_class_weights[task].shape, isnan.shape, labels.shape, logits.shape, labels_smoothed.shape)
                    raise
                loss = loss.mean()
                loss = weight * loss
            else:
                loss = weight * loss_fn(logits, labels)

            task_losses[task] = loss

        if 'rolling_fts' in dfs:
            fts_labels = dfs['rolling_fts']
            fts_logits, fts_loss = self.FTS_decoder(pooled_output, labels = fts_labels)
            task_logits['rolling_fts'] = fts_logits
            task_losses['rolling_fts'] = self.task_weights['rolling_fts'] * fts_loss
            tasks.append('rolling_fts')

        # We need to handle next timepoint separately to deal with masking.
        if 'next_timepoint' and 'next_timepoint_was_measured' in dfs:
            recst_loss = self.next_timepoint_reconstruction_loss(
                dfs['next_timepoint'], task_logits['next_timepoint']
            )
            recst_loss *= dfs['next_timepoint_was_measured'] # Mask out those not obs.

            # Accounting for the completely unmeasured rows.
            # TODO(mmd): we'll need to do the same for some of the classification tasks...
            num_measured_per_patient = dfs['next_timepoint_was_measured'].sum(dim=1)
            num_measured_per_patient = torch.where(
                num_measured_per_patient == 0, torch.ones_like(num_measured_per_patient),
                num_measured_per_patient
            )
            recst_loss = recst_loss.sum(dim=1) / num_measured_per_patient

            recst_loss = recst_loss.sum(dim=0)
            task_losses['next_timepoint'] = self.task_weights['next_timepoint'] * recst_loss

        total_loss = task_losses[tasks[0]]
        for t in tasks[1:]: total_loss += task_losses[t]

        return (
            sequence_output,
            pooled_output,
            {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks},
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )

class CNN(nn.Module):
    """ TODO(this)
    """
    def __init__(
        self, config, data_shape=[48, 128], use_cuda=False, n_gpu = 0, task_class_weights = None,
        conv_layers = [10, 100, 100, 10], kernel_sizes= [7, 5, 5, 3], fc_layer_sizes = [1024],
        conv_layers_per_pool = 1, pooling_method='max', pooling_kernel_size=None, pooling_stride=1,
        task_weights=None,
    ):
        assert pooling_method in ('max', 'avg')

        super(CNN, self).__init__()
        if task_class_weights is None: task_class_weights = {}

        # conv 2d appraoch
        # TODO(mmd): why conv2d? Don't we want conv1d?
        seq_len, num_features = data_shape
        N_layers = len(conv_layers)
        pooler = nn.MaxPool1d if pooling_method == 'max' else nn.AvgPool1d
        pooling_resizer = lambda L_in: floor(
            ((L_in - pooling_kernel_size)/(pooling_kernel_size if pooling_stride is None else pooling_stride))
            + 1
        )

        conv_stack = []
        for layer, (num_filters, kernel_size) in enumerate(zip(conv_layers, kernel_sizes)):
            conv_layer = nn.Conv1d(num_features, num_filters, kernel_size)
            num_features = num_filters
            seq_len = floor(seq_len - kernel_size + 1)

            conv_stack.append(conv_layer)

            conv_stack.append(nn.ReLU())

            if (layer+1) % conv_layers_per_pool == 0:
                pooling = pooler(pooling_kernel_size, pooling_stride)
                seq_len = pooling_resizer(seq_len)
                conv_stack.append(pooling)

            dropout_layer = nn.Dropout(p=config.hidden_dropout_prob)
            conv_stack.append(dropout_layer)

            assert seq_len > 2, "Too small!"

        self.conv_encoder = nn.Sequential(*conv_stack)

        fc_in_dim = seq_len * num_features
        fc_stack = []
        for fc_layer_size in fc_layer_sizes:
            fc_stack.append(nn.Linear(fc_in_dim, fc_layer_size))
            fc_in_dim = fc_layer_size
            fc_stack.append(nn.ReLU())

        fc_stack.append(nn.Linear(fc_in_dim, config.hidden_size))
        self.fc_stack = nn.Sequential(*fc_stack)

        #TODO: (BN) weight init with xavier
        self.use_cuda = use_cuda
        self.n_gpu = n_gpu

        # TODO: API!
        self.task_class_weights = task_class_weights
        self.task_dims = {
            'disch_24h': 20,
            'disch_48h': 20,
            'Final Acuity Outcome': 20,
            'tasks_binary_multilabel': 26, #7,
            'next_timepoint': 56, # TODO put in config
            'next_timepoint_was_measured': 56,
        }

        # TODO(mmd): API?
        self.task_heads = nn.ModuleDict(
            {t: nn.Linear(config.hidden_size, d) for t, d in self.task_dims.items()}
        )
        if task_weights is None:
            self.task_weights = {t: 1 for t in self.task_heads.keys()}
            self.task_weights['rolling_fts'] = 1
        else: self.task_weights = task_weights
        # Setting these here enables to(device) / .cuda() to naturally affect them.

        self.task_permits_missingness = {'disch_24h', 'disch_48h', 'tasks_binary_multilabel'}
        self.task_losses = get_task_losses(self.task_class_weights)
        self.next_timepoint_reconstruction_loss = nn.MSELoss(reduction='none')

        # We do Rolling FTS separately
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )
        self.FTS_decoder = FutureTreatmentSequenceDecoder(
            decoder_module = LSTMDecoder(
                in_dim               = config.hidden_size,
                treatment_embeddings = self.treatment_embeddings,
            ),
            predictor_module = SingleTaskPredictor(
                in_dim      = 25,
                num_classes = 9,
            ),
        )

    def freeze_representation(self):
        for p in self.conv_encoder.parameters(): p.requires_grad = False
        for p in self.fc_stack.parameters(): p.requires_grad = False
            
    def unfreeze_representation(self):
        for p in self.conv_encoder.parameters(): p.requires_grad = True
        for p in self.fc_stack.parameters(): p.requires_grad = True

    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,
        dfs, # Should be a dict...
    ):
        # # TODO(mmd): Embedding Features...
        # input_sequence     = self.ts_continuous_projector(ts_continuous)
        # statics_continuous = self.statics_projector(statics)
        # statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        # input_sequence += statics_continuous

        # ts_mask = ts_mask.expand_as(input_sequence)

        # TODO(mmd): Put type conversions in dataset.
        for k in (
            'ts', 'statics', 'ts_mask', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
        ):
            if k in dfs: dfs[k] = dfs[k].float()
        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_fts'):
            if k in dfs: dfs[k] = dfs[k].squeeze().long()

        #input_sequence = torch.cat((ts, statics), dim=2)
        input_sequence = dfs['input_sequence']
        batch_size, seq_len, feat_dim = list(input_sequence.shape)

        # print(input_sequence.shape)
        x = self.conv_encoder(torch.transpose(input_sequence, 1, 2))
        x = x.view(batch_size, -1)
        pooled_output = self.fc_stack(x)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output.shape is batch_size, hidden_dim

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output is batch_size, hidden_dim

        # insert all the prediction tasks here
        task_labels = {
            k: df for k, df in dfs.items() if \
                    k not in ('statics', 'ts', 'rolling_fts', 'ts_mask') and df is not None
        }
        tasks = list(set(task_labels.keys()).intersection(self.task_heads.keys()))
        assert 'rolling_fts' in dfs or tasks, "Must have some tasks!"

        task_logits = {t: self.task_heads[t](pooled_output) for t in tasks}

        #for t in tasks:
        #    if t not in self.task_losses: continue
        #    print(t, 'labels', dfs[t].dtype, dfs[t].shape, 'logits', task_logits[t].dtype, task_logits[t].shape)

        task_losses = {}
        for task, loss_fn, logits, labels, weight in zip_dicts(
            self.task_losses, task_logits, dfs, self.task_weights
        ):
            if task in self.task_permits_missingness:
                isnan = torch.isnan(labels)
                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels)
                try:
                    if task in ('disch_24h', 'disch_48h'):
                        loss = torch.where(
                            isnan, torch.zeros_like(labels, dtype=torch.float32),
                            loss_fn(logits, labels_smoothed)
                        )
                    elif task in ('tasks_binary_multilabel',):
                        loss = (self.task_class_weights[task]!=0).float() * torch.where(isnan, torch.zeros_like(logits), loss_fn(logits, labels_smoothed))
                    else:
                        raise
                except:
                    print(task, isnan.shape, labels.shape, logits.shape, labels_smoothed.shape)
                    raise
                loss = loss.mean()
                loss = weight * loss
            else: loss = weight * loss_fn(logits, labels)

            task_losses[task] = loss

        if 'rolling_fts' in dfs:
            fts_labels = dfs['rolling_fts']
            fts_logits, fts_loss = self.FTS_decoder(pooled_output, labels = fts_labels)
            task_logits['rolling_fts'] = fts_logits
            task_losses['rolling_fts'] = self.task_weights['rolling_fts'] * fts_loss
            tasks.append('rolling_fts')

        # We need to handle next timepoint separately to deal with masking.
        if 'next_timepoint' and 'next_timepoint_was_measured' in dfs:
            recst_loss = self.next_timepoint_reconstruction_loss(
                dfs['next_timepoint'], task_logits['next_timepoint']
            )
            recst_loss *= dfs['next_timepoint_was_measured'] # Mask out those not obs.

            # Accounting for the completely unmeasured rows.
            # TODO(mmd): we'll need to do the same for some of the classification tasks...
            num_measured_per_patient = dfs['next_timepoint_was_measured'].sum(dim=1)
            num_measured_per_patient = torch.where(
                num_measured_per_patient == 0, torch.ones_like(num_measured_per_patient),
                num_measured_per_patient
            )
            recst_loss = recst_loss.sum(dim=1) / num_measured_per_patient

            recst_loss = recst_loss.sum(dim=0)
            task_losses['next_timepoint'] = self.task_weights['next_timepoint'] * recst_loss

        total_loss = task_losses[tasks[0]]
        for t in tasks[1:]: total_loss += task_losses[t]

        return (
            None,
            pooled_output,
            {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks},
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )

POOLING_METHODS = ('max', 'avg', 'last')#, 'attention')
class GRUModel(nn.Module):
    """ TODO(this)
    """
    def __init__(
        self, config, data_shape=[48, 128], use_cuda=False, n_gpu = 0, task_class_weights=None,
        task_weights = None, hidden_dim=512, num_layers=2, bidirectional=False,
        pooling_method = 'last', fc_layer_sizes = [],
    ):
        assert pooling_method in POOLING_METHODS, "Don't know how to do %s pooling" % pooling_method
        super(GRUModel, self).__init__()

        if task_class_weights is None: task_class_weights = {}

        # initialise the model and the weights
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=data_shape[-1], hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=config.hidden_dropout_prob, bidirectional=bidirectional,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        fc_stack = []
        for fc_layer_size in fc_layer_sizes:
            fc_stack.append(nn.Linear(out_dim, fc_layer_size))
            fc_stack.append(nn.ReLU())
            out_dim = fc_layer_size

        fc_stack.append(nn.Linear(out_dim, config.hidden_size))

        self.fc_stack = nn.Sequential(*fc_stack)

        if self.bidirectional:
            self.h_0 = torch.zeros(2*num_layers,  1, hidden_dim).float().to('cuda' if use_cuda else 'cpu')
        else:
            self.h_0 = torch.zeros(num_layers,  1, hidden_dim).float().to('cuda' if use_cuda else 'cpu')

        self.hidden_dim=hidden_dim

        self.pooling_method = pooling_method
        #elif pooling_method == 'attention':
        #    self.attention

#         self.cls = ContinuousBertPreTrainingHeads(config) # modify this to get all of the necessary tasks
        self.use_cuda = use_cuda
        self.n_gpu = n_gpu

        # TODO: API!
        self.task_class_weights = task_class_weights
        self.task_dims = {
            'disch_24h': 20,
            'disch_48h': 20,
            'Final Acuity Outcome': 20,
            'tasks_binary_multilabel': 26, #7,
            'next_timepoint': 56, # TODO put in config
            'next_timepoint_was_measured': 56,
        }

        # TODO(mmd): API?
        self.task_heads = nn.ModuleDict(
            {t: nn.Linear(config.hidden_size, d) for t, d in self.task_dims.items()}
        )
        if task_weights is None:
            self.task_weights = {t: 1 for t in self.task_heads.keys()}
            self.task_weights['rolling_fts'] = 1
        else: self.task_weights = task_weights
        # Setting these here enables to(device) / .cuda() to naturally affect them.

        self.task_permits_missingness = {'disch_24h', 'disch_48h', 'tasks_binary_multilabel'}
        self.task_losses = get_task_losses(self.task_class_weights)
        self.next_timepoint_reconstruction_loss = nn.MSELoss(reduction='none')

        # We do Rolling FTS separately
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )
        self.FTS_decoder = FutureTreatmentSequenceDecoder(
            decoder_module = LSTMDecoder(
                in_dim               = config.hidden_size,
                treatment_embeddings = self.treatment_embeddings,
            ),
            predictor_module = SingleTaskPredictor(
                in_dim      = 25,
                num_classes = 9,
            ),
        )
    def freeze_representation(self):
        for p in self.gru.parameters(): p.requires_grad = False
        for p in self.fc_stack.parameters(): p.requires_grad = False
            
    def unfreeze_representation(self):
        for p in self.gru.parameters(): p.requires_grad = True
        for p in self.fc_stack.parameters(): p.requires_grad = True

    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,
        dfs, # Should be a dict...
        h_0=None
    ):
        # # TODO(mmd): Embedding Features...
        # input_sequence     = self.ts_continuous_projector(ts_continuous)
        # statics_continuous = self.statics_projector(statics)
        # statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        # input_sequence += statics_continuous

        # ts_mask = ts_mask.expand_as(input_sequence)

        # TODO(mmd): Put type conversions in dataset.
        for k in (
            'ts', 'statics', 'ts_mask', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
        ):
            if k in dfs: dfs[k] = dfs[k].float()
        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_fts'):
            if k in dfs: dfs[k] = dfs[k].squeeze().long()

        #input_sequence = torch.cat((ts, statics), dim=2)
        input_sequence = dfs['input_sequence']
        batch_size, seq_len, feat_dim = list(input_sequence.shape)

        # ts_mask = ts_mask.expand_as(input_sequence)
        if h_0 is None:
            h_0 = self.h_0

        if batch_size != 1:
            h_0 = h_0.expand(-1, batch_size, -1).contiguous()

        out, h = self.gru(input_sequence, h_0) # for gru

        # print(out.shape)
        # print(batch_size, seq_len, feat_dim, self.hidden_dim)
        if self.pooling_method == 'last': out = out[:, -1, :]
        elif self.pooling_method == 'max': out = out.max(dim=1)[0]
        elif self.pooling_method == 'avg': out = out.mean(dim=1)

        out = out.contiguous().view(batch_size, -1) # num directions is 1 for forward-only rnn

        pooled_output = self.fc_stack(out)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output is batch_size, hidden_dim

        # insert all the prediction tasks here
        task_labels = {
            k: df for k, df in dfs.items() if \
                    k not in ('statics', 'ts', 'rolling_fts', 'ts_mask') and df is not None
        }
        tasks = list(set(task_labels.keys()).intersection(self.task_heads.keys()))
        assert 'rolling_fts' in dfs or tasks, "Must have some tasks!"

        task_logits = {t: self.task_heads[t](pooled_output) for t in tasks}

        #for t in tasks:
        #    if t not in self.task_losses: continue
        #    print(t, 'labels', dfs[t].dtype, dfs[t].shape, 'logits', task_logits[t].dtype, task_logits[t].shape)

        task_losses = {}
        for task, loss_fn, logits, labels, weight in zip_dicts(
            self.task_losses, task_logits, dfs, self.task_weights
        ):
            if task in self.task_permits_missingness:
                isnan = torch.isnan(labels)
                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels)
                try:
                    if task in ('disch_24h', 'disch_48h'):
                        loss = torch.where(
                            isnan, torch.zeros_like(labels, dtype=torch.float32),
                            loss_fn(logits, labels_smoothed)
                        )
                    elif task in ('tasks_binary_multilabel',):
                        loss = (self.task_class_weights[task]!=0).float() * torch.where(isnan, torch.zeros_like(logits), loss_fn(logits, labels_smoothed))
                    else:
                        raise
                except:
                    print(task, isnan.shape, labels.shape, logits.shape, labels_smoothed.shape)
                    raise
                loss = loss.mean()
                loss = weight * loss
            else: loss = weight * loss_fn(logits, labels)

            task_losses[task] = loss

        if 'rolling_fts' in dfs:
            fts_labels = dfs['rolling_fts']
            fts_logits, fts_loss = self.FTS_decoder(pooled_output, labels = fts_labels)
            task_logits['rolling_fts'] = fts_logits
            task_losses['rolling_fts'] = self.task_weights['rolling_fts'] * fts_loss
            tasks.append('rolling_fts')

        # We need to handle next timepoint separately to deal with masking.
        if 'next_timepoint' and 'next_timepoint_was_measured' in dfs:
            recst_loss = self.next_timepoint_reconstruction_loss(
                dfs['next_timepoint'], task_logits['next_timepoint']
            )
            recst_loss *= dfs['next_timepoint_was_measured'] # Mask out those not obs.

            # Accounting for the completely unmeasured rows.
            # TODO(mmd): we'll need to do the same for some of the classification tasks...
            num_measured_per_patient = dfs['next_timepoint_was_measured'].sum(dim=1)
            num_measured_per_patient = torch.where(
                num_measured_per_patient == 0, torch.ones_like(num_measured_per_patient),
                num_measured_per_patient
            )
            recst_loss = recst_loss.sum(dim=1) / num_measured_per_patient

            recst_loss = recst_loss.sum(dim=0)
            task_losses['next_timepoint'] = self.task_weights['next_timepoint'] * recst_loss

        total_loss = task_losses[tasks[0]]
        for t in tasks[1:]: total_loss += task_losses[t]

        # formerly returned: {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks}, total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss

        return (
            None,
            pooled_output,
            {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks},
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )

class LinearModel(nn.Module):
    """ TODO(this)
    """
    def __init__(
        self, config, data_shape=[48, 128], use_cuda=False, n_gpu = 0,
        task_class_weights = None, task_weights = None,
    ):
        super(LinearModel, self).__init__()
        if task_class_weights is None: task_class_weights = {}

        #TODO: (BN) weight init with xavier
        self.use_cuda = use_cuda
        self.n_gpu = n_gpu

        # TODO: API!
        self.task_class_weights = task_class_weights
        self.task_dims = {
            'disch_24h': 20,
            'disch_48h': 20,
            'Final Acuity Outcome': 20,
            'tasks_binary_multilabel': 26, # 7 without icd, 25 with ['Long LOS', 'icd_infection', 'icd_neoplasms', 'icd_endocrine', 'icd_blood', 'icd_mental', 'icd_nervous', 'icd_circulatory', 'icd_respiratory', 'icd_digestive', 'icd_genitourinary', 'icd_pregnancy', 'icd_skin', 'icd_musculoskeletal', 'icd_congenital', 'icd_perinatal', 'icd_ill_defined', 'icd_injury', 'icd_unknown', 'mort_24h', 'mort_48h', 'dnr_24h', 'dnr_48h', 'cmo_24h', 'cmo_48h']
            'next_timepoint': 56, # TODO put in config
            'next_timepoint_was_measured': 56,
        }

        # TODO(mmd): API?
        #notice the change is in self.task_heads
        self.task_heads = nn.ModuleDict(
            {t: nn.Linear(data_shape[0]*data_shape[1], d) for t, d in self.task_dims.items()}
        )
        if task_weights is None:
            self.task_weights = {t: 1 for t in self.task_heads.keys()}
            self.task_weights['rolling_fts'] = 1
        else: self.task_weights = task_weights
        # Setting these here enables to(device) / .cuda() to naturally affect them.

        self.task_permits_missingness = {'disch_24h', 'disch_48h', 'tasks_binary_multilabel'}
        self.task_losses = get_task_losses(self.task_class_weights)
        self.next_timepoint_reconstruction_loss = nn.MSELoss(reduction='none')

        # We do Rolling FTS separately
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )
        self.FTS_decoder = FutureTreatmentSequenceDecoder(
            decoder_module = LSTMDecoder(
                in_dim               = data_shape[0]*data_shape[1],
                treatment_embeddings = self.treatment_embeddings,
            ),
            predictor_module = SingleTaskPredictor(
                in_dim      = 25,
                num_classes = 9,
            ),
        )

    def freeze_representation(self):
        return
    
    def unfreeze_representation(self):
        return

    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,
        dfs, # Should be a dict...
    ):
        # # TODO(mmd): Embedding Features...
        # input_sequence     = self.ts_continuous_projector(ts_continuous)
        # statics_continuous = self.statics_projector(statics)
        # statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        # input_sequence += statics_continuous

        # ts_mask = ts_mask.expand_as(input_sequence)

        # TODO(mmd): Put type conversions in dataset.
        for k in (
            'ts', 'statics', 'ts_mask', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
        ):
            if k in dfs: dfs[k] = dfs[k].float()
        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_fts'):
            if k in dfs: dfs[k] = dfs[k].squeeze().long()

        #input_sequence = torch.cat((ts, statics), dim=2)
        input_sequence = dfs['input_sequence']
        batch_size, seq_len, feat_dim = list(input_sequence.shape)

        # Our "representation"
        pooled_output = input_sequence.view(batch_size, -1)

        # insert all the prediction tasks here
        task_labels = {
            k: df for k, df in dfs.items() if \
                    k not in ('statics', 'ts', 'rolling_fts', 'ts_mask') and df is not None
        }
        tasks = list(set(task_labels.keys()).intersection(self.task_heads.keys()))
        assert 'rolling_fts' in dfs or tasks, "Must have some tasks!"
        
#         print(tasks)
#         print(pooled_output.shape)
#         print(self.task_heads['Final Acuity Outcome'].weight.shape)
        task_logits = {t: self.task_heads[t](pooled_output) for t in tasks}

        #for t in tasks:
        #    if t not in self.task_losses: continue
        #    print(t, 'labels', dfs[t].dtype, dfs[t].shape, 'logits', task_logits[t].dtype, task_logits[t].shape)

        task_losses = {}
        for task, loss_fn, logits, labels, weight in zip_dicts(
            self.task_losses, task_logits, dfs, self.task_weights
        ):
            if task in self.task_permits_missingness:
                isnan = torch.isnan(labels)
                labels_smoothed = torch.where(isnan, torch.zeros_like(labels), labels)
                try:
                    if task in ('disch_24h', 'disch_48h'):
                        loss = torch.where(
                            isnan, torch.zeros_like(labels, dtype=torch.float32),
                            loss_fn(logits, labels_smoothed)
                        )
                    elif task in ('tasks_binary_multilabel',):
                        loss = (self.task_class_weights[task]!=0).float() * torch.where(isnan, torch.zeros_like(logits), loss_fn(logits, labels_smoothed))
                    else:
                        raise
                except:
                    raise
                loss = loss.mean()
                loss = weight * loss
            else: loss = weight * loss_fn(logits, labels)

            task_losses[task] = loss

        if 'rolling_fts' in dfs:
            fts_labels = dfs['rolling_fts']
            fts_logits, fts_loss = self.FTS_decoder(input_sequence.view(batch_size, -1), labels = fts_labels)
            task_logits['rolling_fts'] = fts_logits
            task_losses['rolling_fts'] = self.task_weights['rolling_fts'] * fts_loss
            tasks.append('rolling_fts')

        # We need to handle next timepoint separately to deal with masking.
        if 'next_timepoint' and 'next_timepoint_was_measured' in dfs:
            recst_loss = self.next_timepoint_reconstruction_loss(
                dfs['next_timepoint'], task_logits['next_timepoint']
            )
            recst_loss *= dfs['next_timepoint_was_measured'] # Mask out those not obs.

            # Accounting for the completely unmeasured rows.
            # TODO(mmd): we'll need to do the same for some of the classification tasks...
            num_measured_per_patient = dfs['next_timepoint_was_measured'].sum(dim=1)
            num_measured_per_patient = torch.where(
                num_measured_per_patient == 0, torch.ones_like(num_measured_per_patient),
                num_measured_per_patient
            )
            recst_loss = recst_loss.sum(dim=1) / num_measured_per_patient

            recst_loss = recst_loss.sum(dim=0)
            task_losses['next_timepoint'] = self.task_weights['next_timepoint'] * recst_loss

        total_loss = task_losses[tasks[0]]
        for t in tasks[1:]: total_loss += task_losses[t]

        return (
            None,
            pooled_output,
            {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks},
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )
