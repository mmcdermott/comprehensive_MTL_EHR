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
    for t in ('tasks_binary_multilabel', ):
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

POOLING_METHODS = ('max', 'avg', 'last')#, 'attention')
class GRUModel(nn.Module):
    """ TODO(this)
    """
    def __init__(
        self, config, data_shape=[48, 128], use_cuda=False, n_gpu = 0, task_class_weights=None,
        task_weights = None, hidden_dim=512, num_layers=2, bidirectional=False,
        pooling_method = 'last', fc_layer_sizes = [], verbose=False,
        do_eicu = False,
    ):
        super().__init__()

        self.verbose=verbose

        # TODO: need to activation masked_imputation task somehow...
        assert pooling_method in POOLING_METHODS, "Don't know how to do %s pooling" % pooling_method

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
        if do_eicu:
            self.task_dims = {
                'disch_24h': 10,
                'disch_48h': 10,
                'Final Acuity Outcome': 12,
                'tasks_binary_multilabel': 3,
                'next_timepoint': 15, # TODO put in config
                'next_timepoint_was_measured': 15,
                'masked_imputation': 15*2,
            }
        else:
            self.task_dims = {
                'disch_24h': 20,
                'disch_48h': 20,
                'Final Acuity Outcome': 20,
                'tasks_binary_multilabel': 26, #7, # ICD10, los, readmission
                'next_timepoint': 56, # TODO put in config
                'next_timepoint_was_measured': 56,
                'masked_imputation': 56*2,
            }

        self.masked_imputation_loss = nn.BCEWithLogitsLoss(reduction='none')

        # TODO(mmd): API?
        self.task_heads = nn.ModuleDict(
            {t: nn.Linear(config.hidden_size, d) for t, d in self.task_dims.items()}
        )
        if task_weights is None:
            self.task_weights = {t: 1 for t in self.task_heads.keys()}
            self.task_weights['rolling_ftseq'] = 1
            self.task_weights['masked_imputation'] = 0
            # In the case that we're doing all tasks as normal, presume we're doing no masking to mimic
            # prior behavior.
        else: self.task_weights = task_weights

        if 'masked_imputation' not in self.task_weights: self.task_weights['masked_imputation'] = 0
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

        # TODO(mmd): Put type conversions in dataset.
        for k in (
            'ts', 'statics', 'next_timepoint_was_measured', 'next_timepoint',
            'tasks_binary_multilabel',
            'ts_vals', 'ts_is_measured', 'ts_mask',
        ):
            if k in dfs: dfs[k] = dfs[k].float()
        for k in ('disch_24h', 'disch_48h', 'Final Acuity Outcome', 'rolling_ftseq'):
            if k in dfs: dfs[k] = dfs[k].squeeze().long()

        #input_sequence = torch.cat((ts, statics), dim=2)
        input_sequence = dfs['input_sequence']
        batch_size, seq_len, feat_dim = list(input_sequence.shape)

        if h_0 is None:
            h_0 = self.h_0

        if batch_size != 1:
            h_0 = h_0.expand(-1, batch_size, -1).contiguous()

        out_unpooled, h = self.gru(input_sequence, h_0) # for gru

        if self.pooling_method == 'last': out = out_unpooled[:, -1, :]
        elif self.pooling_method == 'max': out = out_unpooled.max(dim=1)[0]
        elif self.pooling_method == 'avg': out = out_unpooled.mean(dim=1)

        out = out.contiguous().view(batch_size, -1) # num directions is 1 for forward-only rnn

        pooled_output = self.fc_stack(out)
        unpooled_output = self.fc_stack(out_unpooled)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output is batch_size, hidden_dim

        # insert all the prediction tasks here
        task_labels = {
            k: df for k, df in dfs.items() if \
                    k not in ('statics', 'ts', 'rolling_ftseq', 'ts_mask') and df is not None
        }
        tasks = list(set(task_labels.keys()).intersection(self.task_heads.keys()))
        assert 'rolling_ftseq' in dfs or tasks, "Must have some tasks!"
        assert 'masked_imputation' not in tasks, "This task must be handled separately."

        task_logits = {t: self.task_heads[t](pooled_output) for t in tasks}

        #for t in tasks:
        #    if t not in self.task_losses: continue
        #    print(t, 'labels', dfs[t].dtype, dfs[t].shape, 'logits', task_logits[t].dtype, task_logits[t].shape)

        task_losses = {}
        weights_sum = 0
        for task, loss_fn, logits, labels, weight in zip_dicts(
            self.task_losses, task_logits, dfs, self.task_weights
        ):
            weights_sum += weight # We do it like this so that we only track tasks that are actually used.
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
                    print(dfs)
                    print(task, isnan.shape, labels.shape, logits.shape, labels_smoothed.shape)
                    raise
                loss = loss.mean()
                loss = weight * loss
            else: loss = weight * loss_fn(logits, labels)

            task_losses[task] = loss

        if 'rolling_ftseq' in dfs:
            weights_sum += self.task_weights['rolling_ftseq']
            fts_labels = dfs['rolling_ftseq']
            fts_logits, fts_loss = self.FTS_decoder(pooled_output, labels = fts_labels)
            task_logits['rolling_ftseq'] = fts_logits
            task_losses['rolling_ftseq'] = self.task_weights['rolling_ftseq'] * fts_loss
            tasks.append('rolling_ftseq')

        # We need to handle next timepoint separately to deal with masking.
        if 'next_timepoint' and 'next_timepoint_was_measured' in dfs:
            weights_sum += self.task_weights['next_timepoint']
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
            #task_losses['next_timepoint'] = self.task_weights['next_timepoint'] * recst_loss
            # Setting this to 0 here as we always want this ablated. This is a poor solution, but doing this
            # to avoid any issues.
            task_losses['next_timepoint'] = 0 * recst_loss

        # We need to handle next timepoint separately to deal with masking.
        if 'masked_imputation' in self.task_weights and self.task_weights['masked_imputation'] > 0:
            assert 'ts_vals' in dfs and 'ts_is_measured' in dfs and 'ts_mask' in dfs, \
                'Expected masked items in the dataset'

            weights_sum += self.task_weights['masked_imputation']

            # unpooled_output is of shape: batch_size X seq_len X feat_dim
            imputation_scores = self.task_heads['masked_imputation'](unpooled_output)
            # imputation_scores is of shape: batch_size X seq_len X task_dims['masked_imputation']
            # TODO(mmd): This should be associated to the task dimensionality.

            N = self.task_dims['masked_imputation'] // 2
            wbm_logits = imputation_scores[:, :, :N]
            imp_preds  = imputation_scores[:, :, N:]

            # per-timepoint:
            #   1) cross entropy loss on wbm logits (multi-label binary)
            #   2) euclidean loss on imp_preds (continuous value) <-- TODO should this be also probabilistic?
            #      masked to only be applied _when the value is observed_

            per_feature_wbm_loss = self.masked_imputation_loss(wbm_logits, dfs['ts_is_measured'])
            per_timepoint_wbm_loss = per_feature_wbm_loss.mean(dim=2)

            per_feature_imp_loss = (imp_preds - dfs['ts_vals'])**2

            num_measured_per_timepoint_real = dfs['ts_is_measured'].sum(dim=2)
            num_measured_per_timepoint_smoothed = torch.where(
                num_measured_per_timepoint_real == 0, torch.ones_like(num_measured_per_timepoint_real),
                num_measured_per_timepoint_real
            )

            # Should possibly use some probabilistic loss.
            per_timepoint_imp_loss = (
                (per_feature_imp_loss * dfs['ts_is_measured']).sum(dim=2)
                /
                num_measured_per_timepoint_smoothed
            )
            # the loss is uniformly zero on timepoints where nothing was measured, and otherwise is the RMSE.
            # Use the MSE loss to avoid sqrt giving nans (unknown why sqrt is giving nans)
            # per_timepoint_imp_loss = torch.where(
            #     num_measured_per_timepoint_real == 0, torch.zeros_like(per_timepoint_imp_loss),
            #     torch.sqrt(per_timepoint_imp_loss)
            # )

            #   Next, sum these two losses to obtain the per-timepoint loss.
            per_timepoint_loss = per_timepoint_wbm_loss + per_timepoint_imp_loss
            # per_timepoint_loss is of shape batch_size X seq_len

            # Then, this entire loss per time-point is masked according to which time-points were masked.
            # dfs['mask_indicators'] is binary and of shape batch_size X seq_len
            # TODO(mmd): Make the `.squeeze()` not necessary.
            per_timepoint_loss = per_timepoint_loss * dfs['ts_mask'].squeeze()
            per_seq_loss = per_timepoint_loss.sum(dim=1)

            # Finally, we want to scale by the # of masks per sequence. We set this scaling factor to `1` when
            # it is actually 0 to avoid a later divide by zero error, noting that by 
            # `per_timepoint_loss = per_timepoint_loss * dfs['mask_indicators']`
            # above the numerator in that case is guaranteed to be zero.
            num_masks_per_seq = dfs['ts_mask'].squeeze().sum(dim=1)
            num_masks_per_seq = torch.where(
                num_masks_per_seq == 0, torch.ones_like(num_masks_per_seq),
                num_masks_per_seq
            )
            per_seq_loss = per_seq_loss / num_masks_per_seq

            per_batch_loss = per_seq_loss.sum(dim=0)
            task_losses['masked_imputation'] = self.task_weights['masked_imputation'] * per_batch_loss

        try:
            total_loss = None
            for l in task_losses.values():
                total_loss = l if total_loss is None else (total_loss + l)
            total_loss /= weights_sum
        except:
            print(task_losses)
            print(weights_sum)
            raise

        # formerly returned: {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks}, total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss

        out_data = {t: (task_logits[t], dfs[t], task_losses[t]) for t in tasks}
        if 'masked_imputation' in self.task_weights and self.task_weights['masked_imputation'] > 0:
            try:
                out_data['masked_imputation'] = tuple([
                    (wbm_logits, imp_preds),
                    (dfs['ts_is_measured'], dfs['ts_vals'], dfs['ts_mask']),
                    (
                        per_timepoint_wbm_loss, per_timepoint_imp_loss, task_losses['masked_imputation'],
                        per_timepoint_loss, per_seq_loss, per_batch_loss
                    )
                ])
            except:
                print(type(out_data), out_data)
                for e in (
                    wbm_logits, imp_preds, dfs['ts_is_measured'], dfs['ts_vals'], dfs['ts_mask'],
                    per_timepoint_wbm_loss, per_timepoint_imp_loss, task_losses['masked_imputation']
                ):
                    print(type(e))
                    try: print(e.shape)
                    except: print("not a tensor", e)
                raise

        return (
            None,
            pooled_output,
            out_data,
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )
