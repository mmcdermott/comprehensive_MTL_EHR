"""
model.py
This contains the source for the pytorch model which learns how to represent
"""

import glob, os, random, numpy as np, pandas as pd
import torch, torch.optim, torch.nn as nn, torch.nn.functional as F
from typing import Sequence
from dataclasses import dataclass
from torch.autograd import Variable, set_detect_anomaly
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

idx = pd.IndexSlice

from ..utils import *
from ..constants import *
from ..data_utils import *
from ..representation_learner.fts_decoder import *
from ..BERT.model import *
from ..BERT.constants import *
from .utils import *

# TODO(mmd): Move to utils.
def fts_decoder_loss(logits, fts_label):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    mask = (fts_label != 0).float()

    loss = (loss_fct(logits, fts_label) * mask).sum(dim=1)
    num_labels_per_el = mask.sum(dim=1)
    num_labels_per_el_or_one = torch.where(
        num_labels_per_el > 0, num_labels_per_el, torch.ones_like(num_labels_per_el)
    ) #loss will be zero where num_labels_per_el is zero and we want to avoid nans.
    loss = (loss / num_labels_per_el_or_one).mean(dim=0) # TODO: validate loss scale.
    return logits, loss

def single_label_loss(logits, labels): F.cross_entropy(logits, labels)
def multi_label_loss(logits, labels): F.multilabel_soft_margin_loss(logits, labels)
def mse_loss(X, Y): F.mse_loss(X, Y)

class GenericPredictor(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        loss_fct
        out_net = nn.Linear
    ):
        super().__init__()
        self.out_net  = out_net(in_dim, out_dim)
        self.loss_fct = loss_fct

    def forward(self, X, labels=None):
        out = self.out_net(X)
        if labels is None: return out, None

        return out, self.loss_fct(out, labels)

def get_tasks_dict(config):
    """
    returns {task: (task_weight, task_head (nn.Module, forward(X, labels=None): out/logits, loss (if labels)}
    X = pooled_output.
    """
    heads = {}
    heads['next_timepoint'] = GenericPredictor(config.hidden_size, config.num_feat, mse_loss)
    heads['next_timepoint_was_measured'] = GenericPredictor(
        config.hidden_size, config.num_feat, multi_label_loss
    )
    heads['static_tasks_continuous'] = GenericPredictor(config.hidden_size, ???, mse_loss)
    heads['rolling_fts'] = FutureTreatmentSequenceDecoder(
        decoder_module = LSTMDecoder(
            in_dim               = config.hidden_size,
            treatment_embeddings = self.treatment_embeddings,
        ),
        predictor_module = SingleTaskPredictor(
            in_dim      = 25, # TODO(mmd): Make params!!
            num_classes = 9,
        ),
    )
    heads['rolling_tasks_continuous'] = GenericPredictor(config.hidden_size, ???, mse_loss)

    # TODO(mmd): Re-do extractors such that it isn't so dumb.

    return {k: (1, v) for k, v in heads.items()}
        'rolling_tasks_to_embed':      (1.0, FullyConnectedNet),
        'static_tasks_to_embed':       (1.0, FullyConnectedNet),
    }

class SelfAttentionTimeseries(BertPreTrainedModel):
    """ TODO(this)
    """
    def __init__(
        self, config, use_cuda=False, tasks={},
    ):
        super().__init__(config)
        self.bert = ContinuousBertModel(config)
#         self.cls = ContinuousBertPreTrainingHeads(config) # modify this to get all of the necessary tasks
        self.apply(self.init_bert_weights)
        self.use_cuda = use_cuda

        self.tasks = nn.ModuleDict(tasks) # {task: (weight, head)}

        self.ts_continuous_projector      = nn.Linear(config.ts_feat_dim, config.hidden_dim)
        self.statics_continuous_projector = nn.Linear(config.statics_feat_dim, config.hidden_dim)

        self.embedders = None # TODO

        # additional losses
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )

    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,

        # Inputs:
        ts_continuous = None, # batch X seq_len X features
        ts_to_embed = None,   # batch X seq_len X features
        ts_mask = None,       # batch X seq_len X 1
        statics = None,       # batch X features

        # Tasks:
        **tasks_kwargs,
        #rolling_fts = None,
        #rolling_tasks_continuous = None,
        #rolling_tasks_to_embed = None,
        #static_tasks_continuous = None,
        #static_tasks_to_embed = None,
        #next_timepoint = None,
        #next_timepoint_was_measured = None,
    ):
        # TODO(mmd): Embedding Features...
        input_sequence     = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous

        ts_mask = ts_mask.expand_as(input_sequence)

        _, pooled_output = self.bert(input_sequence, None, ts_mask,
                                                   output_all_encoded_layers=False)
        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output.shape is batch_size, hidden_dim

        total_loss, tasks_out = 0, {}
        for task_name in set(self.tasks.keys()).intersection(tasks_kwargs.keys()):
            weight, head = self.tasks[task_name]
            task_labels = tasks_kwargs[task_label]
            out, loss = head(pooled_output, task_labels)
            tasks_out[task_name] = (out, task_labels, loss)
            if loss is not None: total_loss += weight * loss

        return (
            pooled_output,
            tasks_out,
            total_loss
        )





class CNN(nn.Module):
    """ TODO(this)
    """
    def __init__(
        self, config, use_cuda=False, tasks={}, conv_layers = [10, 100, 20], filt_size= [7, 5, 5]
    ):
        super(CNN).__init__(config)


        # conv 2d appraoch
        self.conv1 = nn.Conv2d(1, conv_layers[0], (filt_size[0],1)) # in channels, out channels, kernel size
        self.conv2 = nn.Conv2d(conv_layers[0], conv_layers[1], (filt_size[1], 1))
        self.conv3 = nn.Conv2d(conv_layers[1], conv_layers[2], (filt_size[2], 1))


        self.fc1 = nn.Linear(7840, 512)
        self.fc2 = nn.Linear(512, config.hidden_dim)
        self.relu = nn.ReLU()

#         self.cls = ContinuousBertPreTrainingHeads(config) # modify this to get all of the necessary tasks
        self.apply(self.init_bert_weights)
        self.use_cuda = use_cuda

        self.tasks = nn.ModuleDict(tasks) # {task: (weight, head)}

        self.ts_continuous_projector      = nn.Linear(config.ts_feat_dim, config.hidden_dim)
        self.statics_continuous_projector = nn.Linear(config.statics_feat_dim, config.hidden_dim)

        self.embedders = None # TODO

        # additional losses
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )

    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,

        # Inputs:
        ts_continuous = None, # batch X seq_len X features
        ts_to_embed = None,   # batch X seq_len X features
        ts_mask = None,       # batch X seq_len X 1
        statics = None,       # batch X features

        # Tasks:
        **tasks_kwargs,
        #rolling_fts = None,
        #rolling_tasks_continuous = None,
        #rolling_tasks_to_embed = None,
        #static_tasks_continuous = None,
        #static_tasks_to_embed = None,
        #next_timepoint = None,
        #next_timepoint_was_measured = None,
    ):
        # TODO(mmd): Embedding Features...
        input_sequence     = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous

        ts_mask = ts_mask.expand_as(input_sequence)

        # replace with conv layers
        # _, pooled_output = self.bert(input_sequence, None, ts_mask,
                                                   # output_all_encoded_layers=False)

        print(input_sequence.shape)
        input_sequence.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        pooled_output = self.fc2(x)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output.shape is batch_size, hidden_dim

        total_loss, tasks_out = 0, {}
        for task_name in set(self.tasks.keys()).intersection(tasks_kwargs.keys()):
            weight, head = self.tasks[task_name]
            task_labels = tasks_kwargs[task_label]
            out, loss = head(pooled_output, task_labels)
            tasks_out[task_name] = (out, task_labels, loss)
            if loss is not None: total_loss += weight * loss

        return (
            pooled_output,
            tasks_out,
            total_loss
        )


class GRUModel(nn.Module):
    """ TODO(this)
    """
    def __init__(
        self, config, device, use_cuda=False, tasks={}, hidden_dim=512, num_layers=2, drop_prob=0.2,
    ):
        super(GRUModel).__init__(config)
        
        # initialise the model and the weights
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size=config.input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=drop_prob, bidirectional=False)

        # self.fc = nn.Linear(hidden_dim*2, output_dim) # for bidirectional
        self.fc = nn.Linear(hidden_dim*config.input_dim, output_dim) # not bidirectional
        self.relu = nn.ReLU()
#         self.h_0 = torch.zeros(2*n_layers,  1, hidden_dim).float().to('cuda:0') # for bidirectional
        self.h_0 = torch.zeros(n_layers,  1, hidden_dim).float().to('cuda:0' if use_cuda else 'cpu') # not bidirectional
        self.hidden_dim=hidden_dim







#         self.cls = ContinuousBertPreTrainingHeads(config) # modify this to get all of the necessary tasks
        self.apply(self.init_bert_weights)
        self.use_cuda = use_cuda

        self.tasks = nn.ModuleDict(tasks) # {task: (weight, head)}

        self.ts_continuous_projector      = nn.Linear(config.ts_feat_dim, config.hidden_dim)
        self.statics_continuous_projector = nn.Linear(config.statics_feat_dim, config.hidden_dim)

        self.embedders = None # TODO

        # additional losses
        self.treatment_embeddings = nn.Embedding(
            num_embeddings = 9,  # TODO(mmd): Actually set this...
            embedding_dim  = 25  # TODO(mmd): Actually set this... Belongs in config...
        )

    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,

        # Inputs:
        ts_continuous = None, # batch X seq_len X features
        ts_to_embed = None,   # batch X seq_len X features
        ts_mask = None,       # batch X seq_len X 1
        statics = None,       # batch X features
        h_0 = None

        # Tasks:
        **tasks_kwargs,
        #rolling_fts = None,
        #rolling_tasks_continuous = None,
        #rolling_tasks_to_embed = None,
        #static_tasks_continuous = None,
        #static_tasks_to_embed = None,
        #next_timepoint = None,
        #next_timepoint_was_measured = None,
    ):
        # TODO(mmd): Embedding Features...
        input_sequence     = self.ts_continuous_projector(ts_continuous)
        statics_continuous = self.statics_projector(statics)
        statics_continuous = statics_continuous.unsqueeze(1).expand_as(input_sequence)

        input_sequence += statics_continuous

        ts_mask = ts_mask.expand_as(input_sequence)


         if h_0 is None:
            h_0 = self.h_0
        
        bs = x.shape[0]
        if bs!=1:
            h_0 = h_0.expand(-1, bs, -1).contiguous()
            
#         out, (h, c) = self.gru(x, (h_0, c_0)) # for lstm
        out, h = self.gru(input_sequence, h_0) # for gru

#         out = out.view(-1, 2 * self.hidden_dim) # num directions is 2 for forward bachward rnn
        out = out.contiguous().view(bs, -1) # num directions is 1 for forward rnn
        pooled_output = self.fc(self.relu(out))


        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output.shape is batch_size, hidden_dim

        total_loss, tasks_out = 0, {}
        for task_name in set(self.tasks.keys()).intersection(tasks_kwargs.keys()):
            weight, head = self.tasks[task_name]
            task_labels = tasks_kwargs[task_label]
            out, loss = head(pooled_output, task_labels)
            tasks_out[task_name] = (out, task_labels, loss)
            if loss is not None: total_loss += weight * loss

        return (
            pooled_output,
            tasks_out,
            total_loss
        )