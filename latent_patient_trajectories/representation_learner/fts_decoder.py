from ..pytorch_helpers import *
from .constants import *

import torch, torch.nn as nn

def mask_and_avg_sequential_loss(loss_tensor, seq_lengths):
    # loss     = [batch, seq_len]
    # seq_lens = [batch,]

    max_seq_len      = loss_tensor.shape[1]
    seq_indices      = torch.arange(max_seq_len).unsqueeze(0).expand_as(loss)
    seq_lengths      = seq_lengths.unsqueeze(1).expand_as(loss)
    seq_lengths_mask = seq_indices < seq_lengths

    loss_masked          = torch.where(seq_lengths_mask, loss_tensor, torch.zeros_like(loss_tensor))
    loss_masked_averaged = loss_masked.sum(dim=1) / seq_lengths_mask.float().sum(dim=1)

    return loss_masked_averaged

def embed_classifier(
    in_dim,
    out_dim,
    embeddings_or_none = None,
):
    if embeddings_or_none is None: return nn.Linear(in_dim, out_dim)

    num_elements, embedding_dim = embeddings_or_none.shape

    assert out_dim == embedding_dim, "Embeddings mismatched for output!"

    classifier = nn.Linear(embedding_dim, num_elements)
    classifier.weight = embeddings_or_none

    if in_dim == embedding_dim: return classifier

    projection = nn.Linear(in_dim, embedding_dim)
    return nn.Sequential(projection, classifier)

### Predictors
#   We try two predictors--one which uses the fact that the treatments sub-part of this task is actually
#   multi-task. The other just embeds all combinations and outputs over all.

# TODO: make actually work.
# class MultiTaskPredictor(nn.Module):
#     def __init__(
#         self,
#         in_dim,
#         treatments, # Assumes all are binary for now.
#         mort_pp_dim              = len(MortalityPPLabels),
#         treatments_embed_weights = None,
#         mort_pp_embed_weights    = None,
#     ):
#         super().__init__()
# 
#         self.log_softmax = nn.LogSoftmax() # TODO(mmd): dim
#         self.log_sigmoid = nn.LogSigmoid()
#         self.nll_loss    = nn.NLLLoss(reduction='none')
# 
#         self.treatments_classifier = embed_classifier(in_dim, len(treatments), treatments_embed_weights)
#         self.mort_pp_classifier    = embed_classifier(in_dim, mort_pp_dim, mort_pp_embed_weights)
#         self.is_at_end_classifier  = nn.Linear(in_dim, 1)
# 
#     def forward(self, X, mort_pp_label, treatment_label):
#         is_at_end_logit = self.is_at_end_classifier(X)
#         mort_pp_logits = self.mort_pp_classifier(X)
#         treatments_logits = self.treatments_classifier(X)
# 
#         # generative process -> choose if end, if so, mort_pp, else, treatment_label.
# 
#         mort_pp_log_probs = self.log_softmax(mort_pp_logits) + self.log_sigmoid(is_at_end_logit)
#         treatments_log_probs = self.log_sigmoid(treatments_logits) + self.log_sigmoid(-is_at_end_logit)
#         treatments_log_complement_probs = self.log_sigmoid(-treatments_logits) + self.log_sigmoid(-is_at_end_logit)
# 
#         loss = torch.where(
#             torch.isna(mort_pp_label),
#             torch.where(
#                 treatment_label == 1,
#                 treatments_log_probs,
#                 treatments_log_complement_probs,
#             ),
#             self.nll_loss(mort_pp_log_probs, mort_pp_label),
#         )
# 
#         assert not torch.isnan(loss), "FTS Decoding Loss should not be NaN!"

class SingleTaskPredictor(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
    ):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
        self.loss_fct   = nn.CrossEntropyLoss(reduction='none')

    def forward(self, X, fts_label=None, mort_pp_labels=None):
        logits = self.classifier(X).transpose(1, 2)
        if fts_label is None: return logits, None

        mask = (fts_label != 0).float()

        loss = (self.loss_fct(logits, fts_label) * mask).sum(dim=1)
        num_labels_per_el = mask.sum(dim=1)
        num_labels_per_el_or_one = torch.where(
            num_labels_per_el > 0, num_labels_per_el, torch.ones_like(num_labels_per_el)
        ) #loss will be zero where num_labels_per_el is zero and we want to avoid nans.
        loss = (loss / num_labels_per_el_or_one).mean(dim=0) # TODO: validate loss scale.
        return logits, loss

### Decoders
#   We experiment with several decoders
def fts_labels_to_embeddings(fts_labels, treatment_embeddings):
    return torch.matmul(fts_labels, treatment_embeddings)

class LSTMDecoder(nn.Module):
    HIDDEN_SIZE = 'hidden_size'

    def __init__(
        self, in_dim, treatment_embeddings,
        lstm_kwargs={'dropout': 0, }
    ):
        super().__init__()

        num_treatments, embedding_dim = treatment_embeddings.weight.shape
        self.treatment_embeddings     = treatment_embeddings

        if self.HIDDEN_SIZE not in lstm_kwargs: lstm_kwargs[self.HIDDEN_SIZE] = embedding_dim
        lstm_kwargs['batch_first'] = True
        lstm_kwargs['bidirectional'] = False
        lstm_kwargs['input_size'] = lstm_kwargs[self.HIDDEN_SIZE]

        self.hidden_size     = lstm_kwargs[self.HIDDEN_SIZE]
        self.sequence_in_dim = embedding_dim

        self.C_proj = nn.Linear(in_dim, self.hidden_size)
        self.H_proj = nn.Linear(in_dim, self.hidden_size)
        self.X_proj = nn.Linear(in_dim, embedding_dim)
        self.LSTM   = nn.LSTM(**lstm_kwargs)

    def forward(self, decoded_state, fts_labels, use_teacher_forcing=True):
        init_c = self.C_proj(decoded_state).unsqueeze(0)
        init_h = self.H_proj(decoded_state).unsqueeze(0) # I don't know why the unsqueeze is needed...
        init_x = self.X_proj(decoded_state).unsqueeze(1)

        assert use_teacher_forcing, "Doesn't support non-forcing yet."

        input_sequence = torch.cat((init_x, self.treatment_embeddings(fts_labels[:, :-1])), dim=1)
        # fts_labels_to_embeddings(fts_labels, self.treatment_embeddings)

        return self.LSTM(input_sequence, (init_h, init_c))[0]

### Overall Module
#   By pushing complexity up, we keep this module very simple.
#   TODO(mmd): Make actually work.
class FutureTreatmentSequenceDecoder(nn.Module):
    def __init__(
        self,
        decoder_module,
        predictor_module,
    ):
        super().__init__()

        self.decoder, self.predictor = decoder_module, predictor_module

    def forward(self, X, labels=None, mort_pp_labels=None):
        # TODO(mmd): Figure this all out.

        decoded_state = self.decoder(X, labels)
        predictions, loss = self.predictor(decoded_state, labels, mort_pp_labels)
        return predictions, loss
