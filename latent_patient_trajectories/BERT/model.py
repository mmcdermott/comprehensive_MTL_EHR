# Adapted from
# https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/pytorch_pretrained_bert
# at commit e6cf62d49945e6277b5e4dc855f9186b3f789e35
from ..pytorch_helpers import *

import numpy as np, torch

from pytorch_pretrained_bert.modeling import (
    BertEncoder, BertPooler, BertPredictionHeadTransform, BertPreTrainedModel, BertModel, BertLayerNorm,
    BertConfig
)
from torch import nn

class ContinuousBertConfig(BertConfig):
    """ An extension of the BERT Config to store continuous params as well.
    """
    def __init__(self, *args, in_dim=-1, **kwargs):
        super().__init__(0, *args, **kwargs)
        self.in_dim = in_dim

class ContinuousBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # should eliminate the below two and fold into meta_model.
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, sequence, token_type_ids=None):
        #input ids should be in the form of (batch_size, time_steps, feature_size)
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence[:, :, 0])
        if token_type_ids is None: token_type_ids = torch.zeros_like(sequence[:, :, 0]).long()

        sequence_embeddings = sequence
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = sequence_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ContinuousBertModel(BertPreTrainedModel):
    """Continuous BERT model ("Bidirectional Embedding Representations from a Transformer").
    Just like BERT but a different kind embedding layer.

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `sequence`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # TODO(): Update
    sequence = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(sequence, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super().__init__(config)
        self.embedder = ContinuousBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, sequence, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None: attention_mask = torch.ones_like(sequence[:, :, 0]).long()
        if token_type_ids is None: token_type_ids = torch.zeros_like(sequence[:, :, 0]).long()

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedded_sequence = self.embedder(sequence, token_type_ids)
        encoded_layers = self.encoder(
            embedded_sequence, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers: encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

#class BertReconstructionPredictionHead(nn.Module):
#    def __init__(self, config):
#        super().__init__()
#        self.transform = BertPredictionHeadTransform(config) # May not need the transform here.
#        self.decoder = nn.Linear(config.hidden_size, config.in_dim, bias=True)
#
#    def forward(self, hidden_states): return self.decoder(self.transform(hidden_states))
#
#class ContinuousBertPreTrainingHeads(nn.Module):
#    def __init__(self, config, task_dims={}):
#        super().__init__()
#        self.predictions = BertReconstructionPredictionHead(config)
#        self.seq_predictions = MultitaskHead(config, task_dims)
#
#    def forward(self, sequence_output, pooled_output):
#        prediction_scores = self.predictions(sequence_output)
#        seq_scores = self.seq_predictions(pooled_output)
#        return prediction_scores, seq_scores
#
#class ContinuousBertForPreTraining(BertPreTrainedModel):
#    """BERT model with continuous reconstruction pre-training heads.
#    Also supports additional, user-specified auxiliary losses.
#
#    This module comprises the BERT model followed by the two pre-training heads:
#        - the masked continuous reconstruction modeling head, and
#        - the next sequence classification head.
#    as well as any user specified auxiliary loss prediction heads.
#
#    Params:
#        config: a BertConfig class instance with the configuration to build a new model.
#
#    Inputs:
#        `sequence`: a torch.LongTensor of shape [batch_size, sequence_length, hidden_size]
#        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#            types indices selected in [0, 1]. Type 0 corresponds to a `sequence A` and type 1 corresponds to
#            a `sequence B` token (see BERT paper for more details).
#        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#            input sequence length in the current batch. It's the mask that we typically use for attention when
#            a batch has varying length sequences.
#        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
#            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
#            is only computed for the labels set in [0, ..., vocab_size]
#        `next_sequence_label`: optional next sequence classification loss: torch.LongTensor of shape [batch_size]
#            with indices selected in [0, 1].
#            0 => next sequence is the continuation, 1 => next sequence is a random sequence.
#
#    Outputs:
#        if `masked_lm_labels` and `next_sequence_label` are not `None`:
#            Outputs the total_loss which is the sum of the masked language modeling loss and the next
#            sequence classification loss.
#        if `masked_lm_labels` or `next_sequence_label` is `None`:
#            Outputs a tuple comprising
#            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
#            - the next sequence classification logits of shape [batch_size, 2].
#
#    Example usage:
#    ```python
#    # Already been converted into WordPiece token ids
#    sequence = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
#    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
#    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
#
#    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
#
#    model = BertForPreTraining(config)
#    masked_lm_logits_scores, seq_relationship_logits = model(sequence, token_type_ids, input_mask)
#    ```
#    """
#    def __init__(self, config, seq_task_dims, lambda_seq_tasks=1):
#        super().__init__(config)
#        self.bert = ContinuousBertModel(config)
#        self.cls = ContinuousBertPreTrainingHeads(config, seq_task_dims)
#        self.lambda_seq_tasks = lambda_seq_tasks
#        self.apply(self.init_bert_weights)
#
#    def forward(
#        self,
#        masked_sequence_targets, input_sequence, attention_mask, token_type_ids, el_was_masked,
#        whole_sequence_labels, whole_sequence_labels_present,
#    ):
#        # TODO(mmd): Auxiliary Losses.
#        sequence_output, pooled_output = self.bert(input_sequence, token_type_ids, attention_mask,
#                                                   output_all_encoded_layers=False)
#        reconstructed_sequence, seq_scores = self.cls(sequence_output, pooled_output)
#
#        total_loss = 0
#        if masked_sequence_targets is not None:
#            reconstruction_loss_fct = nn.MSELoss(reduction="none")
#            masked_reconstruction_loss = reconstruction_loss_fct(
#                reconstructed_sequence, masked_sequence_targets
#            )
#            masked_reconstruction_loss *= el_was_masked.unsqueeze(2).expand_as(masked_reconstruction_loss)
#            masked_reconstruction_loss = (masked_reconstruction_loss.sum())/(el_was_masked.sum())
#            total_loss = masked_reconstruction_loss + total_loss
#
#        for task, label in whole_sequence_labels.items():
#            labels_present = whole_sequence_labels_present[task]
#            labels_present_sum = labels_present.sum()
#
#            scores = seq_scores[task]
#
#            # TODO(mmd): Generalize
#            whole_sequence_loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
#
#            whole_sequence_loss = whole_sequence_loss_fct(scores, label.view(-1))
#            whole_sequence_loss = torch.where(
#                labels_present == 1, whole_sequence_loss, torch.zeros_like(whole_sequence_loss)
#            ).sum()
#
#            whole_sequence_loss = torch.where(
#                labels_present_sum > 0,
#                whole_sequence_loss/labels_present_sum,
#                torch.zeros_like(whole_sequence_loss)
#            )
#
#            total_loss = total_loss + self.lambda_seq_tasks * whole_sequence_loss
#
#        return pooled_output, total_loss, reconstructed_sequence, seq_scores
