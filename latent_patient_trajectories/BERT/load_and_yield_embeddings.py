# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pdb

import os, random, tables, torch, warnings, numpy as np, pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm_notebook

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

from .data_processor import *

def load_and_get_ids(
    df,
    bert_model,
    processor,
    processor_args,
    do_lower_case,
    max_seq_length
):
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    examples = processor.get_examples(df, **processor_args)
    
    features = convert_examples_to_features(
        examples, None, max_seq_length, tokenizer
    )

    all_input_ids = [f.input_ids for f in features]
    all_input_mask = [f.input_mask for f in features]
    all_segment_ids = [f.segment_ids for f in features]
    return (all_input_ids, all_input_mask, all_segment_ids)


def load_and_get_outputs(
    df,
    bert_model,
    processor,
    model_class                 = BertModel, # TODO(mmd): Better init
    model_kwargs                = {},
    processor_args              = {},
    use_gpu                     = False,
    seed                        = 42,
    do_lower_case               = False,
    max_seq_length              = 128,
    batch_size                  = 8,
    learning_rate               = 5e-5,
    num_train_epochs            = 3,
    cache_dir                   = None,
    tqdm                        = tqdm_notebook,
    save_to_filepath            = None,
    chunksize                   = 0,
    start_at                    = 0,
    categories                  = None
):

    print('Loading pretrained model from %s' % bert_model)
    if chunksize > 0: assert save_to_filepath is not None, "Must save if chunking!"
    if start_at > 0:
        print('Filtering df down to rows %d - %d' % (start_at, len(df)))
        df = df.iloc[start_at:]

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu  = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Prepare model
    ### TODO(mmd): this is where to reload the model properly.
    cache_dir = cache_dir if cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_-1')
    # Load a trained model and config that you have fine-tuned
    model = model_class.from_pretrained(
        bert_model,
        cache_dir=cache_dir,
        **model_kwargs,
    )

    model.to(device)
    if n_gpu > 1: model = torch.nn.DataParallel(model)

    all_input_ids, all_input_mask, all_segment_ids = load_and_get_ids(
        df, bert_model, processor, processor_args, do_lower_case, max_seq_length
    )

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

    # Run prediction for full data
    all_outputs, start, end = generate_input_embeddings(
        model, all_input_ids, all_input_mask, all_segment_ids, device,
        batch_size = batch_size, chunksize=chunksize, tqdm=tqdm, start_at=start_at
    )

    output_df = pd.DataFrame(all_outputs, index=df.index[start:end])
    if save_to_filepath is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', tables.NaturalNameWarning)
            if categories:
                output_df.to_hdf(save_to_filepath, categories)
            else:
                output_df.to_hdf(save_to_filepath, '[%d-%d)' % (start_at + start, start_at + end))
    else: return output_df


def generate_input_embeddings(
    model, all_input_ids, all_input_mask, all_segment_ids, device,
    batch_size = 8, chunksize=0, tqdm=tqdm_notebook, start_at=0, disable_tqdm=False
):

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    model.eval()

    all_outputs, start = [], 0
    tqdm_loader = tqdm(dataloader, desc="Saved @ %d" % start_at, disable=False)
    for i, (input_ids, input_mask, segment_ids) in enumerate(tqdm_loader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad(): _, output = model(input_ids, segment_ids, input_mask)
        all_outputs.extend(output.detach().cpu().numpy())

        if chunksize > 0 and i > 0 and i % chunksize == 0:
            end = start + len(all_outputs)
            output_df = pd.DataFrame(all_outputs, index=df.index[start:end])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', tables.NaturalNameWarning)
                output_df.to_hdf(save_to_filepath, '[%d-%d)' % (start_at + start, start_at + end))
            tqdm_loader.set_description("Saved @ %d" % (start_at + end))

            start = end
            all_outputs = []

    end = start + len(all_outputs)

    return (all_outputs, start, end)

def generate_input_embeddings_test(
    model, all_input_ids, all_input_mask, all_segment_ids, device,
    batch_size = 4, tqdm=tqdm_notebook
):
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    all_outputs = []
    for i, (input_ids, input_mask, segment_ids) in enumerate(dataloader):
        _, output = model(input_ids, segment_ids, input_mask)
        all_outputs.append(output)

    return all_outputs
