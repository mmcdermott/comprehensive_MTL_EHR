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

import gc, math, os, random, torch, numpy as np

from torch.autograd import detect_anomaly
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, tqdm_notebook

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam


from .continuous_pretraining_data_processor import *
from .model import *

import glob
from multiprocessing import Pool
from functools import partial


def flatten(arr):
  if type(arr) is np.ndarray: return np.reshape(arr, [len(arr), -1])
  elif type(arr) is list:
    r = []
    for l in arr: r += l
    return r
  raise NotImplementedError


def process_chunk(chunk_num, processor_filepath, examples_filepath, max_seq_length, tqdm, featurization_kwargs):
    
    chunk_num, processor_file, example_file=chunk_num
    print(chunk_num)


    # print("Loading ",(processor_file))
    with open(processor_file, mode='rb') as f:
        examples = pickle.load(f)

    train_features = convert_examples_to_features(
        examples, max_seq_length, tqdm=tqdm, **featurization_kwargs
    )
    
    # print(example_file)
    with open(example_file, mode='wb') as f:
        pickle.dump(train_features, f)
    print("completed: ",example_file)

    del train_features
    del examples
    gc.collect()
    return

def build_dataset(
    df,
    processor,
    dataset_dir,
    featurization_kwargs        = {},
    seed                        = 42,
    max_seq_length              = 128,
    tqdm                        = tqdm_notebook,
    parallel                    = False
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    whole_sequence_tasks = {SEQUENCES_ORDERED: len(LABEL_ENUMS[SEQUENCES_ORDERED])}

    if dataset_dir is not None and not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)

    dataset_filepath   = os.path.join(dataset_dir, DATASET_FILENAME) if dataset_dir is not None else None
    processor_filepath = os.path.join(dataset_dir, PROCESSOR_FILENAME) if dataset_dir is not None else None
    examples_filepath  = os.path.join(dataset_dir, EXAMPLES_FILENAME) if dataset_dir is not None else None

    assert dataset_filepath is not None, "Sorry, this code is actually terrible."
    print("Building dataset from scratch.")

    if os.path.isfile(examples_filepath):
        del df

        print("Loading featurized list directly from %s" % examples_filepath)
        with open(examples_filepath, mode='rb') as f: train_features = pickle.load(f)
    elif os.path.isfile('%s.chunk_0' % examples_filepath):
        print("Loading distributed training examples from %s" % examples_filepath)
        chunk_num = 0
        while chunk_num >= 0 and chunk_num <= 159: #os.path.isfile('%s.chunk_%d' % (examples_filepath, chunk_num)):
            dataset_chunk_path = '%s.chunk_%d' % (dataset_filepath, chunk_num)
            examples_chunk_path = '%s.chunk_%d' % (examples_filepath, chunk_num)
            if os.path.isfile(dataset_chunk_path):
                chunk_num += 1
                continue
            elif not os.path.isfile(examples_chunk_path):
                print("Skipping non-existent chunk: %s" % examples_chunk_path)
                chunk_num += 1
                continue

            print("Loading %s" % examples_chunk_path)
            with open(examples_chunk_path, mode='rb') as f:
                train_features = pickle.load(f)

            all_input_sequence_orig   = torch.tensor([f.input_sequence_orig for f in train_features]).float()
            all_input_sequence_masked = torch.tensor([f.input_sequence_masked for f in train_features]).float()
            all_input_mask            = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids           = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_el_was_masked         = torch.tensor([f.el_was_masked for f in train_features]).float()

            whole_sequence_labels = {k: [] for k in whole_sequence_tasks}
            whole_sequence_labels_present = {k: [] for k in whole_sequence_tasks}
            for f in train_features:
                assert set(f.whole_sequence_labels.keys()).issubset(set(whole_sequence_tasks)), "Tasks mismatch!"

                for k in whole_sequence_tasks:
                    if k in f.whole_sequence_labels:
                        whole_sequence_labels[k].append(f.whole_sequence_labels[k].value)
                        whole_sequence_labels_present[k].append(1)
                    else: 
                        whole_sequence_labels[k].append(np.NaN)
                        whole_sequence_labels_present[k].append(0)

            whole_sequence_labels = {k: torch.tensor(v).long() for k, v in whole_sequence_labels.items()}
            whole_sequence_labels_present = {
                k: torch.tensor(v).float() for k, v in whole_sequence_labels_present.items()
            }

            whole_sequence_labels_tensor = torch.stack(
                [whole_sequence_labels[k] for k in whole_sequence_tasks], dim=1
            )
            whole_sequence_labels_present_tensor = torch.stack(
                [whole_sequence_labels_present[k] for k in whole_sequence_tasks], dim=1
            )

            train_data_chunk = TensorDataset(
                all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
                all_el_was_masked, whole_sequence_labels_tensor, whole_sequence_labels_present_tensor
            )

            print("Saving dataset chunk to %s.chunk_%d" % (dataset_filepath, chunk_num))
            torch.save(train_data_chunk, dataset_chunk_path)
            del train_data_chunk, train_features
            gc.collect()
            chunk_num += 1
        raise NotImplementedError
    else:
        if os.path.isfile(processor_filepath):
            del df

            print("Loading processor output from %s" % processor_filepath)
            with open(processor_filepath, mode='rb') as f: train_examples = pickle.load(f)
<<<<<<< HEAD
        elif len(glob.glob(processor_filepath+'*'))>0:

            # del df
            gc.collect()

            examples_=glob.glob('{}.chunk_*'.format(examples_filepath))
            # print(len(examples_))
            processors_=glob.glob('{}.chunk_*'.format(processor_filepath))
            # print(len(processors_))
            p_orig=len(processors_)
=======
        elif os.path.isfile('%s.chunk_0' % processor_filepath): # TODO: all chunks present?
            # del df
            gc.collect()

            print("Loading distributed processor output from %s" % processor_filepath)
            chunk_num = 0
            while chunk_num >= 0 and chunk_num < 159: #os.path.isfile('%s.chunk_%d' % (processor_filepath, chunk_num)):
                if os.path.isfile('%s.chunk_%d' % (examples_filepath, chunk_num)):
                    chunk_num += 1
                    continue
                elif not os.path.isfile('%s.chunk_%d' % (processor_filepath, chunk_num)):
                    print("Skipping nonexistent chunk %d" % chunk_num)
                    chunk_num += 1
                    continue
>>>>>>> 008f8c2b4ee9db1d9f1f7f17e21bc61bd492ec0b

            processors_=[p for p in processors_ if p.replace("processor", "examples")not in examples_]
            # print(len(processors_))

            print("{} example files created, {} processor chunks, and {} processors remaining".format(len(examples_),p_orig,len(processors_)))

            chunk_nums=[int(p.split("chunk_")[-1]) for p in processors_]

            chunk_nums=list(zip([int(p.split("chunk_")[-1]) for p in processors_], processors_, [p.replace("processor", "examples") for p in processors_]))

            # print(chunk_nums)

            
            if parallel:
                with Pool(processes=8) as pool:

                    pool.map(partial(process_chunk, processor_filepath=processor_filepath,
                                                     examples_filepath=examples_filepath,
                                                     max_seq_length=max_seq_length,
                                                     tqdm=tqdm,
                                                     featurization_kwargs=featurization_kwargs), chunk_nums)
            else:
                for chunk_num in tqdm(sorted(chunk_nums, key=lambda x:x[0])):
                    process_chunk(chunk_num, processor_filepath, examples_filepath, max_seq_length, tqdm, featurization_kwargs)




            # while chunk_num<160:
            #     if not(os.path.isfile('%s.chunk_%d' % (processor_filepath, chunk_num))):
            #         chunk_num += 1
            #         continue
            #     if os.path.isfile('%s.chunk_%d' % (examples_filepath, chunk_num)):
            #         chunk_num += 1
            #         continue

            #     print("Loading '%s.chunk_%d'" % (processor_filepath, chunk_num))
            #     with open('%s.chunk_%d' % (processor_filepath, chunk_num), mode='rb') as f:
            #         examples = pickle.load(f)

            #     train_features = convert_examples_to_features(
            #         examples, max_seq_length, tqdm=tqdm, **featurization_kwargs
            #     )
            #     with open('%s.chunk_%d' % (examples_filepath, chunk_num), mode='wb') as f:
            #         pickle.dump(train_features, f)

            #     del train_features
            #     del examples
            #     gc.collect()

            #     chunk_num += 1
            # gc.collect()
            raise NotImplementedError
        else:
            print("Running processor.")
            train_examples = processor.get_train_examples(df, save_path=processor_filepath)
            del df
            print("Saving processor output to %s" % processor_filepath)
            with open(processor_filepath, mode='wb') as f: pickle.dump(train_examples, f)

        print("Running convert_examples_to_features.")
        train_features = convert_examples_to_features(
            train_examples, max_seq_length, tqdm=tqdm, **featurization_kwargs
        )
        del train_examples
        print("Saving featurized output to %s" % examples_filepath)
        with open(examples_filepath, mode='wb') as f: pickle.dump(train_features, f)

    all_input_sequence_orig   = torch.tensor([f.input_sequence_orig for f in train_features]).float()
    all_input_sequence_masked = torch.tensor([f.input_sequence_masked for f in train_features]).float()
    all_input_mask            = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids           = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_el_was_masked         = torch.tensor([f.el_was_masked for f in train_features]).float()

    whole_sequence_labels = {k: [] for k in whole_sequence_tasks}
    whole_sequence_labels_present = {k: [] for k in whole_sequence_tasks}
    for f in train_features:
        assert set(f.whole_sequence_labels.keys()).issubset(set(whole_sequence_tasks)), "Tasks mismatch!"

        for k in whole_sequence_tasks:
            if k in f.whole_sequence_labels:
                whole_sequence_labels[k].append(f.whole_sequence_labels[k].value)
                whole_sequence_labels_present[k].append(1)
            else: 
                whole_sequence_labels[k].append(np.NaN)
                whole_sequence_labels_present[k].append(0)

    whole_sequence_labels = {k: torch.tensor(v).long() for k, v in whole_sequence_labels.items()}
    whole_sequence_labels_present = {
        k: torch.tensor(v).float() for k, v in whole_sequence_labels_present.items()
    }

    whole_sequence_labels_tensor = torch.stack(
        [whole_sequence_labels[k] for k in whole_sequence_tasks], dim=1
    )
    whole_sequence_labels_present_tensor = torch.stack(
        [whole_sequence_labels_present[k] for k in whole_sequence_tasks], dim=1
    )

    train_data = TensorDataset(
        all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
        all_el_was_masked, whole_sequence_labels_tensor, whole_sequence_labels_present_tensor
    )
    del train_features

    print("Saving dataset to %s" % dataset_filepath)
    torch.save(train_data, dataset_filepath)


def pretrain(
    bert_model_class,
    model_kwargs,
    config,
    df,
    output_dir,
    processor,
    featurization_kwargs        = {},
    gradient_accumulation_steps = 1,
    use_gpu                     = False,
    seed                        = 42,
    lambda_seq_tasks            = 10,
    max_seq_length              = 128,
    train_batch_size            = 32,
    eval_batch_size             = 8,
    learning_rate               = 5e-5,
    num_train_epochs            = 3,
    warmup_proportion           = 0.1,
    tqdm                        = tqdm_notebook,
    save_every                  = 2,
    dataset_dir                 = None,
    exit_after_processor        = False,
):
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu  = 0

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0: torch.cuda.manual_seed_all(seed)

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    whole_sequence_tasks = processor.whole_sequence_tasks

    if dataset_dir is not None and not os.path.isdir(dataset_dir): os.makedirs(dataset_dir)

    dataset_filepath   = os.path.join(dataset_dir, DATASET_FILENAME) if dataset_dir is not None else None
    processor_filepath = os.path.join(dataset_dir, PROCESSOR_FILENAME) if dataset_dir is not None else None
    examples_filepath  = os.path.join(dataset_dir, EXAMPLES_FILENAME) if dataset_dir is not None else None

    #assert dataset_filepath is not None, "Sorry, this code is actually terrible."
    if os.path.isfile(dataset_filepath):
        print("Loading dataset from %s" % dataset_filepath)
        train_data = torch.load(dataset_filepath)
    elif os.path.isfile('%s.chunk_8' % dataset_filepath):
        chunk_num = 0
        dataset_chunks = []
        while os.path.isfile('%s.chunk_%d' % (dataset_filepath, chunk_num)):
            print("Loading dataset chunk '%s.chunk_%d'" % (dataset_filepath, chunk_num))
            dataset_chunks.append(torch.load('%s.chunk_%d' % (dataset_filepath, chunk_num)))
            chunk_num += 1

        train_data = ConcatDataset(dataset_chunks)
        torch.save(train_data, dataset_filepath)
    else:
        print("Building dataset from scratch.")

        if os.path.isfile(examples_filepath):
            # del df

            print("Loading featurized list directly from %s" % examples_filepath)
            with open(examples_filepath, mode='rb') as f: train_features = pickle.load(f)
        elif os.path.isfile('%s.chunk_8' % examples_filepath):
            print("Loading distributed trainine xamples from %s" % examples_filepath)
            chunk_num = 0
            while os.path.isfile('%s.chunk_%d' % (examples_filepath, chunk_num)):
                dataset_chunk_path = '%s.chunk_%d' % (dataset_filepath, chunk_num)
                if os.path.isfile(dataset_chunk_path):
                    chunk_num += 1
                    continue

                print("Loading '%s.chunk_%d'" % (examples_filepath, chunk_num))
                with open('%s.chunk_%d' % (examples_filepath, chunk_num), mode='rb') as f:
                    train_features = pickle.load(f)

                all_input_sequence_orig   = torch.tensor([f.input_sequence_orig for f in train_features]).float()
                all_input_sequence_masked = torch.tensor([f.input_sequence_masked for f in train_features]).float()
                all_input_mask            = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
                all_segment_ids           = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
                all_el_was_masked         = torch.tensor([f.el_was_masked for f in train_features]).float()

                whole_sequence_labels = {k: [] for k in whole_sequence_tasks}
                whole_sequence_labels_present = {k: [] for k in whole_sequence_tasks}
                for f in train_features:
                    assert set(f.whole_sequence_labels.keys()).issubset(set(whole_sequence_tasks)), "Tasks mismatch!"

                    for k in whole_sequence_tasks:
                        if k in f.whole_sequence_labels:
                            whole_sequence_labels[k].append(f.whole_sequence_labels[k].value)
                            whole_sequence_labels_present[k].append(1)
                        else: 
                            whole_sequence_labels[k].append(np.NaN)
                            whole_sequence_labels_present[k].append(0)

                whole_sequence_labels = {k: torch.tensor(v).long() for k, v in whole_sequence_labels.items()}
                whole_sequence_labels_present = {
                    k: torch.tensor(v).float() for k, v in whole_sequence_labels_present.items()
                }

                whole_sequence_labels_tensor = torch.stack(
                    [whole_sequence_labels[k] for k in whole_sequence_tasks], dim=1
                )
                whole_sequence_labels_present_tensor = torch.stack(
                    [whole_sequence_labels_present[k] for k in whole_sequence_tasks], dim=1
                )

                train_data_chunk = TensorDataset(
                    all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
                    all_el_was_masked, whole_sequence_labels_tensor, whole_sequence_labels_present_tensor
                )

                print("Saving dataset chunk to %s.chunk_%d" % (dataset_filepath, chunk_num))
                torch.save(train_data_chunk, dataset_chunk_path)
                del train_data_chunk, train_features
                gc.collect()
                chunk_num += 1
            raise NotImplementedError
        else:
            if os.path.isfile(processor_filepath):
                del df

                print("Loading processor output from %s" % processor_filepath)
                with open(processor_filepath, mode='rb') as f: train_examples = pickle.load(f)
            elif os.path.isfile('%s.chunk_215' % processor_filepath): # TODO: all chunks present?
                del df
                gc.collect()

                print("Loading distributed processor output from %s" % processor_filepath)
                chunk_num = 0
                while os.path.isfile('%s.chunk_%d' % (processor_filepath, chunk_num)):
                    if os.path.isfile('%s.chunk_%d' % (examples_filepath, chunk_num)):
                        chunk_num += 1
                        continue

                    print("Loading '%s.chunk_%d'" % (processor_filepath, chunk_num))
                    with open('%s.chunk_%d' % (processor_filepath, chunk_num), mode='rb') as f:
                        examples = pickle.load(f)

                    train_features = convert_examples_to_features(
                        examples, max_seq_length, tqdm=tqdm, **featurization_kwargs
                    )
                    with open('%s.chunk_%d' % (examples_filepath, chunk_num), mode='wb') as f:
                        pickle.dump(train_features, f)

                    del train_features
                    del examples
                    gc.collect()

                    chunk_num += 1
                gc.collect()
                raise NotImplementedError
            else:
                print("Running processor.")
                train_examples = processor.get_train_examples(df, save_path=processor_filepath)
                raise NotImplementedError
                #print("Saving processor output to %s" % processor_filepath)
                #with open(processor_filepath, mode='wb') as f: pickle.dump(train_examples, f)

            print("Running convert_examples_to_features.")
            train_features = convert_examples_to_features(
                train_examples, max_seq_length, tqdm=tqdm, **featurization_kwargs
            )
            print("Saving featurized output to %s" % examples_filepath)
            with open(examples_filepath, mode='wb') as f: pickle.dump(train_features, f)

        if exit_after_processor:
            print("Exiting.")
            exit()

        all_input_sequence_orig   = torch.tensor([f.input_sequence_orig for f in train_features]).float()
        all_input_sequence_masked = torch.tensor([f.input_sequence_masked for f in train_features]).float()
        all_input_mask            = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids           = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_el_was_masked         = torch.tensor([f.el_was_masked for f in train_features]).float()

        whole_sequence_labels = {k: [] for k in whole_sequence_tasks}
        whole_sequence_labels_present = {k: [] for k in whole_sequence_tasks}
        for f in train_features:
            assert set(f.whole_sequence_labels.keys()).issubset(set(whole_sequence_tasks)), "Tasks mismatch!"

            for k in whole_sequence_tasks:
                if k in f.whole_sequence_labels:
                    whole_sequence_labels[k].append(f.whole_sequence_labels[k].value)
                    whole_sequence_labels_present[k].append(1)
                else: 
                    whole_sequence_labels[k].append(np.NaN)
                    whole_sequence_labels_present[k].append(0)

        whole_sequence_labels = {k: torch.tensor(v).long() for k, v in whole_sequence_labels.items()}
        whole_sequence_labels_present = {
            k: torch.tensor(v).float() for k, v in whole_sequence_labels_present.items()
        }

        whole_sequence_labels_tensor = torch.stack(
            [whole_sequence_labels[k] for k in whole_sequence_tasks], dim=1
        )
        whole_sequence_labels_present_tensor = torch.stack(
            [whole_sequence_labels_present[k] for k in whole_sequence_tasks], dim=1
        )

        train_data = TensorDataset(
            all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
            all_el_was_masked, whole_sequence_labels_tensor, whole_sequence_labels_present_tensor
        )

        if dataset_filepath is not None:
            print("Saving dataset to %s" % dataset_filepath)
            torch.save(train_data, dataset_filepath)

    print("Finally done with dataset creation.... Running model now over %d gpus." % n_gpu)

    # Prepare model
    model_kwargs['lambda_seq_tasks'] = lambda_seq_tasks

    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    model = bert_model_class( config, **model_kwargs)
    if os.path.isfile(output_model_file):
        print("Reloading from prior model")
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)
    if n_gpu > 1: model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = math.ceil(
        len(train_data) / train_batch_size / gradient_accumulation_steps
    ) * num_train_epochs
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    # TODO(mmd): Add SWA: https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=learning_rate,
        warmup=warmup_proportion,
        t_total=num_train_optimization_steps
    )


    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    model.train()
    bar_epoch = tqdm(range(int(num_train_epochs)), desc="Epoch: N/A") 
    for epoch in bar_epoch:
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        bar_iter = tqdm(train_dataloader, desc="Iteration: N/A")
        for step, batch in enumerate(bar_iter):
            batch = tuple(t.to(device) for t in batch)
            (
                all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
                all_el_was_masked, whole_sequence_labels_tensor, whole_sequence_labels_present_tensor
            ) = batch

            whole_sequence_labels = {
                t: whole_sequence_labels_tensor[:, i] for i, t in enumerate(whole_sequence_tasks)
            }
            whole_sequence_labels_present = {
                t: whole_sequence_labels_present_tensor[:, i] for i, t in enumerate(whole_sequence_tasks)
            }

            with detect_anomaly():
                _, loss, _, _ = model(
                    all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
                    all_el_was_masked, whole_sequence_labels, whole_sequence_labels_present
                )

                if n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1: loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += all_input_sequence_orig.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            bar_iter.set_description('Iteration: %.2f' % (tr_loss / (step + 1)))
        bar_epoch.set_description("Epoch: %.2f" % (tr_loss / (step + 1)))

        if epoch % save_every == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(output_config_file, 'w') as f: f.write(model_to_save.config.to_json_string())

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)
    with open(output_config_file, 'w') as f: f.write(model_to_save.config.to_json_string())

    # Load a trained model and config that you have fine-tuned
    config = ContinuousBertConfig.from_json_file(output_config_file)
    model = bert_model_class(
        config,
        **model_kwargs,
    )
    model.load_state_dict(torch.load(output_model_file))

def load_and_yield_embeddings(
    bert_model_class,
    model_kwargs,
    dataset_filepath,
    model_dir,
    use_gpu          = False,
    seed             = 42,
    batch_size       = 32,
    lambda_seq_tasks = 10,
    tqdm             = tqdm_notebook,
):
    model_filepath  = os.path.join(model_dir, WEIGHTS_NAME)
    config_filepath = os.path.join(model_dir, CONFIG_NAME)
    assert os.path.isfile(dataset_filepath), "To load embeddings, you need to pre-write out the dataset."
    assert os.path.isfile(config_filepath), "Must have a saved config."
    assert os.path.isfile(model_filepath), "Must have a pretrained model."

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    whole_sequence_tasks = {SEQUENCES_ORDERED: len(LABEL_ENUMS[SEQUENCES_ORDERED])}

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")
        n_gpu  = 0

    print("Loading dataset from %s" % dataset_filepath)
    data = torch.load(dataset_filepath)

    # Prepare model
    model_kwargs['lambda_seq_tasks'] = lambda_seq_tasks

    print("Reloading model")
    model = bert_model_class(ContinuousBertConfig.from_json_file(config_filepath), **model_kwargs)
    model.load_state_dict(torch.load(model_filepath))

    model.to(device)
    if n_gpu > 1: model = torch.nn.DataParallel(model)

    model.eval()

    sampler    = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    all_outputs, overall_loss = [], 0

    bar_iter = tqdm(dataloader, desc="Iteration: N/A")
    for step, batch in enumerate(bar_iter):
        batch = tuple(t.to(device) for t in batch)
        (
            all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
            all_el_was_masked, whole_sequence_labels_tensor, whole_sequence_labels_present_tensor
        ) = batch

        whole_sequence_labels = {
            t: whole_sequence_labels_tensor[:, i] for i, t in enumerate(whole_sequence_tasks)
        }
        whole_sequence_labels_present = {
            t: whole_sequence_labels_present_tensor[:, i] for i, t in enumerate(whole_sequence_tasks)
        }

        pooled_output, loss, _, _ = model(
            all_input_sequence_orig, all_input_sequence_masked, all_input_mask, all_segment_ids,
            all_el_was_masked, whole_sequence_labels, whole_sequence_labels_present
        )

        if n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu.

        all_outputs.extend(pooled_output.cpu().data.numpy())

        overall_loss += loss.item()

        bar_iter.set_description('Iteration: %.2f' % (overall_loss / (step + 1)))

    all_outputs = np.vstack(all_outputs)
    return all_outputs
