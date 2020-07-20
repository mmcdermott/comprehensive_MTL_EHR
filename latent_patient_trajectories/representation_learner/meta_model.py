"""
SelfAttentionEncoder.py
"""

import json, os, torch, torch.optim, torch.nn as nn, torch.nn.init as init
from collections import OrderedDict
from pytorch_pretrained_bert.modeling import BertModel, BertConfig

# TODO: check imports
from ..utils import *
from ..constants import *
from ..data_utils import *
from ..representation_learner.fts_decoder import *
from ..representation_learner.adapted_model import *
from ..BERT.model import *
from ..BERT.constants import *

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

def strip_module_and_load(recipient, loaded_state_dict):
    try:
        old_state_dict = recipient.state_dict()
        recipient.load_state_dict(loaded_state_dict)
    except RuntimeError as e:
        if 'Missing key(s) in state_dict: "fc_stack.0.weight", "fc_stack.0.bias"' in str(e):
            print("Fixing old broken GRU...")
            new_state_dict = OrderedDict({
                k: v for k, v in loaded_state_dict.items() if k not in ('fc.weight', 'fc.bias')
            })
            new_state_dict['fc_stack.0.weight'] = loaded_state_dict['fc.weight']
            new_state_dict['fc_stack.0.bias'] = loaded_state_dict['fc.bias']


            k = "task_losses.tasks_binary_multilabel.pos_weight"
            if k in old_state_dict and k not in new_state_dict:
                print("Fixing missing pos_weight key...")
                print(old_state_dict[k])
                new_state_dict[k] = old_state_dict[k]
        elif 'Missing key(s) in state_dict: "task_losses.tasks_binary_multilabel.pos_weight"' in str(e):
            print("Fixing missing pos_weight key...")
            new_state_dict = OrderedDict(loaded_state_dict)
            print(old_state_dict[
                "task_losses.tasks_binary_multilabel.pos_weight"
            ])

            new_state_dict["task_losses.tasks_binary_multilabel.pos_weight"] = old_state_dict[
                "task_losses.tasks_binary_multilabel.pos_weight"
            ]
        elif 'Missing key(s) in state_dict: "task_losses.tasks_binary_multilabel.BCE_LL.pos_weight' in str(e):
            # error introduced in bug fix for BCE
            print("Fixing missing BCE.LL pos_weight key...")
            new_state_dict = OrderedDict(loaded_state_dict)
            print(old_state_dict[
                "task_losses.tasks_binary_multilabel.BCE_LL.pos_weight"
            ])
            print("task_losses.tasks_binary_multilabel.BCE_LL.pos_weight" in old_state_dict.keys())

            new_state_dict["task_losses.tasks_binary_multilabel.BCE_LL.pos_weight"] = old_state_dict[
                "task_losses.tasks_binary_multilabel.BCE_LL.pos_weight"
            ]
        else:
            prefix = "module."
            new_state_dict = OrderedDict({
                (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in loaded_state_dict.items()
            })

        recipient.load_state_dict(new_state_dict)

class MetaModel():
    def __init__(self, args, sample_datum, class_names=None, use_cuda=torch.cuda.is_available()):
        print("curr path:", args.run_dir)
        assert os.path.isdir(args.run_dir)
        if class_names is None: class_names = {}

        self.run_dir = args.run_dir

        device = torch.device("cuda" if use_cuda else "cpu")
        n_gpu = 0 if not use_cuda else torch.cuda.device_count()
        self.device = device
        self.n_gpu = n_gpu

        self.class_names = class_names
        task_weights, task_class_weights = None, {'tasks_binary_multilabel': torch.ones(len(self.class_names['tasks_binary_multilabel'])).to(self.device)}
        if args.task_weights_filepath:
            assert os.path.isfile(args.task_weights_filepath), "Task weights file does not exist!"
            assert args.regression_task_weight == 1, "Can't set both regression task weight and file!"
            assert not args.ablate, "Can't both use a file and an ablation code."

            with open(args.task_weights_filepath, mode='r') as f: task_weights = json.loads(f.read())
        elif args.regression_task_weight != 1:
            assert not os.path.isfile(args.task_weights_filepath), "Can't use both a file and a reg. weight!"
            assert not args.ablate, "Can't both use a reg. weight and an ablation code!"

            task_weights = {t: 1 for t in ALL_TASKS}
            task_weights['next_timepoint'] = args.regression_task_weight
        elif args.ablate:
            assert not os.path.isfile(args.task_weights_filepath), "Can't use both an ablation and a file!"
            assert args.regression_task_weight == 1, "Can't set both regression task weight and ablation!"

            print("Ablating!")

            task_weights, task_class_weights = self.ablate(args.ablate, post_init=False)
        else:
            task_weights, task_class_weights = self.ablate(None, post_init=False)

        self.add_cls_analog = False
        if args.do_add_cls_analog:
            assert args.modeltype.lower() not in ('cnn', 'gru', 'linear'), "CLS analog only works w/ BERT"
            self.add_cls_analog = True
            self.cls_embed = nn.Parameter(data=torch.randn(1, 1, args.hidden_size), requires_grad=True)
        else: self.cls_embed = None

        # No batch size as this is just accessed via dataset[#].
        ts_feat_dim, statics_feat_dim = sample_datum['ts'].shape[1], sample_datum['statics'].shape[0]
        pred_dim = sample_datum['next_timepoint'].shape

        config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": args.hidden_dropout_prob,
            "pred_dim": pred_dim,
            "hidden_size": args.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": args.intermediate_size,
            "max_position_embeddings": args.max_seq_len + 1 if self.add_cls_analog else args.max_seq_len,
            "num_attention_heads": args.num_attention_heads,
            "num_hidden_layers": args.num_hidden_layers,
            "type_vocab_size": 2,
            "vocab_size": None # TODO(mmd): Omit this from config...
        }
        bert_config = BertConfig.from_dict(config)
        bert_config_filepath = os.path.join(args.run_dir, CONFIG_FILENAME)

        if not os.path.isfile(bert_config_filepath) or args.do_overwrite:
            bert_config.to_json_file(os.path.join(args.run_dir, 'bert_config.json'))

        # default arg is self attention timeseries

        # alternative is CNN
        if args.modeltype.lower() == 'cnn':
            model = CNN(
                bert_config, data_shape=[args.max_seq_len, args.hidden_size],
                use_cuda=torch.cuda.is_available(), conv_layers = list(args.num_filters),
                kernel_sizes=list(args.kernel_sizes), fc_layer_sizes = list(args.fc_layer_sizes),
                pooling_method = args.pooling_method, pooling_kernel_size = args.pooling_kernel_size,
                pooling_stride = args.pooling_stride, conv_layers_per_pool = args.conv_layers_per_pool,
                task_weights = task_weights, task_class_weights=task_class_weights,
            )
        elif args.modeltype.lower() == 'gru':
            model = GRUModel(
                bert_config, data_shape=[args.max_seq_len, args.hidden_size], use_cuda=torch.cuda.is_available(),
                hidden_dim=args.gru_hidden_layer_size, num_layers=args.gru_num_hidden,
                bidirectional=args.do_bidirectional, task_weights=task_weights,
                pooling_method=args.gru_pooling_method, task_class_weights=task_class_weights,
            )
        elif args.modeltype.lower() =='linear':
            model = LinearModel(
                bert_config, data_shape=[args.max_seq_len, args.hidden_size],
                use_cuda=torch.cuda.is_available(), task_weights=task_weights,
                task_class_weights=task_class_weights,
            )
        else:
            model = SelfAttentionTimeseries(
                bert_config, use_cuda=torch.cuda.is_available(), task_weights=task_weights,
                task_class_weights=task_class_weights,
            )

        # TODO(mmd): Need to also load ts_projector.
        ts_projector = nn.Linear(ts_feat_dim, bert_config.hidden_size)
        statics_projector = nn.Linear(statics_feat_dim, bert_config.hidden_size)

        model.apply(weight_init)

        for m in (model, ts_projector, statics_projector, self.cls_embed):
            if m is None: continue
            m.to(device)
            if n_gpu > 1: m = torch.nn.DataParallel(m).cuda()

        parameters = (
            list(model.parameters()) + list(ts_projector.parameters()) + list(statics_projector.parameters())
        )
        if self.add_cls_analog: parameters += [self.cls_embed]

        if args.notes == 'integrate_note_bert':
            # initialize pretrained clinical note bert model
            cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_-1')

            model_location = BERT_MODEL_LOCATION

            note_embedding_model = BertModel.from_pretrained(
                model_location,
                cache_dir=cache_dir,
            )
            notes_projector = nn.Linear(768, bert_config.hidden_size)

            note_embedding_model.to(device)
            notes_projector.to(device)

            if n_gpu > 1:
                note_embedding_model = torch.nn.DataParallel(
                    note_embedding_model, device_ids=list(range(torch.cuda.device_count()))
                ).cuda()
                notes_projector = torch.nn.DataParallel(
                    notes_projector, device_ids=list(range(torch.cuda.device_count()))
                ).cuda()

            if args.do_train_note_bert:
                parameters = [
                    {"params": parameters, "lr": args.learning_rate},
                    {
                        "params": note_embedding_model.parameters(),
                        "lr": args.learning_rate / args.note_bert_lr_reduce
                    },
                    {
                        "params": notes_projector.parameters(),
                        "lr": args.learning_rate / args.note_bert_lr_reduce
                    },
                ]
        else:
            note_embedding_model = None
            notes_projector = None

        self.bert_config = bert_config
        self.model = model
        self.ts_projector = ts_projector
        self.statics_projector = statics_projector
        self.notes_projector = notes_projector
        self.note_embedding_model = note_embedding_model
        self.parameters = parameters

        self.trainable_models = [self.model, self.ts_projector, self.statics_projector]
        if args.do_train_note_bert:
            self.trainable_models.extend([self.notes_projector, self.note_embedding_model])

        self.n_gpu = n_gpu
        self.device = device
        self.notes = args.notes

        self.run_dir = args.run_dir
        self.save_name = args.model_file_template.format(**args.to_dict())

    def ablate(self, ablate, post_init=False):
        task_class_weights = {
            'tasks_binary_multilabel': torch.ones(
                len(self.class_names['tasks_binary_multilabel'])
            ).to(self.device)
        } # todo how big?
        if ablate is None:
            with open(os.path.join(self.run_dir, "joint_weights.pkl"), "wb") as f:
                pickle.dump({'task_weights': {}, 'task_class_weights': task_class_weights}, f)
            return None, task_class_weights

        task_weights = {t: 1 for t in ALL_TASKS}

        if isinstance(ablate, str): ablate=[ablate]
        # 'rolling_fts', 'disch_24h', 'disch_48h', 'Final Acuity Outcome', 'tasks_binary_multilabel', 'next_timepoint', 'next_timepoint_was_measured'
        for ablation in ablate:
            if ablation in ABLATION_GROUPS.keys():
                #ablate entire ablation group
                # need a list of each variable for this assertion
                # for tasks that are NOT binary multilabel
                for t in ABLATION_GROUPS[ablation]:
                    if t in task_weights.keys():
                        task_weights[t] = 0
                    elif t in self.class_names['tasks_binary_multilabel']:
                        # self.task_class_weights['tasks_binary_multilabel'] must be a tensor with the 1 element for each binary multilabel class
                        task_class_weights['tasks_binary_multilabel'][
                            self.class_names['tasks_binary_multilabel'].index(t)
                        ]=0
                        #
            elif ablation in ALL_TASKS:
                # ablate individual task
                task_weights[ablation]=0
            else:
                raise Exception(
                    f'Error trying to ablate {ablation}. It is not found in the eligible ablation groups '
                     'or any of the tasks.'
                )


        if post_init:
            self.model.task_weights = task_weights
            self.model.task_losses = get_task_losses(task_class_weights)
            self.model.task_class_weights = task_class_weights

#         print("vals", self.class_names['tasks_binary_multilabel'])
        with open(os.path.join(self.run_dir, "joint_weights.pkl"), "wb") as f:
                pickle.dump({'task_weights': task_weights, 'task_class_weights': task_class_weights}, f)

        return task_weights, task_class_weights

    def parameters(self): return self.parameters

    def freeze_representation(self):
        for module in (self.ts_projector, self.notes_projector, self.statics_projector):
            if module is None: continue
            for p in module.parameters(): p.requires_grad = False
        self.model.freeze_representation()

    def unfreeze_representation(self):
        for module in (self.ts_projector, self.notes_projector, self.statics_projector):
            if module is None: continue
            for p in module.parameters(): p.requires_grad = True
        self.model.unfreeze_representation()

    def train(self):
        for m in self.trainable_models:
            if m is not None: m.train()

    def eval(self):
        for m in self.trainable_models:
            if m is not None: m.eval()

    def state_dict(self):
        state_dict = {
            'model': self.model.state_dict(),
            'ts_projector': self.ts_projector.state_dict(),
            'statics_projector': self.statics_projector.state_dict(),
        }
        if self.add_cls_analog:
            state_dict['cls_token_embed'] = self.cls_embed
        if self.note_embedding_model is not None:
            state_dict['note_embedding_model'] = self.note_embedding_model.state_dict()
        if self.notes_projector is not None:
            state_dict['notes_projector'] = self.notes_projector.state_dict()
        return state_dict

    def save(self, epoch=0):
        to_save = {'epoch': epoch, **self.state_dict()}

        save_path = os.path.join(self.run_dir, '%s.epoch-%d' % (self.save_name, epoch))
        torch.save(to_save, save_path)

    def load(self, epoch='latest'):
        if epoch == 'latest':
            files = os.listdir(self.run_dir)

            all_epochs = []
            prefix = '%s.epoch-' % self.save_name
            for f in files:
                if not f.startswith(prefix): continue
                all_epochs.append(int(f[len(prefix):]))

            if not all_epochs: return False, None
            epoch = max(all_epochs)

        assert type(epoch) is int and epoch >= 0, "epoch must be 'latest' or an epoch #"

        load_path = os.path.join(self.run_dir, '%s.epoch-%d' % (self.save_name, epoch))
        if not os.path.isfile(load_path): return False, None

        to_load = torch.load(load_path, map_location=self.device)
        assert to_load['epoch'] == epoch, "Something is wrong... %d v. %d" % (to_load['epoch'], epoch)

        strip_module_and_load(self.model, to_load['model'])
        strip_module_and_load(self.ts_projector, to_load['ts_projector'])
        strip_module_and_load(self.statics_projector, to_load['statics_projector'])

        if 'note_embedding_model' in to_load:
            assert self.note_embedding_model is not None, "Load/non-none mismatch."
            strip_module_and_load(self.note_embedding_model, to_load['note_embedding_model'])
        if 'notes_projector' in to_load:
            assert self.notes_projector is not None, "Load/non-none mismatch."
            strip_module_and_load(self.notes_projector, to_load['notes_projector'])

        return True, epoch

    #def to_appropriate_device(self, data, for_notes=False, single_batch=False):
    #    if type(data) is dict:
    #        return {k: self.to_appropriate_device(v, for_notes, single_batch) for k, v in data.items()}
    #    elif type(data) in (list, tuple):
    #        return [self.to_appropriate_device(v, for_notes, single_batch) for v in data]

    #    # Otherwise assume its a tensor...
    #    data = data.squeeze()
    #    if single_batch: data = data.unsqueeze(0)

    #    if self.notes == 'integrate_note_bert':
    #        if for_notes:
    #        else: device = final_device

    #    return data.to(device).float()

    def forward(self, batch):
        for k, value in batch.items(): batch[k]=value.float().to(self.device)

        single_batch = (batch['ts'].shape[0] == 1)

        statics, ts = batch['statics'], batch['ts']

        input_sequence = self.ts_projector(ts)

        batch_size, seq_len, ts_feat_dim = list(ts.shape)
        batch_size, statics_feat_dim = list(statics.shape)


        statics = self.statics_projector(statics)
        input_sequence += statics.unsqueeze(1).expand([batch_size, seq_len, self.bert_config.hidden_size])

        if self.add_cls_analog:
            cls_embed_exp = self.cls_embed.expand([batch_size, 1, self.bert_config.hidden_size])
            mask_addition = torch.ones(batch_size, 1, 1)
            ts_mask = batch['ts_mask']

            a = []
            for m in (cls_embed_exp, mask_addition, ts_mask, input_sequence):
                m = m.to(self.device)
                if self.n_gpu > 1: m = torch.nn.DataParallel(m).cuda()
                a.append(m)
            cls_embed_exp, mask_addition, ts_mask, input_sequence = a

            input_sequence = torch.cat([cls_embed_exp, input_sequence], dim=1)
            batch['ts_mask'] = torch.cat((mask_addition, ts_mask), dim=1)

        batch['input_sequence'] = input_sequence


        if self.notes == 'integrate_note_bert':
            note_ids = batch['note_ids']
            note_masks = batch['note_masks']
            note_segment_ids = batch['note_segment_ids']
            note_hours_idx = batch['note_hours_idx']
            note_hours_num = batch['note_hours_num']

            batch_size, max_seq_length, max_note_length = note_ids.shape

            # collect the batch size and number of hours into one axis then only select where there is
            # a note.
            # we need a new squish method for our hour index
            # note_hours_idx looks like [[0,5,7,8, -1,-1,-1,-1], [0,7,9,10,-1, -1,-1...]]
            to_add = (
                torch.arange(0, batch_size, device=self.device)*max_seq_length
            ).unsqueeze(1).expand(-1, max_seq_length)
            to_add=torch.where(note_hours_idx.long()<0, torch.tensor(-1, device=self.device), to_add)
            good_index = (to_add  + note_hours_idx.long()).view(-1)
            good_index = good_index[good_index>=0]

            squished_note_ids = note_ids.view(-1, max_note_length)[good_index, :]
            squished_note_masks = note_masks.view(-1, max_note_length)[good_index, :]
            squished_note_segment_ids = note_segment_ids.view(-1, max_note_length)[good_index, :]

            # append zeros to these otherwise if they are too small for a gpu
            # assume 4 things fit on a batch
            models_that_will_fit = 4*(torch.cuda.device_count())
            remainder = models_that_will_fit - (squished_note_ids.shape[0] % models_that_will_fit)
            #if remainder == 0: remainder = models_that_will_fit

            pad = lambda t: torch.cat(
                [t, torch.zeros(remainder, max_note_length, device=self.device)], dim=0
            )
            squished_note_ids = pad(squished_note_ids)
            squished_note_masks = pad(squished_note_masks)
            squished_note_segment_ids = pad(squished_note_segment_ids)

            num_splits = squished_note_ids.shape[0] // models_that_will_fit

            # The note embedding model is put on cuda1:n so we should put the data for it on the same
            # device (by default cuda sends it to cuda 0)

            if squished_note_ids.shape[0] ==models_that_will_fit:
                # we can go right into the forward pass
                _, squished_note_embeddings = self.note_embedding_model(
                    squished_note_ids.long().to(self.device), squished_note_masks.long().to(self.device),
                    squished_note_segment_ids.long().to(self.device)
                )
            else:
                outputs = []
                for i in range(squished_note_ids.shape[0]//models_that_will_fit):
                    from_=i*models_that_will_fit
                    to_=(i+1)*models_that_will_fit
                    _, output = self.note_embedding_model(squished_note_ids[from_:to_, :].long().to(self.device),
                                                    squished_note_masks[from_:to_, :].long().to(self.device),
                                                    squished_note_segment_ids[from_:to_, :].long().to(self.device))
                    outputs.append(output)
                squished_note_embeddings = torch.cat(outputs, dim=0).to(self.device)
                del outputs

            input_embeddings = torch.zeros(
                batch_size * max_seq_length, squished_note_embeddings.shape[-1]
            ).to(self.device)

            from_idx = 0
            input_embeddings[good_index, :] = squished_note_embeddings[:-remainder]

            # apply the projection
            input_embeddings = self.notes_projector(
                input_embeddings.to(self.device)
            ).view(batch_size, max_seq_length, -1)
            # input_embeddings = torch.cat(
            #     (torch.zeros(batch_size, 1, self.bert_config.hidden_size).to(self.device),
            #      input_embeddings), dim=1
            # )

            batch['input_sequence'] += input_embeddings
            batch = {k: fit_on_device(v, single_batch, self.device) for k, v in batch.items()}


        hidden_states, pooled_output, all_outputs, total_loss = self.model(batch)
        if self.n_gpu > 1: total_loss = total_loss.sum() # Across all gpus...

        return hidden_states, pooled_output, all_outputs, total_loss

