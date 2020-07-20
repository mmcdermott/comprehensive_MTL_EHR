"""
run_model.py
"""

import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

import json, os, pickle
from tqdm import tqdm

# TODO: check these imports.
from ..utils import *
from ..constants import *
from ..data_utils import *
from ..representation_learner.fts_decoder import *
from ..representation_learner.evaluator import *
from ..representation_learner.meta_model import *
from ..BERT.model import *
from ..BERT.constants import *

def train_meta_model(
    meta_model, train_dataloader, args, reloaded=False, epoch=0,
    tuning_dataloader=None, train_embedding_after=-1, tqdm=tqdm
):
    all_train_perfs, all_dev_perfs = [], []
    optimizer = torch.optim.Adam(
        meta_model.parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.learning_rate_step, args.learning_rate_decay)
    
    early_stop_count=0
    prev_err=10e9

    epoch_rng = range(epoch+1 if reloaded else 0, args.epochs)
    if tqdm is not None: epoch_rng = tqdm(epoch_rng, desc='Epoch: N/A', leave=False)

    for epoch in epoch_rng:
        scheduler.step() # This goes before any of the train/validation stuff b/c torch version is 1.0.1
        
        if train_embedding_after >= epoch:
            # to ensure it is unfrozen after reloading
            meta_model.unfreeze_representation()

        meta_model.train()

        dataloader_rng = train_dataloader
        if tqdm is not None:
            dataloader_rng = tqdm(
                dataloader_rng, desc='Batch: N/A', total=len(train_dataloader), leave=False)

        optimizer.zero_grad()

        for i, batch in enumerate(dataloader_rng):
            if args.do_detect_anomaly: set_detect_anomaly(True)

            hidden_states, pooled_output, all_outputs, total_loss = meta_model.forward(batch)

            total_loss.backward()
            if i % args.batches_per_gradient == 0:
                optimizer.step()
                optimizer.zero_grad()
            if args.do_detect_anomaly: set_detect_anomaly(False)

            if tqdm is not None: dataloader_rng.set_description('Batch: %.2e' % total_loss)

        if tqdm is None: print("Epoch %d: %.2f" % (epoch, total_loss.item()))
        elif (tqdm is not None) and (tuning_dataloader is not None): pass
        else: epoch_rng.set_description("Epoch %d: %.2f" % (epoch, total_loss.item()))

        if epoch % args.train_save_every == 0:
            if tuning_dataloader is None:
                meta_model.save(epoch)
            else:
                # do eval to see if this is the best score
                dataloader_rng = tuning_dataloader
                if tqdm is not None:
                    dataloader_rng = tqdm(
                        dataloader_rng, desc='Batch: N/A', total=len(tuning_dataloader), leave=False)
                meta_model.eval()
                tuning_losses=[]
                for i, batch in enumerate(dataloader_rng):
                    hidden_states, pooled_output, all_outputs, total_loss = meta_model.forward(batch)
                    tuning_losses.append(total_loss.cpu().data.numpy().ravel())
                meta_model.train()
                total_err = np.mean(np.concatenate(tuning_losses))
                
                if total_err < prev_err:
                    # this is the best model
                    meta_model.save(epoch)
#                     best_meta_model=meta_model.copy().cpu()
                    prev_err=total_err
                    early_stop_count=0
                    if tqdm is None: print("Epoch %d: %.2f" % (epoch, total_loss.item()))
                    else: epoch_rng.set_description("Epoch %d: %.2f" % (epoch, total_loss.item()))
                else:
                    early_stop_count+=1
                    if early_stop_count==10:
                        print(f"Early stopping at epoch {epoch}. Best model at epoch {epoch} with a loss of {prev_err}")
                        break
                
            
    if tuning_dataloader is None:
        meta_model.save(epoch)
        return meta_model
    else: return best_meta_model

def run_model(args, datasets, train_dataloader, tqdm=None, meta_model=None, tuning_dataloader=None):
    if meta_model is None: meta_model = MetaModel(
        args, datasets['train'][0],
        class_names = {'tasks_binary_multilabel': datasets['train'].get_binary_multilabel_keys()}
    )
    reloaded, epoch = meta_model.load()
    if reloaded: print("Resuming from epoch %d" % (epoch+1))

    if args.do_train:
        print('training')
        train_meta_model(meta_model, train_dataloader, args, reloaded, epoch, tuning_dataloader)

    #for n, do, dataloader in (
    #    ('train', args.record_train_perf, train_dataloader), ('dev', args.val, dev_dataloader),
    #    ('test', args.test, test_dataloader)
    #):
    #    if not do: continue
    #    perf_dict = eval_loop(
    #        model, device, dataloader, tqdm=tqdm, run_dir=args.run_dir, n_gpu=n_gpu, ablations=args.ablate,
    #        integrate_note_bert=args.integrate_note_bert, note_embedding_model=note_embedding_model, notes_projector=notes_projector,
    #        batch_size=args.batch_size, only_notes=args.only_notes, train_note_bert=args.train_note_bert,
    #        using_notes_embeddings=using_notes_embeddings
    #    )
    #    with open(os.path.join(args.run_dir, '{}.eval'.format(n)), mode='wb') as f:
    #        pickle.dump(perf_dict, f)
    return meta_model

def setup_for_run(args):
    # Make run_dir if it doesn't exist.
    if not os.path.exists(args.run_dir): os.mkdir(os.path.abspath(args.run_dir))
    elif not args.do_overwrite:
        raise ValueError("Save dir %s exists and overwrite is not enabled!" % args.run_dir)
    args.to_json_file(os.path.join(args.run_dir, ARGS_FILENAME))

    if not args.dataset_dir:
        args.dataset_dir = os.path.join(ROTATIONS_DIR, args.notes, str(args.rotation))

    datasets = {}
    for split, do in (
        ('train', args.do_train or args.do_eval_train), ('tuning', args.do_eval_tuning),
        ('test', args.do_eval_test)
    ):
        if not do:
            datasets[split] = None
            continue

        load_start = time.time()
        with open(os.path.join(args.dataset_dir, '%s_dataset.pkl' % split), mode='rb') as f:
            datasets[split] = pickle.load(f)
        print('loading %s data from disk took %.2f minutes' % (split, (time.time() - load_start)/60))

    if not args.do_train: return datasets

    args.to_json_file(os.path.join(args.run_dir, ARGS_FILENAME))
    
    print("args.set_to_eval_mode: ", args.set_to_eval_mode)
    datasets['train'].max_seq_len = args.max_seq_len

    if args.set_to_eval_mode: datasets['train'].set_to_eval_mode(args.set_to_eval_mode)

    sampler = SubsetRandomSampler(list(range(50))) if args.do_test_run else RandomSampler(datasets['train'])
    train_dataloader = DataLoader(
        datasets['train'], sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers
    )

    return datasets, train_dataloader

def main(args, tqdm):
    datasets, train_dataloader = setup_for_run(args)
    
    # added to restrict the data in the dataset
    if hasattr(args, 'frac_data'):
        if args.frac_data != 1:
            import random        
            # get index of train_dataset

            if args.frac_data_seed !=0:
                random.seed(args.frac_data_seed)

            orig_len=len(datasets['train'])
            subjects_hours = list(zip(datasets['train'].orig_subjects, datasets['train'].orig_max_hours))
            assert len(set(subjects_hours))==len(subjects_hours)
            random.shuffle(subjects_hours)
            subjects_hours=subjects_hours[:int(args.frac_data *len(subjects_hours))]
            print(len(subjects_hours))
            subjects, hours =zip(*subjects_hours)
            datasets['train'].orig_subjects = subjects
            datasets['train'].orig_max_hours = hours
            datasets['train'].reset_sequence_len(datasets['train'].sequence_len, reset_index=False)
            datasets['train'].reset_index()
            assert len(datasets['train']) < orig_len, f"Failed to assert that {len(datasets['train'])} < {orig_len}"
    #             # reset the dataset to that length
    #             datasets['train'].index = train_data_index[:int(fine_tune_args.frac_fine_tune_data *len(train_data_index))]


            sampler = RandomSampler(datasets['train'])

            train_dataloader = DataLoader(
                datasets['train'], sampler=sampler, batch_size=train_dataloader.batch_size,
                num_workers=train_dataloader.num_workers
            )
    
    if args.frac_female != 1:
        import random 
        # get index of train_dataset
        
        if args.frac_data_seed !=0:
            random.seed(args.frac_data_seed)
            
        # get female and male participants.

        
        subjects = datasets['train'].orig_subjects
#         subjects_index = [datasets['train'].index[item] for item in subjects]
        males = datasets['train'].dfs['statics'].loc[idx[subjects], 'gender_2'].values
        assert len(set(subjects))==len(males)
        male_subjects = [item[0] for item in zip(subjects, males) if item[1]==1] 
        #list(subjects[females==0])
        print(male_subjects[:10])
#         print(" ".join([str(datasets['train'].index[item]) for item in male_subjects[:10]]))
        females = datasets['train'].dfs['statics'].loc[idx[subjects], 'gender_1'].values
        assert len(set(subjects))==len(females)
        female_subjects = [item[0] for item in zip(subjects, females) if item[1]==1] #list(subjects[females==1])
        print('\n\n')
        print(female_subjects[:10])
#         print(" ".join([str(datasets['train'].index[item]) for item in female_subjects[:10]]))
        print(len(male_subjects), len(female_subjects))
        
        # print('check that these are females: ', [datasets['train'].index[item] for item in female_subjects[:3]]) # these subject IDs are checkind in the psql db and they are indeed female
        
        
        random.shuffle(female_subjects)
        num_females = min(int(args.frac_female *len(male_subjects)), len(female_subjects))
        female_subjects = female_subjects[:num_females]
        
        print(f'There are {len(female_subjects)} female patients, and {len(male_subjects)} male subjects')
        subjects = female_subjects+male_subjects
        

        orig_len=len(datasets['train'])
        subjects_hours = list(zip(datasets['train'].orig_subjects, datasets['train'].orig_max_hours))
        subjects_hours = [item for item in subjects_hours if item[0] in subjects]
        print(len(subjects_hours))
        subjects, hours =zip(*subjects_hours)
        datasets['train'].orig_subjects = subjects
        datasets['train'].orig_max_hours = hours
        datasets['train'].reset_sequence_len(datasets['train'].sequence_len, reset_index=False)
        datasets['train'].reset_index()
        assert len(datasets['train']) < orig_len, f"Failed to assert that {len(datasets['train'])} < {orig_len}"
#             # reset the dataset to that length
#             datasets['train'].index = train_data_index[:int(fine_tune_args.frac_fine_tune_data *len(train_data_index))]
                   

        sampler = RandomSampler(datasets['train'])
        
#         print('\n\n', vars(datasets['train'])['max_seq_len'])

        train_dataloader = DataLoader(
            datasets['train'], sampler=sampler, batch_size=train_dataloader.batch_size,
            num_workers=train_dataloader.num_workers
        )
            
    tuning_dataloader=None
    if args.epochs==-1:
        # do random stopping and pass in tuning dataset
        tuning_dataloader=DataLoader(
                datasets['tuning'], sampler=RandomSampler(datasets['tuning']), batch_size=train_dataloader.batch_size,
                num_workers=train_dataloader.num_workers
            )
        args.epochs=500
    return run_model(args, datasets, train_dataloader, tqdm=tqdm, tuning_dataloader=tuning_dataloader)
