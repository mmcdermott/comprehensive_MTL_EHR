"""
fine_tune.py
Fine tunes a pre-trained model on a specific (single) task
"""

import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

import json, os, pickle
from copy import deepcopy
from tqdm import tqdm
import glob


# TODO: check these imports.
from ..utils import *
from ..constants import *
from ..data_utils import *
from ..BERT.model import *
from ..BERT.constants import *

from .fts_decoder import *
from .evaluator import *
from .meta_model import *
from .run_model import setup_for_run, train_meta_model

def fine_tune_model(fine_tune_args, meta_model_args, datasets, train_dataloader, tqdm=None, meta_model=None, tuning_dataloader=None):
    print('in fine tune model')
    reloaded=False
    # check if models are in the subdir:
    if fine_tune_args.frac_fine_tune_data==1:
        check_path_name = os.path.join(fine_tune_args.run_dir, fine_tune_args.fine_tune_task)
        print('1 set dir to: ', check_path_name) 
        if  fine_tune_args.frac_female !=1:
            check_path_name = os.path.join(fine_tune_args.run_dir, fine_tune_args.fine_tune_task+'_F'+str(fine_tune_args.frac_female).replace('.','-')+'_'+str(int(fine_tune_args.frac_fine_tune_data_seed)))
            print('2 set dir to: ', check_path_name) 
    elif (fine_tune_args.frac_fine_tune_data!=1) and (fine_tune_args.frac_fine_tune_data is not None):
        check_path_name = os.path.join(fine_tune_args.run_dir, fine_tune_args.fine_tune_task+'_'+str(fine_tune_args.frac_fine_tune_data).replace('.','-')+'_'+str(int(fine_tune_args.frac_fine_tune_data_seed)))
    print('3 set dir to: ', check_path_name) 
    if os.path.exists(check_path_name):
        print('check_path_name exists')
        all_models = glob.glob(os.path.join(check_path_name, 'model.epoch*'))
        if len(all_models)==0:
            pass
        elif os.path.join(check_path_name,'model.epoch-'+str(int(meta_model_args.epochs-1))) in all_models:
            return # model is fully fine-tuned
        else:
            # load the model from the latest epoch in this dir
            print('creating a model in:', check_path_name)
            if meta_model is None:
                meta_model = MetaModel(
                    meta_model_args, datasets['train'][0],
                    class_names = {'tasks_binary_multilabel': datasets['train'].get_binary_multilabel_keys()}
                )
                meta_model.run_dir=check_path_name
                reloaded, epoch = meta_model.load()
                assert reloaded, "Can't fine-tune a not-yet-trained model!"
                print('Reloaded fine-tuned model from ', meta_model.run_dir)
     
            

    if meta_model is None:
        meta_model = MetaModel(
            meta_model_args, datasets['train'][0],
            class_names = {'tasks_binary_multilabel': datasets['train'].get_binary_multilabel_keys()}
        )
        
    if not(reloaded):
        reloaded, epoch = meta_model.load()
        assert reloaded, "Can't fine-tune a not-yet-trained model!"
        epoch=0
        reloaded=False

    meta_model.run_dir = fine_tune_args.fine_tune_run_dir
    meta_model.freeze_representation()

    # For fine-tuning, we want to ablate away everything *but* the target task.
    ablate = [k for k in ABLATION_GROUPS.keys() if k != fine_tune_args.fine_tune_task]
    meta_model.ablate(ablate, post_init=True)
    
    


    # Train it from scractch with the representation frozen and task_weights appropriately set.
    train_meta_model(meta_model, train_dataloader, meta_model_args, reloaded=reloaded, epoch=epoch, tuning_dataloader=tuning_dataloader, train_embedding_after=fine_tune_args.train_embedding_after)
    return meta_model

def main(fine_tune_args, tqdm):
    assert os.path.isdir(fine_tune_args.run_dir), "Run dir must exist!"
    assert fine_tune_args.fine_tune_task in ABLATION_GROUPS,\
        "Invalid fine tune task: %s" % fine_tune_args.fine_tune_task
    assert ((fine_tune_args.frac_fine_tune_data >0) and (fine_tune_args.frac_fine_tune_data<=1)), "frac_fine_tune_data must be in the range(0, 1]"
    
#     print("meta_model_args filename: ", os.path.join(fine_tune_args.run_dir, ARGS_FILENAME)) 
    print(os.path.join(fine_tune_args.run_dir, ARGS_FILENAME))

    meta_model_args = Args.from_json_file(os.path.join(fine_tune_args.run_dir, ARGS_FILENAME))
    meta_model_args.set_to_eval_mode=""
    
    # make new fine tune
    fine_tune_dir_name = fine_tune_args.fine_tune_task
    if hasattr(fine_tune_args, 'frac_fine_tune_data'):
        if fine_tune_args.frac_fine_tune_data!=1:
            fine_tune_dir_name = fine_tune_args.fine_tune_task + f'_{str(fine_tune_args.frac_fine_tune_data).replace(".","-")}_{str(fine_tune_args.frac_fine_tune_data_seed)}'
    if hasattr(fine_tune_args, 'frac_female'):
        # assert that it cannot have both(for now)
        if hasattr(fine_tune_args, 'frac_fine_tune_data'):
            assert fine_tune_args.frac_fine_tune_data==1, 'cannot have both frac_fine_tune_data and frac_female'
        if fine_tune_args.frac_female!=1:
            fine_tune_dir_name = fine_tune_args.fine_tune_task + f'_F{str(fine_tune_args.frac_female).replace(".","-")}_{str(fine_tune_args.frac_fine_tune_data_seed)}'
        
    
    fine_tune_run_dir = os.path.join(meta_model_args.run_dir, fine_tune_dir_name)
    #assert not os.path.exists(fine_tune_run_dir), "Overwriting existing Fine-tune dir!"
    if not os.path.exists(fine_tune_run_dir):
        os.makedirs(fine_tune_run_dir)

    fine_tune_args.fine_tune_run_dir = fine_tune_run_dir

    fine_tune_meta_model_args = deepcopy(meta_model_args)
    fine_tune_meta_model_args.run_dir = fine_tune_run_dir
    # fine_tune_meta_model_args[''] = fine_tune_args[''] #eventually do fine_tune_data_frac
    ablate = [k for k in ABLATION_GROUPS.keys() if k != fine_tune_args.fine_tune_task]
    fine_tune_meta_model_args.ablate = ablate

    fine_tune_args.to_json_file(os.path.join(fine_tune_args.run_dir, FINE_TUNE_ARGS_FILENAME))
    fine_tune_meta_model_args.to_json_file(os.path.join(fine_tune_meta_model_args.run_dir, ARGS_FILENAME))

    set_to_eval_mode = None
    if fine_tune_args.do_match_train_windows:
        meta_model_args.set_to_eval_mode = EVAL_MODES_BY_ABLATION_GROUPS[fine_tune_args.fine_tune_task]
    datasets, train_dataloader = setup_for_run(meta_model_args)
    
#     print('\n\n', vars(train_dataloader.dataset)['max_seq_len'])
#     print(meta_model_args)
    assert vars(train_dataloader.dataset)['max_seq_len'] == meta_model_args.max_seq_len
    assert vars(datasets['train'])['max_seq_len'] == meta_model_args.max_seq_len

    if fine_tune_args.frac_fine_tune_data != 1:
        import random        
        # get index of train_dataset
        
        if fine_tune_args.frac_fine_tune_data_seed !=0:
            random.seed(fine_tune_args.frac_fine_tune_data_seed)
            

        orig_len=len(datasets['train'])
        subjects_hours = list(zip(datasets['train'].orig_subjects, datasets['train'].orig_max_hours))
        assert len(set(subjects_hours))==len(subjects_hours)
        random.shuffle(subjects_hours)
        subjects_hours=subjects_hours[:int(fine_tune_args.frac_fine_tune_data *len(subjects_hours))]
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
    
    
    if fine_tune_args.frac_female != 1:
        import random 
        # get index of train_dataset
        
        if fine_tune_args.frac_fine_tune_data_seed !=0:
            random.seed(fine_tune_args.frac_fine_tune_data_seed)
            
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
        num_females = min(int(fine_tune_args.frac_female *len(male_subjects)), len(female_subjects))
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
    if meta_model_args.epochs==-1:
        # do random stopping and pass in tuning dataset
        tuning_dataloader=DataLoader(
                datasets['tuning'], sampler=RandomSampler(datasets['tuning']), batch_size=train_dataloader.batch_size,
                num_workers=train_dataloader.num_workers
            )
        meta_model_args.epochs=500
        
#     print(fine_tune_args)
#     print(meta_model_args)

    

    return fine_tune_model(fine_tune_args, meta_model_args, datasets, train_dataloader, tqdm=tqdm, tuning_dataloader=tuning_dataloader)
