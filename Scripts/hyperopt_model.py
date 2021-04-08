import sys
sys.path.append('../')

import pickle
from tqdm import tqdm

from latent_patient_trajectories.constants import *
from latent_patient_trajectories.representation_learner.args import *
from latent_patient_trajectories.representation_learner.hyperparameter_search import *

if __name__=="__main__":
    args = HyperparameterSearchArgs.from_commandline()

    trials = main(args, tqdm=tqdm)

    with open(os.path.join(args.search_dir, str(args.rotation), 'trials.pkl'), mode='wb') as f:
        pickle.dump(trials, f)

    print(trials.best_trial['result'])
    print(trials.best_trial['misc']['vals'])
