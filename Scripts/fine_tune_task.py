import sys
sys.path.append('../')

import pickle
from tqdm import tqdm

from latent_patient_trajectories.constants import *
from latent_patient_trajectories.representation_learner.args import FineTuneArgs
from latent_patient_trajectories.representation_learner.fine_tune import *


if __name__=="__main__":
    args = FineTuneArgs.from_commandline()
    main(args, tqdm)
