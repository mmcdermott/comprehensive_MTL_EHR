import sys
sys.path.append('../')

import pickle
from tqdm import tqdm

from latent_patient_trajectories.constants import *
from latent_patient_trajectories.representation_learner.args import *
from latent_patient_trajectories.representation_learner.task_generalizability import *


if __name__=="__main__":
    args = TaskGeneralizabilityArgs.from_commandline()
    main(args)
