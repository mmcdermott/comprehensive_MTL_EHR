import sys
sys.path.append('../')

from tqdm import tqdm

from latent_patient_trajectories.constants import *
from latent_patient_trajectories.representation_learner.args import *
from latent_patient_trajectories.representation_learner.run_model import *


if __name__=="__main__":
    args = Args.from_commandline()

    main(args, tqdm=tqdm)
