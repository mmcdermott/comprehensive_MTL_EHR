import sys
sys.path.append('../..')

from tqdm import tqdm

from latent_patient_trajectories.constants import *
from latent_patient_trajectories.representation_learner.args import *
from latent_patient_trajectories.representation_learner.evaluator import *


if __name__=="__main__":
    args = EvalArgs.from_commandline()
    print(args.run_dir)
    main(args, tqdm=tqdm, datasets=None)
