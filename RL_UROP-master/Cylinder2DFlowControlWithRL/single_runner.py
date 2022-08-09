import os
import socket
import numpy as np
import csv
import sys
import os

from Env2DCylinderModified import Env2DCylinderModified
from probe_positions import probe_positions
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
import numpy as np
from dolfin import Expression
#from printind.printind_function import printi, printiv
import math
from stable_baselines3.common.monitor import Monitor
from gym.wrappers.time_limit import TimeLimit
import os

import argparse
import os
import json
import pandas as pd
from tqdm import trange
from sb3_contrib import TQC
from stable_baselines3 import SAC
from Env2DCylinderModified import Env2DCylinderModified
from simulation_base.env import resume_env, nb_actuations, simulation_duration


# If previous evaluation results exist, delete them
if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")


if __name__ == '__main__':

    saver_restore = '/rds/general/user/jz1720/home/TQCFrameStack/RL_UROP-master/Cylinder2DFlowControlWithRL/saver_data/TQC_model_277140_steps.zip'
    agent = TQC.load(saver_restore)
    env = SubprocVecEnv([resume_env(nb_actuations,i) for i in range(1)], start_method='spawn')
    env = VecFrameStack(env, n_stack=12)


    state = env.reset()
    #example_environment.render = True

    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    single_run_duration = 250  # In non-dimensional time
    action_steps = int(single_run_duration / action_step_size)

    #internals = agent.initial_internals()

    for k in range(action_steps):
        action, _ = agent.predict(state, deterministic=True)
        state, rw, done, _ = env.step(action)
