import argparse
import os
import sys
import csv
import socket
import numpy as np
from tqdm import tqdm
from simulation_base.env import resume_env, nb_actuations
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
from stable_baselines3.sac import SAC
import torch
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import CheckpointCallback
#from tensorforce.agents import Agent
#from tensorforce.execution import Runner


#from RemoteEnvironmentClient import RemoteEnvironmentClient


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
    ap.add_argument("-s", "--savedir", required=False,
                    help="Directory into which to save the NN. Defaults to 'saver_data'.", type=str,
                    default='saver_data')

    args = vars(ap.parse_args())

    number_servers = args["number_servers"]
    savedir = args["savedir"]

    config = {}

    config["learning_rate"] = 3e-4
    config["learning_starts"] = 0
    config["batch_size"] = 128
    config["top_quantiles_to_drop_per_net"] = 3
    config["policy_kwargs"] = {
                                "n_critics": 5,
                                "n_quantiles": 25,
                                "net_arch": dict(pi=[256, 256], qf=[512, 512, 512])
                                }

    config["tau"] = 5e-3
    config["gamma"] = 0.99
    config["train_freq"] = 1
    config["target_update_interval"] = 1
    config["gradient_steps"] = 20

    config["buffer_size"] = int(1e5)
    config["optimize_memory_usage"] = False

    config["ent_coef"] = "auto_0.01"
    config["target_entropy"] = "auto"

    model = TQC('MlpPolicy', VecFrameStack(env, n_stack=10), tensorboard_log=savedir, **config)

    checkpoint_callback = CheckpointCallback(
                                            save_freq=max(10, 1),
                                            #num_to_keep=5,
                                            #save_buffer=True,
                                            #save_env_stats=True,
                                            save_path=savedir,
                                            name_prefix='TQC_model')


    env = SubprocVecEnv([resume_env(nb_actuations,i) for i in range(number_servers)], start_method='spawn')
    
    model.learn(15000000, callback=[checkpoint_callback], log_interval=1)

   

    name = "returns_tf.csv"
    if (not os.path.exists("saved_models")):
        os.mkdir("saved_models")

    # If continuing previous training - append returns
    if (os.path.exists("saved_models/" + name)):
        prev_eps = np.genfromtxt("saved_models/" + name, delimiter=';', skip_header=1)
        offset = int(prev_eps[-1, 0])
        print(offset)
        with open("saved_models/" + name, "a") as csv_file:
            spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
            for ep in range(len(runner.episode_rewards)):
                spam_writer.writerow([offset + ep + 1, runner.episode_rewards[ep]])
    # If strating training from zero - write returns
    elif (not os.path.exists("saved_models/" + name)):
        with open("saved_models/" + name, "w") as csv_file:
            spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
            spam_writer.writerow(["Episode", "Return"])
            for ep in range(len(runner.episode_rewards)):
                spam_writer.writerow([ep + 1, runner.episode_rewards[ep]])





    print("Agent and Runner closed -- Learning complete -- End of script")
    os._exit(0)

