import argparse
import os
import sys
import csv
import socket
import numpy as np
from tqdm import tqdm
from simulation_base.env import resume_env, nb_actuations

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, HumanOutputFormat, DEBUG
from stable_baselines3.sac import SAC

from gym.wrappers.time_limit import TimeLimit
from checkpoint_callback_sb3 import CheckpointCallback
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

    config["learning_rate"] = 5e-5
    config["learning_starts"] = (1 * nb_actuations * number_servers) if restoredir is None else 0
    config["batch_size"] = 128

    config["tau"] = 5e-3
    config["gamma"] = 0.99
    config["train_freq"] = 1
    config["target_update_interval"] = 1
    config["gradient_steps"] = 32

    config["buffer_size"] = int(1e5)
    config["optimize_memory_usage"] = False

    config["ent_coef"] = "auto_0.01"
    config["target_entropy"] = "auto"

    checkpoint_callback = CheckpointCallback(
                                            save_freq=max(128//number_servers, 1),
                                            #num_to_keep=5,
                                            #save_buffer=True,
                                            #save_env_stats=True,
                                            save_path=savedir,
                                            name_prefix='SAC_model')


    env = SubprocVecEnv([resume_env(nb_actuations,i) for i in range(number_servers)], start_method='spawn')

    model = SAC('MlpPolicy', VecNormalize(env, gamma=config["gamma"]), tensorboard_log=savedir, **config)
    model.learn(100000, callback=[checkpoint_callback], log_interval=1)



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

