import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import tempfile
import sys
from environment import Environment
from agent.agent_continous import TRPOAgent
import parameters as pms
from utils import *
import threading
import gym
import numpy as np
import random
import tensorflow as tf
import time
import threading
import prettytensor as pt
import signal
import multiprocessing
from storage.storage_continous import Storage
from storage.storage_continous import Rollout
import math
import parameters as pms
import krylov
from logger.logger import Logger
from agent.agent_continous_single_process import TRPOAgentContinousSingleProcess
from distribution.diagonal_gaussian import DiagonalGaussian
from baseline.baseline_lstsq import Baseline
from environment import Environment
from network.network_continous import NetworkContinous

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)


class MasterContinous(object):
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1 / 3.0)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.network = NetworkContinous("master")
        self.gf = GetFlat(self.session, self.network.var_list)  # get theta from var_list
        self.sff = SetFromFlat(self.session, self.network.var_list)  # set theta from var_List
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=10)

        self.init_jobs()
        if pms.train_flag:
            self.init_logger()

    def init_jobs(self):
        self.jobs = []
        for thread_id in xrange(pms.jobs):
            job = TRPOAgentContinousSingleProcess(thread_id, self)
            job.daemon = True
            self.jobs.append(job)

    def init_logger(self):
        head = ["average_episode_std", "sum steps episode number" "total number of episodes",
                "Average sum of rewards per episode",
                "KL between old and new distribution", "Surrogate loss", "Surrogate loss prev", "ds", "entropy",
                "mean_advant"]
        self.logger = Logger(head)

    def get_parameters(self):
        return self.gf()

    def apply_gradient(self, gradient):
        theta_prev = self.gf()
        theta_after = theta_prev + gradient
        self.sff(theta_after)

    def train(self):
        signal.signal(signal.SIGINT, signal_handler)
        for job in self.jobs:
            job.start()

    def test(self):
        self.jobs[0].test(pms.checkpoint_file)

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

def signal_handler():
    sys.exit(0)

master = MasterContinous()

if pms.train_flag:
    master.train()
else:
    master.test()
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
