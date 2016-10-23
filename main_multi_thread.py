import os
import logging
import tempfile
import sys
from utils import *
import numpy as np
import tensorflow as tf
import signal
import parameters as pms
from logger.logger import Logger
from agent.agent_cotinous_single_thread import TRPOAgentContinousSingleThread
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
        self.gf = GetFlat(self.network.var_list)  # get theta from var_list
        self.gf.session = self.session
        self.sff = SetFromFlat(self.network.var_list)  # set theta from var_List
        self.sff.session = self.session
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=10)

        self.init_jobs()
        if pms.train_flag:
            self.init_logger()

    def init_jobs(self):
        self.jobs = []
        for thread_id in xrange(pms.jobs):
            job = TRPOAgentContinousSingleThread(thread_id, self)
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
        for job in self.jobs:
            job.join()

    def test(self):
        self.load_model(pms.checkpoint_file)
        self.jobs[0].test()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

    def load_model(self , model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.session , model_name)
            else:
                self.saver.restore(self.session , tf.train.latest_checkpoint("checkpoint/"))
        except:
            print "load model %s fail" % (model_name)

def signal_handler():
    sys.exit(0)


if not os.path.isdir("./checkpoint"):
    os.makedirs("./checkpoint")
if not os.path.isdir("./log"):
    os.makedirs("./log")
master = MasterContinous()
if pms.train_flag:
    master.train()
else:
    master.test()
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
