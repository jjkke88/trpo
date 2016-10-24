from utils import *
import gym
import numpy as np
import random
import tensorflow as tf
import time
import threading
import prettytensor as pt

from storage.storage_continous import Storage
from network.network_continous import NetworkContinous
import math
import parameters as pms
import krylov
from logger.logger import Logger
from distribution.diagonal_gaussian import DiagonalGaussian
from baseline.baseline_lstsq import Baseline

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


class TRPOAgentContinousBase(object):

    def __init__(self, env):
        self.env = env
        # if not isinstance(env.observation_space, Box) or \
        #    not isinstance(env.action_space, Discrete):
        #     print("Incompatible spaces.")
        #     exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        print("Action area, high:%f, low%f" % (env.action_space.high, env.action_space.low))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1 / 3.0)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.end_count = 0
        self.paths = []
        self.train = True
        self.baseline = Baseline()
        self.storage = Storage(self, self.env, self.baseline)
        self.distribution = DiagonalGaussian(pms.action_shape)
        self.net = None

    # def init_logger(self):
    #     head = ["average_episode_std" , "sum steps episode number" "total number of episodes" ,
    #             "Average sum of rewards per episode" ,
    #             "KL between old and new distribution" , "Surrogate loss" , "Surrogate loss prev" , "ds" , "entropy" ,
    #             "mean_advant"]
    #     self.logger = Logger(head)

    def init_network(self):
        raise NotImplementedError

    def get_samples(self, path_number):
        for i in range(path_number):
            self.storage.get_single_path()

    def get_action(self, obs, *args):
        if self.net==None:
            raise NameError("network have not been defined")
        obs = np.expand_dims(obs, 0)
        # action_dist_logstd = np.expand_dims([np.log(pms.std)], 0)
        if pms.use_std_network:
            action_dist_means_n, action_dist_logstds_n = self.session.run(
                [self.action_dist_means_n, self.action_dist_logstds_n],
                {self.obs: obs})
            if pms.train_flag:
                rnd = np.random.normal(size=action_dist_means_n[0].shape)
                action = rnd * np.exp(action_dist_logstds_n[0]) + action_dist_means_n[0]
            else:
                action = action_dist_means_n[0]
            # action = np.clip(action, pms.min_a, pms.max_a)
            return action, dict(mean=action_dist_means_n[0], log_std=action_dist_logstds_n[0])
        else:
            action_dist_logstd = np.expand_dims([np.log(pms.std)], 0)
            action_dist_means_n = self.session.run(self.net.action_dist_means_n,
                                                   {self.net.obs: obs})
            if pms.train_flag:
                rnd = np.random.normal(size=action_dist_means_n[0].shape)
                action = rnd * np.exp(action_dist_logstd[0]) + action_dist_means_n[0]
            else:
                action = action_dist_means_n[0]
            # action = np.clip(action, pms.min_a, pms.max_a)
            return action, dict(mean=action_dist_means_n[0], log_std=action_dist_logstd[0])

    def train_mini_batch(self, parallel=False):
        # Generating paths.
        print("Rollout")
        start_time = time.time()
        self.get_samples(pms.paths_number)
        paths = self.storage.get_paths()  # get_paths
        # Computing returns and estimating advantage function.
        sample_data = self.storage.process_paths(paths)
        agent_infos = sample_data["agent_infos"]
        obs_n = sample_data["observations"]
        action_n = sample_data["actions"]
        advant_n = sample_data["advantages"]
        n_samples = len(obs_n)
        inds = np.random.choice(n_samples , int(math.floor(n_samples * pms.subsample_factor)), replace=False)
        obs_n = obs_n[inds]
        action_n = action_n[inds]
        advant_n = advant_n[inds]
        action_dist_means_n = np.array([agent_info["mean"] for agent_info in agent_infos[inds]])
        action_dist_logstds_n = np.array([agent_info["log_std"] for agent_info in agent_infos[inds]])
        feed = {self.net.obs: obs_n ,
                self.net.advant: advant_n ,
                self.net.old_dist_means_n: action_dist_means_n ,
                self.net.old_dist_logstds_n: action_dist_logstds_n ,
                self.net.action_dist_logstds_n: action_dist_logstds_n ,
                self.net.action_n: action_n
                }

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        average_episode_std = np.mean(np.exp(action_dist_logstds_n))
        thprev = self.gf()  # get theta_old
        def fisher_vector_product(p):
            feed[self.flat_tangent] = p
            return self.session.run(self.fvp , feed) + pms.cg_damping * p
        g = self.session.run(self.pg , feed_dict=feed)
        stepdir = krylov.cg(fisher_vector_product , g , cg_iters=pms.cg_iters)
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
        fullstep = stepdir * np.sqrt(2.0 * pms.max_kl / shs)
        neggdotstepdir = -g.dot(stepdir)
        def loss(th):
            self.sff(th)
            return self.session.run(self.losses , feed_dict=feed)
        surr_prev , kl_prev , ent_prev = loss(thprev)
        mean_advant = np.mean(advant_n)
        if parallel is True:
            theta = linesearch_parallel(loss , thprev , fullstep , neggdotstepdir)
        else:
            theta = linesearch(loss , thprev , fullstep , neggdotstepdir)
        surrafter, kloldnew, entnew = self.session.run(self.losses , feed_dict=feed)
        stats = {}
        stats["average_episode_std"] = average_episode_std
        stats["sum steps of episodes"] = sample_data["sum_episode_steps"]
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        stats["KL between old and new distribution"] = kloldnew
        stats["Surrogate loss"] = surrafter
        stats["Surrogate loss prev"] = surr_prev
        stats["entropy"] = ent_prev
        stats["mean_advant"] = mean_advant
        return stats, theta-thprev

    def learn(self):
        raise NotImplementedError

    def test(self, model_name):
        self.load_model(model_name)
        if pms.record_movie:
            for i in range(100):
                self.storage.get_single_path()
            self.env.env.monitor.close()
            if pms.upload_to_gym:
                gym.upload("log/trpo",algorithm_id='alg_8BgjkAsQRNiWu11xAhS4Hg', api_key='sk_IJhy3b2QkqL3LWzgBXoVA')
        else:
            for i in range(50):
                self.storage.get_single_path()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

    def load_model(self, model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.session, model_name)
            else:
                self.saver.restore(self.session, tf.train.latest_checkpoint("checkpoint/"))
        except:
            print "load model %s fail" % (model_name)
