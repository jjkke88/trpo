from utils import *
import threading
import gym
import numpy as np
import random
import tensorflow as tf
import time
import threading
import prettytensor as pt

from storage.storage_continous import Storage
from storage.storage_continous import Rollout
import math
import parameters as pms
import krylov
from logger.logger import Logger
from distribution.diagonal_gaussian import DiagonalGaussian
from baseline.baseline_lstsq import Baseline
from environment import Environment
from network.network_continous import NetworkContinous
from agent.agent_continous_base import TRPOAgentContinousBase

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


class TRPOAgentContinousSingleThread(TRPOAgentContinousBase, threading.Thread):

    def __init__(self, thread_id, master):
        print "create thread %d"%(thread_id)
        self.thread_id = thread_id
        threading.Thread.__init__(self, name="thread_%d" % thread_id)
        self.master = master
        self.env = env = Environment(gym.make(pms.environment_name))
        TRPOAgentContinousBase.__init__(self, env)

        self.session = self.master.session
        self.init_network()


    def init_network(self):
        """
            [input]
            self.obs
            self.action_n
            self.advant
            self.old_dist_means_n
            self.old_dist_logstds_n
            [output]
            self.action_dist_means_n
            self.action_dist_logstds_n
            var_list
            """
        self.net = NetworkContinous(str(self.thread_id))
        if pms.min_std is not None:
            log_std_var = tf.maximum(self.net.action_dist_logstds_n , np.log(pms.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n , log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n , log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n , self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.net.action_n , self.new_dist_info_vars ,
                                                              self.old_dist_info_vars)
        surr = -tf.reduce_mean(self.ratio_n * self.net.advant)  # Surrogate loss
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars , self.new_dist_info_vars))
        ent = self.distribution.entropy(self.old_dist_info_vars)
        # ent = tf.reduce_sum(-p_n * tf.log(p_n + eps)) / Nf
        self.losses = [surr , kl , ent]
        var_list = self.net.var_list
        self.gf = GetFlat(var_list)  # get theta from var_list
        self.gf.session = self.session
        self.sff = SetFromFlat(var_list)  # set theta from var_List
        self.sff.session = self.session
        # get g
        self.pg = flatgrad(surr , var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(self.new_dist_info_vars)
        grads = tf.gradients(kl_firstfixed , var_list)
        self.flat_tangent = tf.placeholder(dtype , shape=[None])
        shapes = map(var_shape , var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)] , shape)
            tangents.append(param)
            start += size
        self.gvp = [tf.reduce_sum(g * t) for (g , t) in zip(grads , tangents)]
        self.fvp = flatgrad(tf.reduce_sum(self.gvp) , var_list)  # get kl''*p

    def run(self):
        self.learn()

    def learn(self):
        i = 0
        sum_gradient = 0
        while True:
            self.sff(self.master.get_parameters())

            # Generating paths.
            stats, gradient = self.train_mini_batch(parallel=True)
            sum_gradient += gradient
            self.master.apply_gradient(sum_gradient)
            print "\n********** Iteration %i ************" % i
            for k , v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            sum_gradient = 0
            if self.thread_id==1 and i%pms.save_model_times==0:
                self.save_model(pms.environment_name + "-" + str(i))
            i += 1


    def test(self):
        self.sff(self.master.get_parameters())
        for i in range(50):
            self.storage.get_single_path()
