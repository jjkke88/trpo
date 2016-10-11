from utils import *
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

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

class NetworkContinous(object):
    def __init__(self, scope):
        with tf.variable_scope("%s_shared" % scope):
            self.obs = obs = tf.placeholder(
                dtype, shape=[None, pms.obs_shape], name="%s_obs"%scope)
            self.action_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="%s_action"%scope)
            self.advant = tf.placeholder(dtype, shape=[None], name="%s_advant"%scope)
            self.old_dist_means_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                   name="%s_oldaction_dist_means"%scope)
            self.old_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                     name="%s_oldaction_dist_logstds"%scope)

            # Create mean network.
            # self.fp_mean1, weight_fp_mean1, bias_fp_mean1 = linear(self.obs, 32, activation_fn=tf.nn.tanh, name="fp_mean1")
            # self.fp_mean2, weight_fp_mean2, bias_fp_mean2 = linear(self.fp_mean1, 32, activation_fn=tf.nn.tanh, name="fp_mean2")
            # self.action_dist_means_n, weight_action_dist_means_n, bias_action_dist_means_n = linear(self.fp_mean2, pms.action_shape, name="action_dist_means")
            self.action_dist_means_n = (pt.wrap(self.obs).
                                        fully_connected(16, activation_fn=tf.nn.tanh,
                                                        init=tf.random_normal_initializer(stddev=1.0), bias=False, name="%s_fc1"%scope).
                                        fully_connected(16, activation_fn=tf.nn.tanh,
                                                        init=tf.random_normal_initializer(stddev=1.0), bias=False, name="%s_fc2"%scope).
                                        fully_connected(pms.action_shape, init=tf.random_normal_initializer(stddev=1.0),
                                                        bias=False, name="%s_fc3"%scope))

            self.N = tf.shape(obs)[0]
            Nf = tf.cast(self.N, dtype)
            # Create std network.
            if pms.use_std_network:
                self.action_dist_logstds_n = (pt.wrap(self.obs).
                                              fully_connected(16, activation_fn=tf.nn.tanh,
                                                              init=tf.random_normal_initializer(stddev=1.0),
                                                              bias=False, name="%s_fcstd1"%scope).
                                              fully_connected(16, activation_fn=tf.nn.tanh,
                                                              init=tf.random_normal_initializer(stddev=1.0),
                                                              bias=False, name="%s_fcstd2"%scope).
                                              fully_connected(pms.action_shape,
                                                              init=tf.random_normal_initializer(stddev=1.0), bias=False, name="%s_fcstd3"%scope))
            else:
                self.action_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="%s_logstd"%scope)


            self.var_list = [v for v in tf.trainable_variables()if v.name.startswith(scope)]

    def get_action_dist_means_n(self, session, obs):
        return session.run(self.action_dist_means_n,
                         {self.obs: obs})

