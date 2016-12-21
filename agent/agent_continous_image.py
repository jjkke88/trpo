from utils import *
import numpy as np
import tensorflow as tf
from network.network_continous_image import NetworkContinousImage
from baseline.baseline_tf_image import BaselineTfImage
from storage.storage_image import Storage
from parameters import pms
from agent.agent_base import TRPOAgentBase

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

"""
class for continoust action space with image as input
"""
class TRPOAgent(TRPOAgentBase):

    def __init__(self, env):
        super(TRPOAgent, self).__init__(env)
        self.init_network()
        self.saver = tf.train.Saver(max_to_keep=10)
        self.baseline = BaselineTfImage(self.session)
        self.storage = Storage(self, env, self.baseline)

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
        self.net = NetworkContinousImage("network_continous")
        if pms.min_std is not None:
            log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(pms.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.net.action_n, self.new_dist_info_vars,
                                                              self.old_dist_info_vars)
        surr = -tf.reduce_mean(self.ratio_n * self.net.advant)  # Surrogate loss
        batch_size = tf.shape(self.net.obs)[0]
        batch_size_float = tf.cast(batch_size , tf.float32)
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ent = self.distribution.entropy(self.old_dist_info_vars)
        # ent = tf.reduce_sum(-p_n * tf.log(p_n + eps)) / Nf
        self.losses = [surr, kl, ent]
        var_list = self.net.var_list
        self.gf = GetFlat(var_list)  # get theta from var_list
        self.gf.session = self.session
        self.sff = SetFromFlat(var_list)  # set theta from var_List
        self.sff.session = self.session
        # get g
        self.pg = flatgrad(surr, var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(self.new_dist_info_vars) / batch_size_float
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        self.gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(tf.reduce_sum(self.gvp), var_list)  # get kl''*p
        self.session.run(tf.initialize_all_variables())
        # self.saver = tf.train.Saver(max_to_keep=10)
        # self.load_model(pms.checkpoint_file)

    def learn(self):
        iter_num = 0
        while True:
            print "\n********** Iteration %i ************" % iter_num
            print self.gf().mean()
            stats, theta, thprev = self.train_mini_batch(linear_search=False)
            self.sff(theta)
            for k , v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if iter_num % pms.save_model_times == 0:
                self.save_model(pms.environment_name + "-" + str(iter_num))
            iter_num += 1
