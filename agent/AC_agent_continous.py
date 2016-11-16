from utils import *
import numpy as np
import tensorflow as tf
from network.network_continous import NetworkContinous
from parameters import pms
from agent.agent_base import TRPOAgentBase
from logger.logger import Logger
import math
import time

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
class ACAgent(TRPOAgentBase):

    def __init__(self, env):
        super(ACAgent, self).__init__(env)
        self.init_network()
        self.saver = tf.train.Saver(max_to_keep=10)


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
        self.net = NetworkContinous("network_continous_ac")
        if pms.min_std is not None:
            log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(pms.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_new_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        # surr = - log(\pi_\theta)*(Q^\pi-V^\pi)
        value_loss = 0.5*tf.square(self.net.advant)
        surr = -tf.reduce_sum(self.likehood_new_action_dist*tf.stop_gradient(self.net.advant)+value_loss)  # Surrogate loss

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

        self.session.run(tf.initialize_all_variables())
        # self.saver = tf.train.Saver(max_to_keep=10)
        # self.load_model(pms.checkpoint_file)

    def init_logger(self):
        head = ["std", "rewards"]
        self.logger = Logger(head)

    def train_mini_batch(self, parallel=False, linear_search=True):
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
        inds = np.random.choice(n_samples, int(math.floor(n_samples * pms.subsample_factor)), replace=False)
        # inds = range(n_samples)
        obs_n = obs_n[inds]
        action_n = action_n[inds]
        advant_n = advant_n[inds]
        action_dist_means_n = np.array([agent_info["mean"] for agent_info in agent_infos[inds]])
        action_dist_logstds_n = np.array([agent_info["log_std"] for agent_info in agent_infos[inds]])
        feed = {self.net.obs: obs_n,
                self.net.advant: advant_n,
                self.net.old_dist_means_n: action_dist_means_n,
                self.net.old_dist_logstds_n: action_dist_logstds_n,
                self.net.action_n: action_n
                }

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        thprev = self.gf()  # get theta_old

        g = self.session.run(self.pg, feed_dict=feed)
        theta = thprev+0.01*g
        stats = {}
        stats["sum steps of episodes"] = sample_data["sum_episode_steps"]
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        return stats, theta, thprev

    def learn(self):
        self.init_logger()
        iter_num = 0
        while True:
            print "\n********** Iteration %i ************" % iter_num
            stats, theta, thprev = self.train_mini_batch(linear_search=False)
            self.sff(theta)
            self.logger.log_row([stats["Average sum of rewards per episode"], self.session.run(self.net.action_dist_logstd_param)])
            for k , v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))
            if iter_num % pms.save_model_times == 0:
                self.save_model(pms.environment_name + "-" + str(iter_num))
            iter_num += 1
