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

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


class TRPOAgent(object):

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
        self.init_network()
        if pms.train_flag:
            self.init_logger()

    def init_logger(self):
        head = ["average_episode_std", "sum steps episode number" "total number of episodes", "Average sum of rewards per episode",
                "KL between old and new distribution", "Surrogate loss", "Surrogate loss prev", "ds", "entropy",
                "mean_advant"]
        self.logger = Logger(head)

    def init_network(self):
        self.obs = obs = tf.placeholder(
            dtype, shape=[None, pms.obs_shape], name="obs")
        self.action_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="action")
        self.advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.old_dist_means_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                               name="oldaction_dist_means")
        self.old_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                 name="oldaction_dist_logstds")

        # Create mean network.
        # self.fp_mean1, weight_fp_mean1, bias_fp_mean1 = linear(self.obs, 32, activation_fn=tf.nn.tanh, name="fp_mean1")
        # self.fp_mean2, weight_fp_mean2, bias_fp_mean2 = linear(self.fp_mean1, 32, activation_fn=tf.nn.tanh, name="fp_mean2")
        # self.action_dist_means_n, weight_action_dist_means_n, bias_action_dist_means_n = linear(self.fp_mean2, pms.action_shape, name="action_dist_means")
        self.action_dist_means_n = (pt.wrap(self.obs).
                                    fully_connected(16, activation_fn=tf.nn.tanh,
                                                    init=tf.random_normal_initializer(stddev=1.0), bias=False).
                                    fully_connected(16, activation_fn=tf.nn.tanh,
                                                    init=tf.random_normal_initializer(stddev=1.0), bias=False).
                                    fully_connected(pms.action_shape, init=tf.random_normal_initializer(stddev=1.0),
                                                    bias=False))

        self.N = tf.shape(obs)[0]
        Nf = tf.cast(self.N, dtype)
        # Create std network.
        if pms.use_std_network:
            self.action_dist_logstds_n = (pt.wrap(self.obs).
                                          fully_connected(16, activation_fn=tf.nn.tanh,
                                                          init=tf.random_normal_initializer(stddev=1.0),
                                                          bias=False).
                                          fully_connected(16, activation_fn=tf.nn.tanh,
                                                          init=tf.random_normal_initializer(stddev=1.0),
                                                          bias=False).
                                          fully_connected(pms.action_shape,
                                                          init=tf.random_normal_initializer(stddev=1.0), bias=False))
        else:
            self.action_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="logstd")
        if pms.min_std is not None:
            log_std_var = tf.maximum(self.action_dist_logstds_n, np.log(pms.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)

        self.old_dist_info_vars = dict(mean=self.old_dist_means_n, log_std=self.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.action_dist_means_n, log_std=self.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.action_n, self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.action_n, self.new_dist_info_vars,
                                                              self.old_dist_info_vars)

        surr = -tf.reduce_mean(self.ratio_n * self.advant)  # Surrogate loss
        kl = tf.reduce_mean(self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars))
        ent = self.distribution.entropy(self.old_dist_info_vars)
        # ent = tf.reduce_sum(-p_n * tf.log(p_n + eps)) / Nf
        self.losses = [surr, kl, ent]

        var_list = tf.trainable_variables()
        self.gf = GetFlat(var_list)  # get theta from var_list
        self.gf.session = self.session
        self.sff = SetFromFlat(var_list)  # set theta from var_List
        self.sff.session = self.session
        # get g
        self.pg = flatgrad(surr, var_list)
        # get A

        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = self.distribution.kl_sym_firstfixed(dict(mean=self.action_dist_means_n,
                                        log_std=self.action_dist_logstds_n))

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
        self.saver = tf.train.Saver(max_to_keep=10)
        self.load_model(pms.checkpoint_file)

    def get_samples(self, path_number):
        for i in range(path_number):
            self.storage.get_single_path()

    def get_action(self, obs, *args):
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
            action_dist_means_n = self.session.run(self.action_dist_means_n,
                                                   {self.obs: obs})
            if pms.train_flag:
                rnd = np.random.normal(size=action_dist_means_n[0].shape)
                action = rnd * np.exp(action_dist_logstd[0]) + action_dist_means_n[0]
            else:
                action = action_dist_means_n[0]
            # action = np.clip(action, pms.min_a, pms.max_a)
            return action, dict(mean=action_dist_means_n[0], log_std=action_dist_logstd[0])

    def learn(self):
        start_time = time.time()
        numeptotal = 0
        i = 0
        while True:
            # Generating paths.
            print("Rollout")
            self.get_samples(pms.paths_number)
            paths = self.storage.get_paths()  # get_paths
            # Computing returns and estimating advantage function.
            sample_data = self.storage.process_paths(paths)

            agent_infos = sample_data["agent_infos"]
            obs_n = sample_data["observations"]
            action_n = sample_data["actions"]
            advant_n = sample_data["advantages"]
            n_samples = len(obs_n)
            inds = np.random.choice(n_samples, math.floor(n_samples * pms.subsample_factor), replace=False)
            obs_n = obs_n[inds]
            action_n = action_n[inds]
            advant_n = advant_n[inds]
            action_dist_means_n = np.array([agent_info["mean"] for agent_info in agent_infos[inds]])
            action_dist_logstds_n = np.array([agent_info["log_std"] for agent_info in agent_infos[inds]])
            feed = {self.obs: obs_n,
                    self.advant: advant_n,
                    self.old_dist_means_n: action_dist_means_n,
                    self.old_dist_logstds_n: action_dist_logstds_n,
                    self.action_dist_logstds_n: action_dist_logstds_n,
                    self.action_n: action_n
                    }

            episoderewards = np.array([path["rewards"].sum() for path in paths])
            average_episode_std = np.mean(np.exp(action_dist_logstds_n))

            print "\n********** Iteration %i ************" % i
            for iter_num_per_train in range(pms.iter_num_per_train):
                # if not self.train:
                #     print("Episode mean: %f" % episoderewards.mean())
                #     self.end_count += 1
                #     if self.end_count > 100:
                #         break
                if self.train:
                    thprev = self.gf()  # get theta_old

                    def fisher_vector_product(p):
                        feed[self.flat_tangent] = p
                        return self.session.run(self.fvp, feed) + pms.cg_damping * p

                    g = self.session.run(self.pg, feed_dict=feed)
                    stepdir = krylov.cg(fisher_vector_product, g, cg_iters=pms.cg_iters)
                    shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
                    fullstep = stepdir * np.sqrt(2.0 * pms.max_kl / shs)
                    neggdotstepdir = -g.dot(stepdir)

                    def loss(th):
                        self.sff(th)
                        return self.session.run(self.losses, feed_dict=feed)

                    surr_prev, kl_prev, ent_prev = loss(thprev)
                    mean_advant = np.mean(advant_n)
                    theta = linesearch(loss, thprev, fullstep, neggdotstepdir)
                    self.sff(theta)

                    surrafter, kloldnew, entnew = self.session.run(self.losses, feed_dict=feed)

                    stats = {}

                    numeptotal += len(episoderewards)

                    stats["average_episode_std"] = average_episode_std
                    stats["sum steps of episodes"] = sample_data["sum_episode_steps"]
                    stats["Total number of episodes"] = numeptotal
                    stats["Average sum of rewards per episode"] = episoderewards.mean()
                    # stats["Entropy"] = entropy
                    # exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                    # stats["Baseline explained"] = exp
                    stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                    stats["KL between old and new distribution"] = kloldnew
                    stats["Surrogate loss"] = surrafter
                    stats["Surrogate loss prev"] = surr_prev
                    stats["entropy"] = ent_prev
                    stats["mean_advant"] = mean_advant
                    log_data = [average_episode_std, len(episoderewards), numeptotal, episoderewards.mean(), kloldnew, surrafter, surr_prev,
                                surrafter - surr_prev,
                                ent_prev, mean_advant]
                    self.logger.log_row(log_data)
                    print episoderewards.mean()
                    # for k, v in stats.iteritems():
                    #     print(k + ": " + " " * (40 - len(k)) + str(v))
                        # if entropy != entropy:
                        #     exit(-1)
                        # if exp > 0.95:
                        #     self.train = False
            if i%10==0:
                self.save_model("iter_mountaincar" + str(i))
            i += 1

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
            self.saver.restore(self.session, model_name)
        except:
            print "load model %s fail" % (model_name)
