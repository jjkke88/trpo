from utils import *
from dealImage import *
from logger.logger import Logger
import krylov
import numpy as np
import random
import tensorflow as tf
import time

import prettytensor as pt

from storage.storage import Storage
import parameters as pms
from distribution.diagonal_category import DiagonalCategory
from baseline.baseline_lstsq import Baseline
import gym
from environment import Environment

class TRPOAgent(object):
    def __init__(self):
        self.env = env = Environment(gym.make(pms.environment_name))
        # if not isinstance(env.observation_space, Box) or \
        #    not isinstance(env.action_space, Discrete):
        #     print("Incompatible spaces.")
        #     exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        self.distribution = DiagonalCategory()
        self.session = tf.Session()
        self.baseline = Baseline()
        self.end_count = 0
        self.paths = []
        self.train = True
        self.storage = Storage(self, self.env, self.baseline)
        self.init_network()
        if pms.train_flag:
            self.init_logger()

    def init_logger(self):
        head = ["average_episode_std", "total number of episodes", "Average sum of rewards per episode",
                "KL between old and new distribution", "Surrogate loss", "Surrogate loss prev", "ds", "entropy",
                "mean_advant", "sum_episode_steps"]
        self.logger = Logger(head)

    def init_network(self):
        self.obs = obs = tf.placeholder(
            dtype, shape=[None, self.env.observation_space.shape[0]], name="obs")
        self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, self.env.action_space.n],
                                                              name="oldaction_dist")

        # Create neural network.
        action_dist_n, _ = (pt.wrap(self.obs).
                            fully_connected(32, activation_fn=tf.nn.relu).
                            fully_connected(32, activation_fn=tf.nn.relu).
                            softmax_classifier(self.env.action_space.n))
        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(obs)[0]
        ratio_n = self.distribution.likelihood_ratio_sym(action, action_dist_n, oldaction_dist)
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        kl = self.distribution.kl_sym(oldaction_dist, action_dist_n)
        ent = self.distribution.entropy(action_dist_n)

        self.losses = [surr, kl, ent]

        var_list = tf.trainable_variables()
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            action_dist_n) * tf.log(tf.stop_gradient(action_dist_n + eps) / (action_dist_n + eps))) / Nf
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
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.session.run(tf.initialize_all_variables())

        # self.load_model(pms.checkpoint_file)

    def get_samples(self, path_number):
        for i in range(path_number):
            self.storage.get_single_path()

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs})

        if self.train:
            action = int(cat_sample(action_dist_n)[0])
        else:
            action = int(np.argmax(action_dist_n))
        return action, action_dist_n, np.squeeze(obs)

    def learn(self):
        start_time = time.time()
        numeptotal = 0
        for iteration in range(pms.max_iter_number):
            # Generating paths.
            print("Rollout")
            self.get_samples(pms.paths_number)
            paths = self.storage.get_paths()  # get_paths
            # Computing returns and estimating advantage function.

            sample_data = self.storage.process_paths(paths)
            # shape = sample_data["observations"].shape
            # vis_square(np.reshape(sample_data["observations"],(shape[0], shape[2], shape[3]))[1:10])
            feed = {self.obs: sample_data["observations"],
                    self.action: sample_data["actions"],
                    self.advant: sample_data["advantages"],
                    self.oldaction_dist: sample_data["agent_infos"]}

            print "\n********** Iteration %i ************" % iteration
            if self.train:
                thprev = self.gf()
                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + pms.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = krylov.cg(fisher_vector_product, g)
                shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
                fullstep = stepdir * np.sqrt(2.0 * pms.max_kl / shs)
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses, feed_dict=feed)

                surr_prev, kl_prev, entropy = loss(thprev)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)

                stats = {}
                episoderewards = np.sum(sample_data["rewards"])
                numeptotal += len(sample_data["rewards"])
                mean_advant = np.mean(sample_data["advantages"])
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = np.mean(sample_data["rewards"])
                # stats["Entropy"] = entropy
                # exp = explained_variance(np.array(sample_data[""]), np.array(returns_n))
                # stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                stats['Sum episode steps'] = sample_data["sum_episode_steps"]
                log_data = [0, numeptotal, episoderewards.mean(), kloldnew, surrafter, surr_prev,
                            surrafter - surr_prev,
                            entropy, mean_advant, sample_data["sum_episode_steps"]]
                if pms.train_flag:
                    self.logger.log_row(log_data)
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                self.save_model("iter" + str(iteration))

    def test(self, model_name="checkpoint/checkpoint"):
        self.load_model(model_name)
        self.success_number = 0.0
        test_iter_number = 50
        for i in range(test_iter_number):
            self.get_samples(1)
            paths = self.storage.get_paths()
            sample_data = self.storage.process_paths(paths)
            max_reward = np.max(sample_data["rewards"])
            # print "step number%d"%(len(sample_data["rewards"]))
            # print "max reward%f" % (max_reward)
            # print "everage reward%f"%(np.mean(sample_data["rewards"]))
            # print "\n"
            if max_reward > 0:
                self.success_number += 1.0
        print "success_rate%f" % (self.success_number / float(test_iter_number))

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/" + model_name + ".ckpt")

    def load_model(self, model_name="checkpoint/checkpoint"):
        try:
            self.saver.restore(self.session, model_name)
        except:
            print "load model %s fail" % (model_name)
