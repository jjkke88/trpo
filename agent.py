
from utils import *
import numpy as np
import random
import tensorflow as tf
import time

import prettytensor as pt

from storage import Storage
import parameters as pms
import krylov
from distribution.diagonal_gaussian import DiagonalGaussian

class TRPOAgent(object):
    config = dict2(**{
        "timesteps_per_batch": 1000,
        "max_pathlength": 10000,
        "max_kl": 0.01,
        "cg_damping": 0.1,
        "gamma": 0.95,
        "deviation":0.1})

    def __init__(self, env):
        self.env = env
        # if not isinstance(env.observation_space, Box) or \
        #    not isinstance(env.action_space, Discrete):
        #     print("Incompatible spaces.")
        #     exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        print("Action area, high:%f, low%f"%(env.action_space.high, env.action_space.low))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1/3.0)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.end_count = 0
        self.paths = []
        self.train = True
        self.storage = Storage(self, self.env)
        self.init_network()

    def init_network(self):
        self.obs = obs = tf.placeholder(
            dtype, shape=[None, pms.obs_shape], name="obs")
        self.action_n = tf.placeholder(dtype, shape=[None, pms.action_shape], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.old_dist_means_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                               name="oldaction_dist_means")
        self.old_dist_logstds_n = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                 name="oldaction_dist_logstds")
        # Create mean network.
        self.action_dist_means_n = (pt.wrap(self.obs).
                            fully_connected(64, activation_fn=tf.nn.tanh).
                            fully_connected(pms.action_shape))
        # Create std network.
        if pms.use_std_network:
            self.action_dist_logstds_n = (pt.wrap(self.obs).
                                fully_connected(64, activation_fn=tf.nn.tanh).
                                fully_connected(pms.action_shape))
        else:
            self.action_dist_logstds_n = tf.Variable([0.01], trainable=False)
        self.action_dist_stds_n = tf.exp(self.action_dist_logstds_n)

        self.distribution = DiagonalGaussian(pms.action_shape)

        self.N = tf.shape(obs)[0]

        var_list = tf.trainable_variables()

        eps = 1e-6

        # function slice_2d choose the probability of action, for action is int, slice is needed, for
        # continous condition, slice is unneeded.
        Nf = tf.cast(self.N, dtype)
        self.old_dist_info_vars = dict(mean=self.old_dist_means_n, log_std=self.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.action_dist_means_n, log_std=self.action_dist_logstds_n)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.action_n, self.old_dist_info_vars, self.new_dist_info_vars)
        surr = -tf.reduce_mean(self.ratio_n * advant)  # Surrogate loss
        kl = self.distribution.kl_sym(self.old_dist_info_vars, self.new_dist_info_vars)/ Nf
        # ent = tf.reduce_sum(-p_n * tf.log(p_n + eps)) / Nf

        self.losses = [surr, kl]

        # get g
        self.pg = flatgrad(surr, var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = kl_sym_gradient(self.old_dist_means_n, self.old_dist_logstds_n, self.action_dist_means_n, self.action_dist_logstds_n)/ Nf
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
        self.fvp = flatgrad(gvp, var_list) # get kl''*p

        self.gf = GetFlat(self.session, var_list) # get theta from var_list
        self.sff = SetFromFlat(self.session, var_list) # set theta from var_List
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.train.SummaryWriter("log", self.session.graph)
        # self.load_model()

    def get_action(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        action_dist_means_n, action_dist_logstds_n = self.session.run([self.action_dist_means_n, self.action_dist_logstds_n], {self.obs: obs})

        if pms.train_flag:
            rnd = np.random.normal(size=action_dist_means_n[0].shape)
            action = rnd * np.exp(action_dist_logstds_n[0]) + action_dist_means_n[0]
        else:
            action = action_dist_means_n[0]
        action = np.clip(action, pms.min_a, pms.max_a)
        return action, dict(mean=action_dist_means_n, log_std=action_dist_logstds_n)

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = self.storage.get_paths() # get_paths
            # Computing returns and estimating advantage function.
            paths = self.storage.process_paths(paths)

            agent_infos = np.concatenate([path["agent_infos"] for path in paths])
            obs_n = np.concatenate([path["observations"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            advant_n = np.concatenate([path["advantages"] for path in paths])

            n_samples = len(obs_n)
            inds = np.random.choice(
                n_samples, n_samples * pms.subsample_factor, replace=False)
            obs_n = obs_n[inds]
            action_n = action_n[inds]
            advant_n = advant_n[inds]
            action_dist_means_n = agent_infos[inds]["mean"]
            action_dist_logstds_n = agent_infos[inds]["log_std"]
            feed = {self.obs: obs_n,
                    self.advant: advant_n,
                    self.old_dist_means_n: action_dist_means_n,
                    self.old_dist_logstds_n: action_dist_logstds_n,
                    self.action_n:action_n}

            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                # self.vf.fit(paths)
                thprev = self.gf() # get theta_old

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                # g_input = tf.placeholder(dtype="float32",shape=None,name="g")
                # summary_g_op = tf.scalar_summary('g', g_input)
                # self.writer.add_summary(self.session.run(summary_g_op, feed_dict={g_input:np.mean(g)}))
                stepdir = krylov.cg(fisher_vector_product, g, cg_iters=pms.cg_iters)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir)) # theta
                lm = np.sqrt(shs / pms.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew = self.session.run(self.losses, feed_dict=feed)
                if kloldnew > 2.0 * pms.max_kl:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                # stats["Entropy"] = entropy
                # exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                # stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                # if entropy != entropy:
                #     exit(-1)
                # if exp > 0.95:
                #     self.train = False
                self.save_model("iter"+str(i))
            i += 1

    def test(self, model_name):
        self.load_model(model_name)
        for i in range(100):
            self.storage.get_paths()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/"+model_name+".ckpt")

    def load_model(self, model_name):
        try:
            self.saver.restore(self.session, model_name)
        except:
            print "load model %s fail"%(model_name)

