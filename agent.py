
from utils import *
import numpy as np
import random
import tensorflow as tf
import time

import prettytensor as pt

from storage import Storage
import parameters as pms

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
        # Create mean network.
        self.action_dist_mean_n = (pt.wrap(self.obs).
                            fully_connected(64, activation_fn=tf.nn.relu).
                            fully_connected(pms.action_shape))
        # Create std network.
        self.action_dist_std_n = action_dist_std_n = (pt.wrap(self.obs).
                            fully_connected(64, activation_fn=tf.nn.relu).
                            fully_connected(pms.action_shape))
        self.action_dist_std_n = tf.clip_by_value(self.action_dist_std_n, 1e-6 ,100)
        dist = tf.contrib.distributions.Normal(mu=self.action_dist_mean_n, sigma=self.action_dist_std_n)
        self.action_dist_n = dist.pdf(self.action_n)
        self.N = N = tf.shape(obs)[0]
        self.dist_mean = self.action_dist_n
        var_list = tf.trainable_variables()
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist_n = oldaction_dist = tf.placeholder(dtype, shape=[None, pms.action_shape],
                                                              name="oldaction_dist")
        self.old_dist_r = oldaction_dist
        eps = 1e-6

        # function slice_2d choose the probability of action, for action is int, slice is needed, for
        # continous condition, slice is unneeded.
        self.p_n = p_n = self.action_dist_n
        self.oldp_n = oldp_n = self.oldaction_dist_n
        self.ratio_n = ratio_n = p_n / oldp_n
        Nf = tf.cast(N, dtype)

        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        self.kl = kl = tf.reduce_sum(oldp_n * tf.log((oldp_n + eps) / (p_n + eps))) / Nf
        ent = tf.reduce_sum(-p_n * tf.log(p_n + eps)) / Nf

        self.losses = [surr, kl, ent]


        # get g
        self.pg = flatgrad(surr, var_list)
        # get A
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            p_n) * tf.log(tf.stop_gradient(p_n + eps) / (p_n + eps))) / Nf
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
        self.vf = VF(self.session)
        self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.train.SummaryWriter("log", self.session.graph)
        # self.load_model()

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)

        action_dist_mean_n, action_dist_std_n = self.session.run([self.action_dist_mean_n, self.action_dist_std_n], {self.obs: obs})
        action = np.clip(np.random.normal(loc=action_dist_mean_n[0], scale=action_dist_std_n[0], size=action_dist_mean_n[0].shape), pms.min_a, pms.max_a)
        action_dist_n = self.session.run(self.action_dist_n, feed_dict={self.obs:obs, self.action_n:[action]})
        return action, action_dist_n, np.squeeze(obs)

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

            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], pms.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.advant: advant_n,
                    self.oldaction_dist_n: action_dist_n}

            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            # if episoderewards.mean() > 1.1 * self.env._env.spec.reward_threshold:
            #     self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                self.vf.fit(paths)
                thprev = self.gf() # get theta_old

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                g = self.session.run(self.pg, feed_dict=feed)
                # g_input = tf.placeholder(dtype="float32",shape=None,name="g")
                # summary_g_op = tf.scalar_summary('g', g_input)
                # self.writer.add_summary(self.session.run(summary_g_op, feed_dict={g_input:np.mean(g)}))
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir)) # theta
                lm = np.sqrt(shs / pms.max_kl)
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config.max_kl:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                # if entropy != entropy:
                #     exit(-1)
                if exp > 0.8:
                    self.train = False
                self.save_model("iter"+str(i))
            i += 1

    def test(self, model_name):
        self.load_model(model_name)
        self.storage.get_paths()

    def save_model(self, model_name):
        self.saver.save(self.session, "checkpoint/"+model_name+".ckpt")

    def load_model(self, model_name):
        try:
            self.saver.restore(self.session, model_name)
        except:
            print "load model %s fail"%(model_name)

