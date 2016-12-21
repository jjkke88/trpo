from utils import *
import numpy as np
import tensorflow as tf
from network.network_continous_image import NetworkContinousImage
from parameters import pms

import multiprocessing
import krylov
from baseline.baseline_zeros import Baseline
from distribution.diagonal_gaussian import DiagonalGaussian
import time
import math
from logger.logger import Logger

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


"""
class for continoust action space in multi process
"""
class TRPOAgentParallelImage(multiprocessing.Process):


    def __init__(self , observation_space , action_space , task_q , result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space = observation_space
        self.action_space = action_space
        self.args = pms
        self.baseline = Baseline()
        self.distribution = DiagonalGaussian(pms.action_shape)
        self.init_logger()

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
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)
        self.net = NetworkContinousImage("network_continous_image")
        if pms.min_std is not None:
            log_std_var = tf.maximum(self.net.action_dist_logstds_n, np.log(pms.min_std))
        self.action_dist_stds_n = tf.exp(log_std_var)
        self.old_dist_info_vars = dict(mean=self.net.old_dist_means_n, log_std=self.net.old_dist_logstds_n)
        self.new_dist_info_vars = dict(mean=self.net.action_dist_means_n, log_std=self.net.action_dist_logstds_n)
        self.likehood_action_dist = self.distribution.log_likelihood_sym(self.net.action_n, self.new_dist_info_vars)
        self.ratio_n = self.distribution.likelihood_ratio_sym(self.net.action_n, self.new_dist_info_vars,
                                                              self.old_dist_info_vars)
        surr = -tf.reduce_sum(self.ratio_n * self.net.advant)  # Surrogate loss
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
        self.saver = tf.train.Saver(max_to_keep=5)

    def init_logger(self):
        head = ["factor", "rewards", "std"]
        self.logger = Logger(head)

    def run(self):
        self.init_network()
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.gf())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.args.max_kl = paths[1]
                if paths[2] == 1:
                    print "saving checkpoint..."
                    self.save_model(pms.environment_name + "-" + str(paths[3]))
                self.task_q.task_done()
            else:
                stats , theta, thprev = self.learn(paths)
                self.sff(theta)
                self.task_q.task_done()
                self.result_q.put((stats, theta, thprev))
        return

    def learn(self, paths, parallel=False, linear_search=False):
        start_time = time.time()
        sample_data = self.process_paths(paths)
        agent_infos = sample_data["agent_infos"]
        obs_all = sample_data["observations"]
        action_all = sample_data["actions"]
        advant_all = sample_data["advantages"]
        n_samples = len(obs_all)
        batch = int(1/pms.subsample_factor)
        batch_size = int(math.floor(n_samples * pms.subsample_factor))
        accum_fullstep = 0.0
        for iteration in range(batch):
            print "batch: %d, batch_size: %d"%(iteration+1, batch_size)
            inds = np.random.choice(n_samples , batch_size , replace=False)
            obs_n = obs_all[inds]
            action_n = action_all[inds]
            advant_n = advant_all[inds]
            action_dist_means_n = np.array([agent_info["mean"] for agent_info in agent_infos[inds]])
            action_dist_logstds_n = np.array([agent_info["log_std"] for agent_info in agent_infos[inds]])
            feed = {self.net.obs: obs_n ,
                    self.net.advant: advant_n ,
                    self.net.old_dist_means_n: action_dist_means_n ,
                    self.net.old_dist_logstds_n: action_dist_logstds_n ,
                    self.net.action_n: action_n
                    }

            episoderewards = np.array([path["rewards"].sum() for path in paths])
            thprev = self.gf()  # get theta_old

            def fisher_vector_product(p):
                feed[self.flat_tangent] = p
                return self.session.run(self.fvp , feed) + pms.cg_damping * p

            g = self.session.run(self.pg , feed_dict=feed)
            stepdir = krylov.cg(fisher_vector_product , -g , cg_iters=pms.cg_iters)
            shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))  # theta
            # if shs<0, then the nan error would appear
            lm = np.sqrt(shs / pms.max_kl)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.sff(th)
                return self.session.run(self.losses , feed_dict=feed)

            if parallel is True:
                theta = linesearch_parallel(loss , thprev , fullstep , neggdotstepdir / lm)
            else:
                if linear_search:
                    theta = linesearch(loss , thprev , fullstep , neggdotstepdir / lm)
                else:
                    theta = thprev + fullstep
                    if math.isnan(theta.mean()):
                        print shs is None
                        theta = thprev
            accum_fullstep += (theta - thprev)
        theta = thprev + accum_fullstep * pms.subsample_factor
        stats = {}
        stats["sum steps of episodes"] = sample_data["sum_episode_steps"]
        stats["Average sum of rewards per episode"] = episoderewards.mean()
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
        self.logger.log_row([pms.subsample_factor, stats["Average sum of rewards per episode"], self.session.run(self.net.action_dist_logstd_param)[0][0]])
        return stats , theta , thprev

    def process_paths(self, paths):
        sum_episode_steps = 0
        for path in paths:
            sum_episode_steps += path['episode_steps']
            # r_t+V(S_{t+1})-V(S_t) = returns-baseline
            # path_baselines = np.append(self.baseline.predict(path) , 0)
            # # r_t+V(S_{t+1})-V(S_t) = returns-baseline
            # path["advantages"] = np.concatenate(path["rewards"]) + \
            #          pms.discount * path_baselines[1:] - \
            #          path_baselines[:-1]
            # path["returns"] = np.concatenate(discount(path["rewards"], pms.discount))
            path_baselines = np.append(self.baseline.predict(path) , 0)
            deltas = np.concatenate(path["rewards"]) + \
                     pms.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = discount(
                deltas , pms.discount * pms.gae_lambda)
            path["returns"] = np.concatenate(discount(path["rewards"] , pms.discount))
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = np.concatenate([path["env_infos"] for path in paths])
        agent_infos = np.concatenate([path["agent_infos"] for path in paths])
        if pms.center_adv:
            advantages -= np.mean(advantages)
            advantages /= (advantages.std() + 1e-8)

        # for some unknown reaseon, it can not be used
        # if pms.positive_adv:
        #     advantages = (advantages - np.min(advantages)) + 1e-8

        # average_discounted_return = \
        #     np.mean([path["returns"][0] for path in paths])
        #
        # undiscounted_returns = [sum(path["rewards"]) for path in paths]


        # ev = self.explained_variance_1d(
        #     np.concatenate(baselines),
        #     np.concatenate(returns)
        # )
        samples_data = dict(
            observations=observations ,
            actions=actions ,
            rewards=rewards ,
            advantages=advantages ,
            env_infos=env_infos ,
            agent_infos=agent_infos ,
            paths=paths ,
            sum_episode_steps=sum_episode_steps
        )
        self.baseline.fit(paths)
        return samples_data

    def save_model(self , model_name):
        self.saver.save(self.session , "checkpoint/" + model_name + ".ckpt")

    def load_model(self , model_name):
        try:
            if model_name is not None:
                self.saver.restore(self.session , model_name)
            else:
                self.saver.restore(self.session , tf.train.latest_checkpoint(pms.checkpoint_dir))
        except:
            print "load model %s fail" % (model_name)