import numpy as np
import tensorflow as tf
import random
import scipy.signal
import prettytensor as pt
import parameters as pms
<<<<<<< HEAD
=======
import threading
from tensorflow.contrib.layers.python.layers import initializers
>>>>>>> trpo-continues-action

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

dtype = tf.float32

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# class VF(object):
#     coeffs = None
#
#     def __init__(self, session, type="origin"):
#         self.net = None
#         self.session = session
#         self.type = type
#
#     def create_net(self, shape):
#         self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
#         self.y = tf.placeholder(tf.float32, shape=[None], name="y")
#         self.net = (pt.wrap(self.x).
#                     fully_connected(64, activation_fn=tf.nn.relu).
#                     fully_connected(64, activation_fn=tf.nn.relu).
#                     fully_connected(1))
#         self.net = tf.reshape(self.net, (-1, ))
#         l2 = (self.net - self.y) * (self.net - self.y)
#         self.train = tf.train.AdamOptimizer().minimize(l2)
#         self.session.run(tf.initialize_all_variables())
#
#     def _features(self, path):
#         if self.type == "origin":
#             o = path["obs"].astype('float32')
#             o = o.reshape(o.shape[0], -1)
#             act = path["action_dists"].astype('float32')
#             l = len(path["rewards"])
#             al = np.arange(l).reshape(-1, 1) / 10.0
#             # up to down stack obs, action_dst, al, np.ones((l,1))
#             ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
#             # ret = np.concatenate([o], axis=1)
#         elif self.type == "rgb_array":
#             ret = path["obs"].astype('float32')
#         return ret
#
#     def fit(self, paths):
#         featmat = np.concatenate([self._features(path) for path in paths])
#         if self.net is None:
#             self.create_net(featmat.shape[1])
#         returns = np.concatenate([path["returns"] for path in paths])
#         for _ in range(50):
#             self.session.run(self.train, {self.x: featmat, self.y: returns})
#
#     def predict(self, path):
#         if self.net is None:
#             return np.zeros(len(path["rewards"]))
#         else:
#             ret = self.session.run(self.net, {self.x: self._features(path)})
#             return np.reshape(ret, (ret.shape[0], ))

def rollout(env, agent, paths):
    """
        :param:observations:obs list
        :param:actions:action list
        :param:rewards:reward list
        :param:agent_infos: mean+log_std dictlist
        :param:env_infos: no use, just information about environment
        :return: a path, list
        """
    paths = []
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    path_sum_length = 0
    if pms.render:
        env.render()
    o = env.reset()
    path_length = 0
    while path_length < pms.max_path_length:
        a , agent_info = agent.get_action(o)
        next_o , reward , terminal , env_info = env.step(a)
        observations.append(o)
        rewards.append(np.array([reward]))
        actions.append(a)
        agent_infos.append([agent_info])
        env_infos.append([env_info])
        path_length += 1
        path_sum_length += 1
        if terminal:
            break
        o = next_o
        if pms.render:
            env.render()
<<<<<<< HEAD
            ob = res[0]
            rewards.append(res[1])
            episode_steps += 1
            if res[2]:
                break
        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "action_dists": np.concatenate(action_dists),
                "rewards": np.array(rewards),
                "actions": np.array(actions)}
        paths.append(path)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        timesteps_sofar += episode_steps
    return paths


class VF(object):
    coeffs = None

    def __init__(self, session, type="origin"):
        self.net = None
        self.session = session
        self.type = type

    def create_net(self, shape):
        self.x = tf.placeholder(tf.float32, shape=[None, shape[1], shape[2], shape[3]], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.net = (pt.wrap(self.x).
                    reshape((None, pms.obs_height, pms.obs_width, 1)).
                    conv2d(4, 1).
                    conv2d(4, 1).
                    flatten().
                    fully_connected(32, activation_fn=tf.nn.relu).
                    fully_connected(32, activation_fn=tf.nn.relu).
                    fully_connected(1))
        self.net = tf.reshape(self.net, (-1, ))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        self.session.run(tf.initialize_all_variables())

    def _features(self, path):
        if self.type == "origin":
            o = path["obs"].astype('float32')
            o = o.reshape(o.shape[0], -1)
            act = path["action_dists"].astype('float32')
            l = len(path["rewards"])
            al = np.arange(l).reshape(-1, 1) / 10.0
            # up to down stack obs, action_dst, al, np.ones((l,1))
            ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        elif self.type == "rgb_array":
            ret = path["obs"].astype('float32')
        return ret


    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape)
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(50):
            self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"])) 
        else:
            ret = self.session.run(self.net, {self.x: self._features(path)})
            return np.reshape(ret, (ret.shape[0], ))

=======
    np.save("datalog/action.npy" , actions)
    paths.append(dict(
        observations=np.array(observations) ,
        actions=np.array(actions) ,
        rewards=np.array(rewards) ,
        agent_infos=np.concatenate(agent_infos) ,
        env_infos=np.concatenate(env_infos) ,
    ))
>>>>>>> trpo-continues-action

def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(xrange(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(v))])
                         for (grad, v) in zip( grads, var_list)])

# set theta
class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        assigns = []
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(
                tf.assign(
                    v,
                    tf.reshape(
                        theta[
                            start:start +
                            size],
                        shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})

# get theta
class GetFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


def slice_2d(x, inds0, inds1):
    # assume that a path have 1000 vector, then ncols=action dims, inds0=1000,inds1=
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
<<<<<<< HEAD
    fval, old_kl, entropy = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.3**np.arange(max_backtracks)):
        xnew = x - stepfrac * fullstep
        newfval, new_kl, new_ent= f(xnew)
=======
    fval, old_kl = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.3**np.arange(max_backtracks)):
        xnew = x - stepfrac * fullstep
        newfval, new_kl = f(xnew)
>>>>>>> trpo-continues-action
        # actual_improve = newfval - fval # minimize target object
        # expected_improve = expected_improve_rate * stepfrac
        # ratio = actual_improve / expected_improve
        # if ratio > accept_ratio and actual_improve > 0:
        #     return xnew
        if newfval<fval and new_kl<=pms.max_kl:
            return xnew
    return x


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def countMatrixMultiply(matrix):
    result_end = []
    for j in matrix:
        result = 1.0
        for i in j:
            result *= i
        result_end.append(result)
    return np.array(result_end)

def kl_sym(old_dist_means, old_dist_logstds, new_dist_means, new_dist_logstds):
    old_std = tf.exp(old_dist_logstds)
    new_std = tf.exp(new_dist_logstds)
    # means: (N*A)
    # std: (N*A)
    # formula:
    # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
    # ln(\sigma_2/\sigma_1)
    numerator = tf.square(old_dist_means - new_dist_means) + \
                tf.square(old_std) - tf.square(new_std)
    denominator = 2 * tf.square(new_std) + 1e-8
    return tf.reduce_sum(
        numerator / denominator + new_dist_logstds - old_dist_logstds)

def kl_sym_gradient(old_dist_means, old_dist_logstds, new_dist_means, new_dist_logstds):
    old_std = tf.exp(old_dist_logstds)
    new_std = tf.exp(new_dist_logstds)
    numerator = tf.square(tf.stop_gradient(new_dist_means) - new_dist_means) + \
                tf.square(tf.stop_gradient(new_std)) - tf.square(new_std)


    denominator = 2 * tf.square(new_std) + 1e-8
    return tf.reduce_sum(
        numerator / denominator + new_dist_logstds - tf.stop_gradient(new_dist_logstds))

