import numpy as np
import tensorflow as tf
import random
import scipy.signal
import prettytensor as pt
import parameters as pms
import threading
from tensorflow.contrib.layers.python.layers import initializers

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
    np.save("datalog/action.npy" , actions)
    paths.append(dict(
        observations=np.array(observations) ,
        actions=np.array(actions) ,
        rewards=np.array(rewards) ,
        agent_infos=np.concatenate(agent_infos) ,
        env_infos=np.concatenate(env_infos) ,
    ))

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
    fval, old_kl = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.3**np.arange(max_backtracks)):
        xnew = x - stepfrac * fullstep
        newfval, new_kl = f(xnew)
        # actual_improve = newfval - fval # minimize target object
        # expected_improve = expected_improve_rate * stepfrac
        # ratio = actual_improve / expected_improve
        # if ratio > accept_ratio and actual_improve > 0:
        #     return xnew
        if newfval<fval and new_kl<=pms.max_kl:
            return xnew
    return x


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in xrange(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
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


def log_likelihood_sym(x_var, dist_info_means, dist_info_logstds):
    means = dist_info_means
    log_stds = dist_info_logstds
    zs = (x_var - means) / tf.exp(log_stds)
    return - tf.reduce_sum(log_stds) - \
           0.5 * tf.reduce_sum(tf.square(zs)) - \
           0.5 * means.get_shape()[-1].value * np.log(2 * np.pi)

def likelihood_ratio_sym(x_var, old_dist_means, old_dist_logstds, new_dist_means, new_dist_logstds):
    logli_new = log_likelihood_sym(x_var, new_dist_means, new_dist_logstds)
    logli_old = log_likelihood_sym(x_var, old_dist_means, old_dist_logstds)
    return tf.exp(logli_new - logli_old)



def conv2d(x,
        output_dim,
        kernel_size,
        stride,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation_fn=tf.nn.relu,
        data_format='NHWC',
        padding='VALID',
        name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

        if activation_fn != None:
            out = activation_fn(out)

    return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        else:
            return out, w, b
