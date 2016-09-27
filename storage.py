import cv2
from utils import *
import parameters as pms
import threading


class Storage(object):
    def __init__(self, agent, env, baseline):
        self.paths = []
        self.env = env
        self.agent = agent
<<<<<<< HEAD
        self.obs = []
        self.obs_origin = []
        self.baseline = baseline

    def get_single_path(self):
        self.obs_origin, self.obs, actions, rewards, action_dists = [], [], [], [], []
        ob = self.env.reset()
        ob = self.env.render('rgb_array')
        # self.agent.prev_action *= 0.0
        # self.agent.prev_obs *= 0.0
        episode_steps = 0
        for _ in xrange(pms.max_pathlength):
            self.obs_origin.append(ob)
            deal_ob = self.deal_image(ob)
            action, action_dist, ob = self.agent.act(deal_ob)
            self.obs.append(deal_ob)
            actions.append(action)
            action_dists.append(action_dist)
            res = self.env.step(action) # res
            if pms.render:
                self.env.render()
            ob = res[0]
            ob = self.env.render('rgb_array')
            rewards.append([res[1]])
            episode_steps += 1
            if res[2]:
                break
        path = dict(
            observations=np.concatenate(np.expand_dims(self.obs, 0)),
            agent_infos=np.concatenate(action_dists),
            rewards=np.array(rewards),
            actions=np.array(actions),
            episode_steps=episode_steps
        )
        self.paths.append(path)
        # self.agent.prev_action *= 0.0
        # self.agent.prev_obs *= 0.0
        return path
=======
        self.baseline = baseline

    def get_single_path(self):
        """
        :param:observations:obs list
        :param:actions:action list
        :param:rewards:reward list
        :param:agent_infos: mean+log_std dictlist
        :param:env_infos: no use, just information about environment
        :return: a path, list
        """
        # if pms.record_movie:
        #     outdir = 'log/trpo'
        #     self.env.monitor.start(outdir , force=True)
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        o = self.env.reset()
        path_sum_length = 0
        if pms.render:
            self.env.render()
        o = self.env.reset()
        path_length = 0
        while path_length < pms.max_path_length:
            a , agent_info = self.agent.get_action(o)
            next_o , reward , terminal , env_info = self.env.step(2*np.tanh(a))
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
                self.env.render()
        self.paths.append(dict(
            observations=np.array(observations) ,
            actions=np.array(actions) ,
            rewards=np.array(rewards) ,
            agent_infos=np.concatenate(agent_infos) ,
            env_infos=np.concatenate(env_infos) ,
        ))
>>>>>>> trpo-continues-action

    def get_paths(self):
        paths = self.paths
        self.paths = []
        return paths

    def process_paths(self, paths):
<<<<<<< HEAD
        # baselines = []
        # returns = []
        sum_episode_steps = 0
        for path in paths:
            sum_episode_steps += path['episode_steps']
=======
        baselines = []
        returns = []
        for path in paths:
>>>>>>> trpo-continues-action
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = np.concatenate(path["rewards"]) + \
                     pms.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = discount(
                deltas, pms.discount * pms.gae_lambda)
            path["returns"] = np.concatenate(discount(path["rewards"], pms.discount))
<<<<<<< HEAD
            # baselines.append(path_baselines[:-1])
            # returns.append(path["returns"])

        # Updating policy.
        action_dist_n = np.concatenate([path["agent_infos"] for path in paths])
        obs_n = np.concatenate([path["observations"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
=======
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = np.concatenate([path["env_infos"] for path in paths])
        agent_infos = np.concatenate([path["agent_infos"] for path in paths])
>>>>>>> trpo-continues-action

        if pms.center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

<<<<<<< HEAD
        self.baseline.fit(paths)

        samples_data = dict(
            observations=obs_n,
            actions=action_n,
            rewards=rewards,
            advantages=advantages,
            agent_infos=action_dist_n,
            paths=paths,
            sum_episode_steps=sum_episode_steps
        )
        return samples_data

    def deal_image(self, image):
        index = len(self.obs_origin)
        image_end = []
        if index<pms.history_number:
            image_end = self.obs_origin[0:index]
            for i in range(pms.history_number-index):
                image_end.append(image)
        else:
            image_end = self.obs_origin[index-pms.history_number:index]

        image_end = np.concatenate(image_end)
        # image_end = image_end.reshape((pms.obs_height, pms.obs_width, pms.history_number))
        obs = cv2.resize(cv2.cvtColor(image_end, cv2.COLOR_RGB2GRAY) / 255., (pms.obs_height, pms.obs_width))
        return np.expand_dims(obs, 0)
=======
        # for some unknown reaseon, it can not be used
        # if pms.positive_adv:
        #     advantages = (advantages - np.min(advantages)) + 1e-8

        # average_discounted_return = \
        #     np.mean([path["returns"][0] for path in paths])
        #
        # undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.mean(self.agent.distribution.entropy(agent_infos))

        # ev = self.explained_variance_1d(
        #     np.concatenate(baselines),
        #     np.concatenate(returns)
        # )
        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
            entropy=ent
        )

        self.baseline.fit(paths)
        return samples_data

    def explained_variance_1d(ypred, y):
        assert y.ndim == 1 and ypred.ndim == 1
        vary = np.var(y)
        if np.isclose(vary, 0):
            if np.var(ypred) > 0:
                return 0
            else:
                return 1
        if abs(1 - np.var(y - ypred) / (vary + 1e-8)) > 1e5:
            import ipdb; ipdb.set_trace()
        return 1 - np.var(y - ypred) / (vary + 1e-8)


class Rollout(threading.Thread):
    def __init__(self , thread_number , storage):
        super(Rollout , self).__init__()
        self.thread_number = thread_number
        self.storage = storage

    def run(self):
        self.storage.get_single_path()

>>>>>>> trpo-continues-action
