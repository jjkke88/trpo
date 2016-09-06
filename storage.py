import cv2
from utils import *
import parameters as pms


class Storage(object):
    def __init__(self, agent, env):
        self.paths = []
        self.env = env
        self.agent = agent
        self.obs = []
        self.obs_origin = []

    def get_paths(self):
        paths = []
        timesteps_sofar = 0
        while timesteps_sofar < pms.timesteps_per_batch:
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
                rewards.append(res[1])
                episode_steps += 1
                if res[2]:
                    break
            path = {"obs": np.concatenate(np.expand_dims(self.obs, 0)),
                    "action_dists": np.concatenate(action_dists),
                    "rewards": np.array(rewards),
                    "actions": np.array(actions)}
            paths.append(path)
            # self.agent.prev_action *= 0.0
            # self.agent.prev_obs *= 0.0
            timesteps_sofar += episode_steps
        return paths

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