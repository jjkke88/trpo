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
            rewards.append(res[1])
            episode_steps += 1
            if res[2]:
                break
        path = {"obs": np.concatenate(np.expand_dims(self.obs, 0)),
                "action_dists": np.concatenate(action_dists),
                "rewards": np.array(rewards),
                "actions": np.array(actions)}
        self.paths.append(path)
        # self.agent.prev_action *= 0.0
        # self.agent.prev_obs *= 0.0
        return path

    def get_paths(self):
        paths = self.paths
        self.paths = []
        return paths

    def process_paths(self, paths):
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = discount(path["rewards"], pms.gamma)
            path["advant"] = path["returns"] - path["baseline"]

        # Updating policy.
        action_dist_n = np.concatenate([path["action_dists"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        baseline_n = np.concatenate([path["baseline"] for path in paths])
        returns_n = np.concatenate([path["returns"] for path in paths])

        # Standardize the advantage function to have mean=0 and std=1.
        advant_n = np.concatenate([path["advant"] for path in paths])
        advant_n -= advant_n.mean()

        # Computing baseline function for next iter.

        advant_n /= (advant_n.std() + 1e-8)
        self.baseline.fit(paths)
        samples_data = dict(
            observations=obs_n,
            actions=action_n,
            rewards=rewards,
            advantages=advant_n,
            agent_infos=action_dist_n,
            paths=paths,
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