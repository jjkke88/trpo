from utils import *
from parameters import pms


class Storage(object):
    def __init__(self, agent, env, baseline):
        self.paths = []
        self.env = env
        self.agent = agent
        self.baseline = baseline

    def get_single_path(self):
        self.obs, actions, rewards, action_dists = [], [], [], []
        ob = self.env.reset()
        episode_steps = 0
        for _ in xrange(pms.max_path_length):
            action, action_dist, ob = self.agent.act(ob)
            self.obs.append(ob)
            actions.append(action)
            action_dists.append(action_dist)
            res = self.env.step(action)  # res
            if pms.render:
                self.env.render()
            ob = res[0]
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

    def get_paths(self):
        paths = self.paths
        self.paths = []
        return paths

    def process_paths(self, paths):
        # baselines = []
        # returns = []
        sum_episode_steps = 0
        for path in paths:
            sum_episode_steps += path['episode_steps']
            path_baselines = np.append(self.baseline.predict(path), 0)
            # deltas = r_t+V(s_{t+1})-V(s_t)
            deltas = np.concatenate(path["rewards"]) + \
                     pms.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = discount(
                deltas, pms.discount * pms.gae_lambda)
            path["returns"] = np.concatenate(discount(path["rewards"], pms.discount))
            # baselines.append(path_baselines[:-1])
            # returns.append(path["returns"])

        # Updating policy.
        action_dist_n = np.concatenate([path["agent_infos"] for path in paths])
        obs_n = np.concatenate([path["observations"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        if pms.center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

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
