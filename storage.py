from utils import *
import parameters as pms


class Storage(object):
    def __init__(self, agent, env):
        self.paths = []
        self.env = env
        self.agent = agent

    def get_paths(self):
        paths = []
        timesteps_sofar = 0
        while timesteps_sofar < pms.timesteps_per_batch:
            obs, actions, rewards, action_dists = [], [], [], []
            ob = self.env.reset()
            self.agent.prev_action *= 0.0
            self.agent.prev_obs *= 0.0
            episode_steps = 0
            for _ in xrange(pms.max_pathlength):
                action, action_dist, ob = self.agent.act(ob)
                obs.append(ob)
                actions.append(action)
                action_dists.append(action_dist)
                res = self.env.step(action) # res
                self.env.render()
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
            self.agent.prev_action *= 0.0
            self.agent.prev_obs *= 0.0
            timesteps_sofar += episode_steps
        return paths