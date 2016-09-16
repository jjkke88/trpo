from utils import *
import parameters as pms
from baseline.baseline import Baseline

class Storage(object):
    def __init__(self, agent, env):
        self.paths = []
        self.env = env
        self.agent = agent
        self.baseline = Baseline()

    def get_paths(self):
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
        o = self.env.reset()
        path_sum_length = 0
        if pms.render:
            self.env.render()
        while path_sum_length<pms.timesteps_per_batch:
            o = self.env.reset()
            path_length = 0
            while path_length < pms.max_path_length:
                a, agent_info = self.agent.get_action(o)

                next_o, reward, terminal, env_info = self.env.step(a)
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
            np.save("datalog/action.npy", actions)
            paths.append(dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                agent_infos=np.concatenate(agent_infos),
                env_infos=np.concatenate(env_infos),
            ))

        return paths

    def process_paths(self, paths):
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = np.concatenate(path["rewards"]) + \
                     pms.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = discount(
                deltas, pms.discount * pms.gae_lambda)
            path["returns"] = discount(path["rewards"], pms.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = np.concatenate([path["env_infos"] for path in paths])
        agent_infos = np.concatenate([path["agent_infos"] for path in paths])

        if pms.center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

        if pms.positive_adv:
            advantages = (advantages - np.min(advantages)) + 1e-8

        # average_discounted_return = \
        #     np.mean([path["returns"][0] for path in paths])
        #
        # undiscounted_returns = [sum(path["rewards"]) for path in paths]

        # ent = np.mean(self.policy.distribution.entropy(agent_infos))

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