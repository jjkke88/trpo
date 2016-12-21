import os
import gym
from environment import Environment
from agent.agent_continous_rnn import TRPOAgent
from parameters import pms

if not os.path.isdir("./checkpoint"):
    os.makedirs("./checkpoint")
if not os.path.isdir("./log"):
    os.makedirs("./log")
env = Environment(gym.make(pms.environment_name))
agent = TRPOAgent(env)

if pms.train_flag:
    agent.learn()
else:
    agent.test(pms.checkpoint_file)
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
