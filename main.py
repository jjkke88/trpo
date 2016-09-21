import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import tempfile
import sys
from environment import Environment
from agent import TRPOAgent
import parameters as pms

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)
if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = pms.environment_name
env = gym.make(task)
# env.monitor.start(training_dir)

env = Environment(env)

agent = TRPOAgent(env)

if pms.train_flag:
    agent.learn()
else:
    agent.test(pms.checkpoint_file)
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
