import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
from space_conversion import SpaceConversionEnv
import tempfile
import sys
from environment import Environment
from agent import TRPOAgent


training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)
if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = "Breakout-v0"
env = gym.make(task)
# env.monitor.start(training_dir)

env = Environment(env)

agent = TRPOAgent(env)
agent.learn()
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
