import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import tempfile
import sys
from environment import Environment
from agent.agent_continous_rnn import TRPOAgent
import parameters as pms

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

agent = TRPOAgent()

if pms.train_flag:
    agent.learn()
else:
    agent.test(pms.checkpoint_file)
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
