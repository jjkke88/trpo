import os
import logging
import gym
import tensorflow as tf
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import tempfile
import sys
from environment import Environment
from agent.agent_continous_parallel_storage import TRPOAgent
from parameters import pms

if not os.path.isdir("./checkpoint"):
    os.makedirs("./checkpoint")
if not os.path.isdir("./log"):
    os.makedirs("./log")

# ps_hosts = ["127.0.0.1:2233"]
# cluster = tf.train.ClusterSpec({"ps": ps_hosts})
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0 / 3.0)
# server = tf.train.Server(cluster ,
#                          job_name="ps" ,
#                          task_index=0 ,
#                          config=tf.ConfigProto(gpu_options=gpu_options))
# server.join()

env = Environment(gym.make(pms.environment_name))
agent = TRPOAgent(env)



if pms.train_flag:
    agent.learn()
else:
    agent.test(pms.checkpoint_file)
# env.monitor.close()
# gym.upload(training_dir,
#            algorithm_id='trpo_ff')
