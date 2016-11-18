import os
if not os.path.isdir("./checkpoint"):
    os.makedirs("./checkpoint")
if not os.path.isdir("./log"):
    os.makedirs("./log")

import numpy as np
import tensorflow as tf
import gym
from utils import *
from agent.agent_continous_parallel_storage import TRPOAgentParallel
import argparse
from rollouts import *
import json
from parameters import pms
from storage.storage_continous_parallel import ParallelStorage
from baseline.baseline_lstsq import Baseline

args = pms
args.max_pathlength = gym.spec(args.environment_name).timestep_limit

learner_tasks = multiprocessing.JoinableQueue()
learner_results = multiprocessing.Queue()
learner_env = gym.make(args.environment_name)

learner = TRPOAgentParallel(learner_env.observation_space, learner_env.action_space, learner_tasks, learner_results)
learner.start()
rollouts = ParallelStorage(Baseline())

learner_tasks.put(1)
learner_tasks.join()
starting_weights = learner_results.get()
rollouts.set_policy_weights(starting_weights)

start_time = time.time()
history = {}
history["rollout_time"] = []
history["learn_time"] = []
history["mean_reward"] = []
history["timesteps"] = []

# start it off with a big negative number
last_reward = -1000000
recent_total_reward = 0

if pms.train_flag is True:
    for iteration in xrange(args.max_iter_number):
        # runs a bunch of async processes that collect rollouts
        paths = rollouts.get_paths()
        # Why is the learner in an async process?
        # Well, it turns out tensorflow has an issue: when there's a tf.Session in the main thread
        # and an async process creates another tf.Session, it will freeze up.
        # To solve this, we just make the learner's tf.Session in its own async process,
        # and wait until the learner's done before continuing the main thread.
        learn_start = time.time()
        if iteration%20 == 0:
            learner_tasks.put((2 , args.max_kl, 1, iteration))
        else:
            learner_tasks.put((2, args.max_kl, 0, iteration))
        learner_tasks.put(paths)
        learner_tasks.join()
        stats , theta , thprev = learner_results.get()
        learn_time = (time.time() - learn_start) / 60.0
        print
        print "-------- Iteration %d ----------" % iteration
        # print "Total time: %.2f mins" % ((time.time() - start_time) / 60.0)
        #
        # history["rollout_time"].append(rollout_time)
        # history["learn_time"].append(learn_time)
        # history["mean_reward"].append(mean_reward)
        # history["timesteps"].append(args.timesteps_per_batch)
        for k , v in stats.iteritems():
            print(k + ": " + " " * (40 - len(k)) + str(v))
        recent_total_reward = stats["Average sum of rewards per episode"]

        if args.decay_method == "adaptive":
            if iteration % 10 == 0:
                if recent_total_reward < last_reward:
                    print "Policy is not improving. Decrease KL and increase steps."
                    if args.max_kl > 0.001:
                        args.max_kl -= args.kl_adapt
                else:
                    print "Policy is improving. Increase KL and decrease steps."
                    if args.max_kl < 0.01:
                        args.max_kl += args.kl_adapt
                last_reward = recent_total_reward
                recent_total_reward = 0

        if args.decay_method == "linear":
            if args.max_kl > 0.001:
                args.max_kl -= args.kl_adapt

        if args.decay_method == "exponential":
            if args.max_kl > 0.001:
                args.max_kl *= args.kl_adapt

        # print "Current steps is " + str(args.timesteps_per_batch) + " and KL is " + str(args.max_kl)
        #
        # if iteration % 100 == 0:
        #     with open("%s-%s-%f-%f" % (args.task, args.decay_method, args.kl_adapt, args.timestep_adapt), "w") as outfile:
        #         json.dump(history,outfile)
        rollouts.set_policy_weights(theta)
else:
    from agent.agent_continous import TRPOAgent
    from environment import Environment
    env = Environment(gym.make(pms.environment_name))
    agent = TRPOAgent(env)
    agent.test(pms.checkpoint_file)


rollouts.end()
