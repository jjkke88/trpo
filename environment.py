"""
`SpaceConversionEnv` acts as a wrapper on
any environment. It allows to convert some action spaces, and observation spaces to others.
"""

import numpy as np
from gym.spaces import Discrete, Box, Tuple
from gym import Env
import cv2
import parameters as pms


class Environment(Env):

    def __init__(self, env, type="origin"):
        self.env = env
        self.type = type

    def step(self, action, **kwargs):
        self._observation, reward, done, info = self.env.step(action)
        return self.observation, reward, done, {}

    def reset(self, **kwargs):
        self._observation = self.env.reset()
        return self.observation

    def render(self):
        self.env.render()

    @property
    def observation(self):
        if self.type == "origin":
            return self._observation
        elif self.type == "gray_image":
            return cv2.resize(cv2.cvtColor(self._observation, cv2.COLOR_RGB2GRAY)/255., pms.dims)

    @property
    def action_space(self):
        return self.env.action_space


    @property
    def observation_space(self):
        if self.type == "origin":
            return self.env.observation_space
        else:
            return pms.dims

    # @property
    # def obs_dims(self):
    #     if self.type == "origin":
    #         return self.env.observation_space.shape
    #     else:
    #         return pms.dims