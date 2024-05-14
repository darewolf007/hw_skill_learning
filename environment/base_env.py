import os
import collections
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

class base_env(base.Task):
    def __init__(self, input_config, random=None):
        super().__init__(random=random)
        self.camera_names = input_config.camera_names

    def before_step(self, action, physics):
        raise NotImplementedError("Needs to be implemented by child class!")
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos = physics.data.qpos.copy()
        return qpos

    @staticmethod
    def get_qvel(physics):
        qvel = physics.data.qvel.copy()
        return qvel

    @staticmethod
    def get_env_state(physics):
        state = physics.data.qpos.copy()
        return state

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        for camera_name in self.camera_names:
            obs['images'][camera_name] = physics.render(height=480, width=640, camera_id=camera_name)
        return obs

    def get_reward(self, physics):
        raise NotImplementedError("Needs to be implemented by child class!")