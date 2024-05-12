import numpy as np
import contextlib
from collections import deque
from utils.general_utils import ParamDict, AttrDict, listdict2dictlist, obj2np

class Sampler:
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)

        self._env = env
        self._agent = agent
        self._logger = logger
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step, self._episode_reward = 0, 0

    def _default_hparams(self):
        return ParamDict({})

    def init(self, is_train):
        """Starts a new rollout. Render indicates whether output should contain image."""
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                self._episode_reset()

    def sample_action(self, obs):
        return self._agent.act(obs)

    def sample_batch(self, batch_size, is_train=True, global_step=None):
        """Samples an experience batch of the required size."""
        experience_batch = []
        step = 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while step < batch_size:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        if agent_output.action is None:
                            self._episode_reset(global_step)
                            continue
                        agent_output = self._postprocess_agent_output(agent_output)
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs = self._postprocess_obs(obs)
                        experience_batch.append(AttrDict(
                            observation=self._obs,
                            reward=reward,
                            done=done,
                            action=agent_output.action,
                            observation_next=obs,
                        ))

                        # update stored observation
                        self._obs = obs
                        step += 1; self._episode_step += 1; self._episode_reward += reward

                        # reset if episode ends
                        if done or self._episode_step >= self._max_episode_len:
                            if not done:    # force done to be True for timeout
                                experience_batch[-1].done = True
                            self._episode_reset(global_step)

        return listdict2dictlist(experience_batch), step

    def sample_episode(self, is_train, render=False):
        """Samples one episode from the environment."""
        self.init(is_train)
        episode, done = [], False
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while not done and self._episode_step < self._max_episode_len:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        if agent_output.action is None:
                            break
                        agent_output = self._postprocess_agent_output(agent_output)
                        if render:
                            render_obs = self._env.render()
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs = self._postprocess_obs(obs)
                        episode.append(AttrDict(
                            observation=self._obs,
                            reward=reward,
                            done=done,
                            action=agent_output.action,
                            observation_next=obs,
                            info=obj2np(info),
                        ))
                        if render:
                            episode[-1].update(AttrDict(image=render_obs))

                        # update stored observation
                        self._obs = obs
                        self._episode_step += 1
        episode[-1].done = True     # make sure episode is marked as done at final time step

        return listdict2dictlist(episode)

    def get_episode_info(self):
        episode_info = AttrDict(episode_reward=self._episode_reward,
                                episode_length=self._episode_step,)
        if hasattr(self._env, "get_episode_info"):
            episode_info.update(self._env.get_episode_info())
        return episode_info

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        if global_step is not None and self._logger is not None:    # logger is none in non-master threads
            self._logger.log_scalar_dict(self.get_episode_info(),
                                         prefix='train' if self._agent._is_train else 'val',
                                         step=global_step)
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._postprocess_obs(self._reset_env())
        self._agent.reset()

    def _reset_env(self):
        return self._env.reset()

    def _postprocess_obs(self, obs):
        """Optionally post-process observation."""
        return obs

    def _postprocess_agent_output(self, agent_output):
        """Optionally post-process / store agent output."""
        return agent_output