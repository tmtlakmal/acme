# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""

import time
from typing import Optional

from acme import core
# Internal imports.
from acme.utils import counting
from acme.utils import loggers

import tensorflow as tf
import dm_env
import numpy as np
from acme.agents.gurobi.lp.agent import LP


class EnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      label: str = 'environment_loop',
      tensorboard_writer = None
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label, False, 5)
    self._tensorboard_writer = tensorboard_writer

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0
    episode_return = 0
    timestep = self._environment.reset()

    # Make the first observation.
    self._actor.observe_first(timestep)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      action = self._actor.select_action(timestep.observation)
      timestep = self._environment.step(action)

      # Have the agent observe the timestep and let the actor update itself.
      self._actor.observe(action, next_timestep=timestep)
      self._actor.update()

      # Book-keeping.
      episode_steps += 1
      episode_return += timestep.reward

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)

    if not self._tensorboard_writer is None:
      current_step = self._counter.get_counts()["steps"]   \
                     if "steps" in self._counter.get_counts() else 0
      with self._tensorboard_writer.as_default():
        with tf.name_scope('environment loop'):
          if isinstance(episode_return, np.ndarray):
            tf.summary.scalar('episode_reward', sum(episode_return), step=current_step)
            tf.summary.scalar('episode_reward_0', episode_return[0], step=current_step)
            tf.summary.scalar('episode_reward_1', episode_return[1], step=current_step)
            if len(episode_return) > 2:
              tf.summary.scalar('episode_reward_2', episode_return[2], step=current_step)
          else:
            tf.summary.scalar('episode_reward', episode_return, step=current_step)

    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    result.update(counts)
    return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count = 0, 0
    while not should_terminate(episode_count, step_count):
      result = self.run_episode()
      episode_count += 1
      step_count += result['episode_length']
      # Log the given results.
      self._logger.write(result)

class EnvironmentLoopSplit(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      label: str = 'environment_loop',
      tensorboard_writer = None,
      id: int = 0
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label, False, 5)
    self._tensorboard_writer = tensorboard_writer

    self.episode_return = 0
    self.episode_steps = 0
    self.needs_reset = True
    self.action = 0
    self.timestep = None
    self.id = id
    self.episode_number = 0

    self.returns_per_vehicle = dict()

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0
    episode_return = 0
    timestep = self._environment.reset()

    # Make the first observation.
    self._actor.observe_first(timestep)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      action = self._actor.select_action(timestep.observation)
      self._environment.step(action)
      timestep = self._environment.get_step()

      # Have the agent observe the timestep and let the actor update itself.
      self._actor.observe(action, next_timestep=timestep)
      self._actor.update()

      # Book-keeping.
      episode_steps += 1
      episode_return += timestep.reward

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)

    if not self._tensorboard_writer is None:
      current_step = self._counter.get_counts()["steps"]   \
                     if "steps" in self._counter.get_counts() else 0
      with self._tensorboard_writer.as_default():
        with tf.name_scope('environment loop'):
          if isinstance(episode_return, np.ndarray):
            tf.summary.scalar('episode_reward', sum(episode_return), step=current_step)
            tf.summary.scalar('episode_reward_0', episode_return[0], step=current_step)
            tf.summary.scalar('episode_reward_1', episode_return[1], step=current_step)
            tf.summary.scalar('episode_reward_2', episode_return[2], step=current_step)
          else:
            tf.summary.scalar('episode_reward', episode_return, step=current_step)

    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    result.update(counts)
    return result

  def run_step(self):

    if self.needs_reset:
      self.episode_steps = 0
      self.episode_return = 0
      self.needs_reset = False
      self.start_time = time.time()
      # Make the first observation.
      self.timestep = self._environment.reset()
      self._actor.observe_first(self.timestep)

    if not self.timestep.last():
      print("Vehicle step data: ", self.id, self.timestep.observation)
      self.action = self._actor.select_action(self.timestep.observation)
      self._environment.step(self.action)
    else:
      self.needs_reset = True
      self._counter.increment(episodes=1, steps=self.episode_steps)

  def fetch_data(self):

    self.timestep = self._environment.get_step()

    self._actor.observe(self.action, next_timestep=self.timestep)
    self._actor.update()
    self.episode_steps += 1
    self.episode_return += self.timestep.reward

    if self.timestep.last():
      self.needs_reset = True

      counts = self._counter.increment(episodes=1, steps=self.episode_steps)
      steps_per_second = self.episode_steps / (time.time() - self.start_time)

      if not self._tensorboard_writer is None:
        current_step = self._counter.get_counts()["steps"] \
          if "steps" in self._counter.get_counts() else 0
        with self._tensorboard_writer.as_default():
          with tf.name_scope('environment loop'):
            if isinstance(self.episode_return, np.ndarray):
              tf.summary.scalar('episode_reward', sum(self.episode_return), step=current_step)
            else:
              tf.summary.scalar('episode_reward', self.episode_return, step=current_step)

      result = {
        'episode_length': self.episode_steps,
        'episode_return': self.episode_return,
        'steps_per_second': steps_per_second,
      }
      result.update(counts)
      self._logger.write(result)
      return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count = 0, 0
    while not should_terminate(episode_count, step_count):
      result = self.run_episode()
      episode_count += 1
      step_count += result['episode_length']
      # Log the given results.
      self._logger.write(result)

  def set_id(self, id):
    self.id = id
    self._environment.set_id(id)

  def online_step(self):


    self.timestep = self._environment.get_step()

    if not self.id in self.returns_per_vehicle:
      #print("time: ", self.timestep.observation[1])
      self.returns_per_vehicle[self.id] = 0

    #if self.id % 1 == 0:
    #  print("Init states:",self.id, self.timestep.observation)

    if isinstance(self._actor, LP):
      self._actor.set_id(self.id)
    action = self._actor.select_action(self.timestep.observation)


    if not self.timestep.reward is None:
      self.returns_per_vehicle[self.id] += self.timestep.reward

    if self.timestep.last():
      print("Episode reward: ", self.id, self.episode_number, self.returns_per_vehicle[self.id])
      del self.returns_per_vehicle[self.id]
      self.episode_return = 0
      self.episode_number += 1
    self._environment.step(action)

  def load(self):
    self._actor.restore()

  def close(self):
    self._environment.close()
    self._actor.save_checkpoints(force=True)

# Internal class.
