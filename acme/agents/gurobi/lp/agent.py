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

"""DQN agent implementation."""

import copy
from typing import Optional, Union

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import paths
from acme.utils import loggers
from acme.utils.schedulers import Schedule
from acme.tf.networks import GreedyEpsilonWithDecay
import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class LP(agent.Agent):
  """Linear Programming agent.
  """

  def __init__(
      self,
      batch_size: int = 256,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
  ):


    # Create the actor which defines how we take actions.
    actor = actors.MultiGurobiLpActor(step_size=0.2, max_velocity=22.0)

    # The learner updates the parameters (and initializes them).


    super().__init__(
        actor=actor,
        learner=None,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)

  def update(self):
    pass

  def save_checkpoints(self, force=False):
    pass

  def restore(self):
    pass