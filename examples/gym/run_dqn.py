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

"""Run DQN on Atari."""

import functools

from absl import app
from absl import flags
import acme
from acme import wrappers
from acme.agents.tf import dqn
from acme.tf import networks
import dm_env
import gym
from typing import Mapping, Sequence

from absl import app
from absl import flags
import acme
from acme import specs
from acme.utils.schedulers import LinearSchedule
from acme import wrappers
from acme.tf import networks
from acme.utils import paths
from acme.utils.loggers import tf_summary
from external_env.vehicle_env import Vehicle_env

import dm_env
import tensorflow as tf


flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to train for.')
flags.DEFINE_integer('num_steps', 200000, 'Number of steps to train for.')
FLAGS = flags.FLAGS



def make_environment() -> dm_env.Environment:

  environment =  Vehicle_env(1,3)
  environment = wrappers.Monitor_save_step_data(environment)
  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment

def createTensorboardWriter(tensorboard_log_dir, suffix):
    id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
    train_log_dir = tensorboard_log_dir + suffix + "_"+ str(id)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    return train_summary_writer

def createNextFileName(tensorboard_log_dir, suffix):
    id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
    return  tensorboard_log_dir + suffix + "_"+ str(id)

def main(_):
  env = make_environment()
  environment_spec = specs.make_environment_spec(env)
  network = networks.DuellingMLP(3,  (32, 32, 32))

  tensorboard_writer = createTensorboardWriter("./train/", "DQN")
  #file_log = createNextFileName("/home/pgunarathna/PycharmProjects/acme/examples/gym/train/", "DQN")
  #logger_dqn = tf_summary.TFSummaryLogger(file_log, "dqn")
  #logger_env = tf_summary.TFSummaryLogger(file_log, "env")
  epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=0.3, eps_start=1, eps_end=0)

  agent = dqn.DQN(environment_spec, network, discount=1, epsilon=epsilon_schedule, learning_rate=1e-3,
                  tensorboard_writer=tensorboard_writer)

  loop = acme.EnvironmentLoop(env, agent, tensorboard_writer=tensorboard_writer)
  loop.run(num_steps=FLAGS.num_steps)

  env.close()


if __name__ == '__main__':
  app.run(main)
