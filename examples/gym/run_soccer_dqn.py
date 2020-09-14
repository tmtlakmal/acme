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
from acme.agents.tf import dqn
from absl import app
from absl import flags
import acme
from acme import specs
from acme.utils.schedulers import LinearSchedule
from acme import wrappers
from acme.tf import networks
from acme.utils import paths
from external_env.vehicle_controller.vehicle_env import Vehicle_env
from external_env.soccer.soccer_gym import SoccerGym
from acme.agents import agent

import dm_env
import tensorflow as tf


def make_environment() -> dm_env.Environment:

  environment =  SoccerGym(9,6)
  #environment = wrappers.Monitor_save_step_data(environment)
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

flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to train for.')
flags.DEFINE_integer('num_steps', 400000, 'Number of steps to train for.')
FLAGS = flags.FLAGS

def main(_):

  # Parameters to save and restore
  use_pre_trained = False

  env = make_environment()
  environment_spec = specs.make_environment_spec(env)
  network = networks.DuellingMLP(5,  (32, 32, 32))
  tensorboard_writer = createTensorboardWriter("./train/", "DQN")

  if use_pre_trained:
    epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)
  else:
    epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=0.3, eps_start=1, eps_end=0)

  agent = dqn.DQN(environment_spec, network, discount=1, epsilon=epsilon_schedule, learning_rate=1e-4,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath='./soccer/', target_update_period=200)

  if use_pre_trained:
      agent.restore()

  loop = acme.EnvironmentLoop(env, agent, tensorboard_writer=tensorboard_writer)
  loop.run(num_steps=FLAGS.num_steps)
  #agent.save_checkpoints(force=True)

  test_trained_agent(agent, env, 4000)
  env.close()

def test_trained_agent(agent : agent.Agent,
                       env : dm_env.Environment,
                       num_time_steps : int):
    timestep = env.reset()
    reward = 0
    for _ in range(num_time_steps):
        action = agent.select_action(timestep.observation)
        timestep = env.step(action)
        reward += timestep.reward
        if timestep.last():
            timestep = env.reset()
            print("Episode reward: ", reward)
            reward = 0

if __name__ == '__main__':
  app.run(main)
