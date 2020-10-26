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
from acme.agents.tf import dqn, r2d2
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
import numpy as np
np.random.seed(0)

def make_environment(model_opponent=False) -> dm_env.Environment:

  environment =  SoccerGym(9,6,opponent_history=model_opponent)
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
flags.DEFINE_integer('num_steps', 800000, 'Number of steps to train for.')
FLAGS = flags.FLAGS

tf.random.set_seed(8)

def main(_):

  # Parameters to save and restore
  use_pre_trained = False
  use_recurrence = False
  model_opponent = False
  simulate_only = False

  env = make_environment(model_opponent)
  environment_spec = specs.make_environment_spec(env)
  if not model_opponent:
    network = networks.DuellingMLP(5,  [32, 32, 32])
  else:
    network = networks.SplitInputDuelling(num_actions=5,  hidden_size=(32, 32, 32), concat_hidden_size=(8,8), split=15)

  if use_pre_trained:
    epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)
  else:
    epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=0.3, eps_start=1, eps_end=0.02)

  tensorboard_writer = createTensorboardWriter("./train/", "DQN")
  if not use_recurrence:
    agent = dqn.DQN(environment_spec, network, discount=1, epsilon=epsilon_schedule, learning_rate=1e-4,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath='./soccer/', target_update_period=400)
  else:
      network = networks.R2D2DeullingNetwork(num_actions=environment_spec.actions.num_values,
                                     lstm_layer_size=128,
                                     feedforward_layers=[128])
      agent = r2d2.R2D2(environment_spec, network, burn_in_length=4, trace_length=10, prefetch_size=4,
                        replay_period=1, discount=1, epsilon=epsilon_schedule, learning_rate=1e-4,
                        tensorboard_writer=tensorboard_writer, target_update_period=400, samples_per_insert=256.0,
                        store_lstm_state=True)

  if use_pre_trained:
      agent.restore()

  if not simulate_only:
    loop = acme.EnvironmentLoop(env, agent, tensorboard_writer=tensorboard_writer)
    loop.run(num_steps=FLAGS.num_steps)
  #agent.save_checkpoints(force=True)

  test_trained_agent(agent, env, 4000)
  env.close()

def test_trained_agent(agent : agent.Agent,
                       env : dm_env.Environment,
                       num_time_steps : int):
    time_step = env.reset()
    reward = 0
    for _ in range(num_time_steps):
        action = agent.select_action(time_step.observation)
        time_step = env.step(action)
        reward += time_step.reward
        env.render()
        if time_step.last():
            time_step = env.reset()
            print("Episode reward: ", reward)
            reward = 0

if __name__ == '__main__':
  app.run(main)
