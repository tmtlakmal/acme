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
from acme.agents.tf import MOdqn
from absl import app
from absl import flags
import acme
from acme import specs
from acme.utils.schedulers import LinearSchedule
from acme import wrappers
from acme.tf import networks
from acme.utils import paths
from external_env.vehicle_controller.vehicle_env import Vehicle_env
from external_env.vehicle_controller.vehicle_env_cruise import VehicleEnvCruise
from external_env.vehicle_controller.split_env import Vehicle_env_mp_split
from acme.agents import  agent

import dm_env
import tensorflow as tf

tf.random.set_seed(1234)

def make_environment(multi_objective=True, additional_discount='') -> dm_env.Environment:

  environment =  VehicleEnvCruise(2, 3, front_vehicle=True, multi_objective=multi_objective)
  step_data_file = "episode_data_cruise_"+additional_discount+".csv" if multi_objective else "episode_data_cruise.csv"
  environment = wrappers.Monitor_save_step_data(environment, step_data_file=step_data_file)
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

def array_to_string(array):
    s = ''
    for i in array:
        s += 'x'+str(i)
    return  s

def main(_):

  # Parameters to save and restore
  discounts = [1, 0.95]
  use_pre_trained = False
  multi_objective = True
  env = make_environment(multi_objective, array_to_string(discounts))
  environment_spec = specs.make_environment_spec(env)
  network = networks.DQN(3,  (128, 128))
  tensorboard_writer = createTensorboardWriter("./train/", "DQN")

  if use_pre_trained:
    epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)
  else:
    epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=0.15, eps_start=1, eps_end=0)

  if multi_objective:
      agent = MOdqn.MODQN(environment_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=1e-5,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath='./checkpoints/', target_update_period=200)
  else:
      agent = dqn.DQN(environment_spec, network, discount=1, epsilon=epsilon_schedule, learning_rate=1e-3,
                          batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                          checkpoint=True, checkpoint_subpath='./checkpoints/', target_update_period=200)

  if use_pre_trained:
      agent.restore()

  loop = acme.EnvironmentLoop(env, agent, tensorboard_writer=tensorboard_writer)
  loop.run(num_steps=FLAGS.num_steps)

  #for i in range(FLAGS.num_steps):
  #    loop.run_step()
  #    loop.fetch_data()
  agent.save_checkpoints(force=True)

  test_trained_agent(agent, env, 1000)
  print("Pre-trained ##### ")
  epsilon_schedule = LinearSchedule(FLAGS.num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)
  agent = dqn.DQN(environment_spec, network, discount=1, epsilon=epsilon_schedule, learning_rate=1e-3,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath='./checkpoints_1/', target_update_period=200)
  agent.restore()
  test_trained_agent(agent, env, 1000)
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
