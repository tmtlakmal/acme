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
import os.path
import json
from acme.agents.tf import dqn
from acme.agents.tf import MOdqn
from acme.agents.tf import tldqn
from acme.agents.gurobi.lp.agent import LP
import pprint as pp
from absl import flags
import acme
from acme import specs
from acme.utils.schedulers import LinearSchedule
from acme import wrappers
from acme.tf import networks
from acme.utils import paths
from external_env.vehicle_controller.vehicle_env_mp import VehicleEnvMp
from external_env.vehicle_controller.vehicle_env_cruise import VehicleEnvCruise
from acme.agents import  agent

import argparse
import dm_env
import tensorflow as tf

tf.random.set_seed(1234)


def make_environment(num_actions=3, multi_objective=True, lexicographic=False, front_vehicle=False,
                     path='./') -> dm_env.Environment:

  environment =  VehicleEnvMp(2, num_actions=num_actions, front_vehicle=front_vehicle,
                              multi_objective=multi_objective, lexicographic=lexicographic, use_smarts=True)
  step_data_file = os.path.join(path, "episode_data.csv")
  environment = wrappers.Monitor_save_step_data(environment, step_data_file=step_data_file)

  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment

def createNextFileWriter(tensorboard_log_dir, suffix):
    id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
    train_log_dir = tensorboard_log_dir + suffix + "_"+ str(id)
    print("Logs at: ", train_log_dir)
    return train_log_dir

def createNextFileName(tensorboard_log_dir, suffix):
    id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
    return  tensorboard_log_dir + suffix + "_"+ str(id)

flags.DEFINE_integer('num_episodes', 10000, 'Number of episodes to train for.')
flags.DEFINE_integer('num_steps', 500000, 'Number of steps to train for.')

def make_network(num_actions, lexicograhic=False, reward_spec=None, hidden_dim=128):
    if lexicograhic:
        try:
            num_objectives = reward_spec.shape[0]-1 # 3 rewards
        except:
            print("Reward spec is Zero. Setting to One")
            num_objectives = 1
        network = []
        for i in range(num_objectives):
            network.append(networks.DuellingMLP(num_actions, (hidden_dim, hidden_dim)))
    else:
        network = networks.DuellingMLP(num_actions, (hidden_dim, hidden_dim))
    return network

def array_to_string(array):
    s = ''
    for i in array:
        s += 'x'+str(i)
    return  s

def main(opts):

  # Create log directory and the TensorboardWriter
  path = createNextFileWriter(opts.log_dir, "DQN")
  tensorboard_writer = tf.summary.create_file_writer(path)

  # Save settings in the same 'path' directory
  with open(os.path.join(path, "args.json"), 'w') as f:
      json.dump(vars(opts), f, indent=True)

  # Create dm env from gym env
  env = make_environment(num_actions=opts.num_actions, multi_objective=opts.multi_objective,
                         lexicographic=opts.lexicographic, front_vehicle=opts.front_vehicle, path=path)
  environment_spec = specs.make_environment_spec(env)

  # Create NN network
  network = make_network(opts.num_actions, opts.lexicographic, env.reward_spec(), opts.hidden_embed_dim)

  if opts.pretrained is not None:
        epsilon_schedule = LinearSchedule(opts.num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)
  else:
        epsilon_schedule = LinearSchedule(opts.num_steps, eps_fraction=0.15, eps_start=1, eps_end=0)

  checkpoint_path = os.path.join(opts.pretrained if opts.pretrained is not None else path, 'checkpoints_single')


  # Select the agent
  if opts.controller == 'Gurobi':
      agent = LP()
  elif opts.controller == 'Heuristic':
      agent = LP()
  else:
      if opts.multi_objective and opts.lexicographic:
        agent = tldqn.TLDQN(environment_spec, network, discount=opts.discounts, epsilon=epsilon_schedule, learning_rate=opts.lr,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)
      elif opts.multi_objective:
          agent = MOdqn.MODQN(environment_spec, network, discount=opts.discounts, epsilon=epsilon_schedule,
                              learning_rate=opts.lr,
                              batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                              checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)
      else:
            agent = dqn.DQN(environment_spec, network, discount=opts.discounts[0], epsilon=epsilon_schedule, learning_rate=opts.lr,
                          batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                          checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)


  if opts.pretrained is not None:
      agent.restore()
  if not opts.eval_only:
    loop = acme.EnvironmentLoop(env, agent, tensorboard_writer=tensorboard_writer)
    loop.run(num_steps=opts.num_steps)
    agent.save_checkpoints(force=True)

  test_trained_agent(agent, env, opts.test_steps)
  env.close()

import time
def test_trained_agent(agent : agent.Agent,
                       env : dm_env.Environment,
                       num_time_steps : int):
    timestep = env.reset()
    reward = 0
    for _ in range(num_time_steps):
        #s = time.time()
        action = agent.select_action(timestep.observation)
        #print(time.time()-s)
        timestep = env.step(action)
        #print("reward: ", timestep.reward)
        reward += timestep.reward
        if timestep.last():
            timestep = env.reset()
            print("Episode reward: ", reward)
            reward = 0
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent to connect with SMARTS")

    parser.add_argument('--controller', default='RL', help="Available controllers (RL, Gurobi, Heuristic)")
    parser.add_argument('--multi_objective', action='store_true', help="if true use MD-DQN")
    parser.add_argument('--lexicographic', action='store_true', help="if true use TL-DQN")
    parser.add_argument('--log_dir', default='../../logs/', help='Directory to write TensorBoard information to')
    parser.add_argument('--output_dir', default='../../outputs/', help='Directory to write output models to')
    parser.add_argument('--pretrained', help='pretrained file location')
    parser.add_argument('--eval_only', help='Evaluation only, no training')
    parser.add_argument('--discounts', nargs='+', default=[1, 1, 0.9],
                        help="Discount factors for each objective")

    parser.add_argument('--num_steps', default=500000, help='Number of steps to train for.')
    parser.add_argument('--test_steps', default=4000, help='Number of steps to test for.')

    parser.add_argument('--lr', default=5e-5, help='Learning Rate for Adam')
    # Neural Network
    parser.add_argument('--num_layers', default=3, type=int, help='Number of layers in NN')
    parser.add_argument('--hidden_embed_dim', default=128, type=int, help='Hidden layer size')

    ## Similation setting
    parser.add_argument('--front_vehicle', action='store_true', help='Use front vehicle information')
    parser.add_argument('--step_size', default=0.2, help='Use front vehicle information')
    parser.add_argument('--num_actions', default=3, type=int, help='The number of actions the agent can take')


    opts = parser.parse_args()

    ## Set opts here or in the terminal
    opts.multi_objective = True
    opts.lexicographic = True
    opts.controller = "RL"
    opts.eval_only = False
    #opts.pretrained = "../../checkpoints/"
    opts.discounts = [0.9, 0.9, 0.9]
    if opts.lexicographic:
        opts.discounts = [1, 1, 0.95]
    opts.front_vehicle = True

    if opts.controller is not "RL":
        opts.eval_only = True

    assert not (opts.multi_objective == True and len(opts.discounts) <= 1) , "Discount size should be equals to the number of objectives"
    assert (not opts.lexicographic or (opts.multi_objective and opts.lexicographic)), "Lexicograhic should be used with Multiobjective"

    pp.pprint(vars(opts))
    main(opts)
