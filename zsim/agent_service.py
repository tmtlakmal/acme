import argparse
import pprint as pp
import os.path
import json

import gym

from acme.utils import paths
from acme.utils.loggers.tf_summary import TFSummaryLogger
from external_interface.zeromq_client import ZeroMqClient
from zsim.agents.qn_agent import QLearningAgent
from zsim.dummy.zeromq_server import ZeroMQServer
from zsim.envs.vehicle_envs import NoFollowEnvironment, RearFollowEnvironment, SafeRearFollowEnvironment, \
    SafeRearFollowWithBackEnvironment
from agents import dqn_agent
from enum import Enum
import tensorflow as tf
import acme
from acme import specs
from acme import wrappers
from acme.tf import networks
from acme.agents.tf import dqn
from acme.agents.tf import MOdqn
from acme.agents.tf import tldqn
from acme.utils.schedulers import LinearSchedule
from acme.agents import agent
import dm_env
import numpy as np
from gym import spaces
from acme.utils import loggers

from zsim.external_agent import ExternalEnvironmentAgent


class ExternalEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, lower_bounds, upper_bounds, num_actions, num_rewards):
        super(ExternalEnvironment, self).__init__()
        self.is_episodic = True
        self.reward_space = 1
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds), dtype=np.float32)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass


def init_log_dir(base_log_dir, agent_name):
    prefix = agent_name
    path_id = paths.find_next_path_id(base_log_dir, prefix) + 1
    new_log_dir = base_log_dir + prefix + "_" + str(path_id)
    print("Logs at: ", new_log_dir)
    return new_log_dir


def save_args(log_dir, configs):
    # Save settings in the same 'path' directory
    pp.pprint(configs)
    with open(os.path.join(log_dir, "args.json"), 'w') as f:
        json.dump(configs, f, indent=True)
    return log_dir


def get_checkpoint_path(pretrained, log_dir):
    if pretrained is not None:
        return
    else:
        return


def get_env_spec(lower_bounds, upper_bounds, num_actions, num_rewards):
    env = ExternalEnvironment(lower_bounds, upper_bounds, num_actions, num_rewards)
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    return specs.make_environment_spec(env)


def create_agent(configs):
    num_rewards = configs["num_rewards"]
    lower_bounds = configs["lower_bounds"]
    upper_bounds = configs["upper_bounds"]
    pretrained = configs["pretrained"]
    num_steps = configs["train_steps"]
    num_actions = configs["num_actions"]
    discounts = configs["discounts"]
    hidden_dim = configs["hidden_dim"]
    learning_rate = configs["learning_rate"]
    address = configs["address"]

    def_logger = loggers.make_default_logger("Default", False, 0)
    logger_list = [def_logger]

    if len(pretrained) == 0:
        base_log_dir = configs["base_log_dir"]
        agent_name = configs["agent_name"]
        log_dir = init_log_dir(base_log_dir, agent_name)
        # tb_writer = tf.summary.create_file_writer(log_dir)
        logger1 = TFSummaryLogger(log_dir)
        logger_list.append(logger1)
        tb_writer = logger1.summary
        save_args(log_dir, configs)
        checkpoint_path = os.path.join(log_dir, 'checkpoints_single')
    else:
        tb_writer = None
        checkpoint_path = os.path.join(pretrained, 'checkpoints_single')

    env_spec = get_env_spec(lower_bounds, upper_bounds, num_actions, num_rewards)
    epsilon_schedule = LinearSchedule(num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)

    network = networks.DuellingMLP(num_actions, (hidden_dim, hidden_dim))
    agent = MOdqn.MODQN(env_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=learning_rate,
                        batch_size=256, samples_per_insert=256.0, tensorboard_writer=tb_writer, n_step=5,
                        checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)
    if len(pretrained) != 0:
        agent.restore()
    return ExternalEnvironmentAgent(pretrained, agent, logger_list, address)


if __name__ == '__main__':
    service_agents = dict()
    server = ZeroMQServer()
    message = server.receive()
    while message is not None:
        name = message["name"]
        if name == "End":
            break
        agent = create_agent(message)
        if agent:
            service_agents[name] = agent
            agent.start()
            server.send("Done")
        else:
            server.send("Fail")
        message = server.receive()
    server.close()