import argparse
import pprint as pp
import os.path
import json
from acme.utils import paths
from zsim.agents.qn_agent import QLearningAgent
from zsim.envs.vehicle_envs import NoFollowEnvironment, RearFollowEnvironment
from agents import dqn_agent
from enum import Enum
from acme.agents import agent
import dm_env
import tensorflow as tf

class EnvMode(str, Enum):
    NO_FOLLOW = 'NO_FOLLOW'
    REAR_FOLLOW = 'REAR_FOLLOW'

class AgentMode(str, Enum):
    DQN = 'DQN'
    QN = 'QN'
    OTHER = 'OTHER'

def init_log_dir(opts):
    base_log_dir = opts["log_dir"] + opts["env_mode"].name + "/"
    prefix = opts["agent_mode"].name
    path_id = paths.find_next_path_id(base_log_dir, prefix) + 1
    new_log_dir = base_log_dir + prefix + "_" + str(path_id)
    print("Logs at: ", new_log_dir)
    return new_log_dir

def save_args(log_dir, general_opts, env_opts, agent_opts):
    # Save settings in the same 'path' directory
    opts = dict()
    opts.update(general_opts)
    opts.update(env_opts)
    opts.update(agent_opts)
    pp.pprint(opts)
    with open(os.path.join(log_dir, "args.json"), 'w') as f:
        json.dump(opts, f, indent=True)
    return log_dir

def test_trained_agent(agent: agent.Agent,
                       env: dm_env.Environment,
                       num_time_steps: int):
    timestep = env.reset()
    reward = 0
    for _ in range(num_time_steps):
        # s = time.time()
        action = agent.select_action(timestep.observation)
        # print(time.time()-s)
        timestep = env.step(action)
        # print("reward: ", timestep.reward)
        reward += timestep.reward
        if timestep.last():
            timestep = env.reset()
            print("Episode reward: ", reward)
            reward = 0
    env.close()

if __name__ == '__main__':
    general_opts = {}
    agent_opts = {}
    env_opts = {}
    ## Set opts here or in the terminal
    general_opts["agent_mode"] = AgentMode.DQN
    general_opts["env_mode"] = EnvMode.REAR_FOLLOW
    general_opts["log_dir"] = "../logs/"

    env_opts["num_actions"] = 3
    env_opts["num_rewards"] = 3
    env_opts["control_length"] =  200
    env_opts["max_speed"] =  20
    env_opts["time_to_reach"] =  45
    env_opts["distance"] =  300
    env_opts["training_vid"] = 2

    agent_opts["multi_objective"] = True
    agent_opts["lexicographic"] = False
    agent_opts["eval_only"] = False
    agent_opts["pretrained"] = "../../checkpoints/"
    agent_opts["discounts"] = [1, 1, 0.9]
    agent_opts["train_steps"] = 500000
    agent_opts["test_steps"] = 4000
    agent_opts["hidden_dim"] = 128
    agent_opts["num_actions"] = 3
    agent_opts["learning_rate"] = 5e-5

    #Initialize the log directory to save the results
    log_dir = init_log_dir(general_opts)
    tb_writer = tf.summary.create_file_writer(log_dir)
    save_args(log_dir, general_opts, env_opts, agent_opts)

    #Create the environment required
    env_mode = general_opts["env_mode"]
    lexicographic = agent_opts["lexicographic"]
    multi_objective = agent_opts["multi_objective"]
    env = None
    if env_mode == EnvMode.NO_FOLLOW:
        env = NoFollowEnvironment(env_opts, lexicographic, multi_objective)
    elif env_mode == EnvMode.REAR_FOLLOW:
        env = RearFollowEnvironment(env_opts, lexicographic, multi_objective)

    #Create the agent required
    agent_mode = general_opts["agent_mode"]
    agent = None
    if agent_mode == AgentMode.DQN:
        agent = dqn_agent.generate_agent(log_dir, tb_writer, env, agent_opts)
    elif agent_mode == AgentMode.QN:
        agent = QLearningAgent(log_dir, env, agent_opts)

    #Test with the trained agent and close the environment
    test_trained_agent(agent, env, agent_opts["test_steps"])
    env.close()
