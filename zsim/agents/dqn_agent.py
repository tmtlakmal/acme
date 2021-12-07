import abc
import os.path
import json
from acme.utils import paths
import tensorflow as tf
import acme
from acme import specs
from acme import wrappers
from acme.tf import networks
from acme.agents.tf import dqn
from acme.agents.tf import MOdqn
from acme.agents.tf import tldqn
from acme.utils.schedulers import LinearSchedule


def get_checkpoint_path(pretrained, log_dir):
    if pretrained is not None:
        checkpoint_path = os.path.join(pretrained, 'checkpoints_single')
    else:
        checkpoint_path = os.path.join(log_dir, 'checkpoints_single')
    return checkpoint_path


def get_epsilon_schedule(pretrained, num_steps):
    if pretrained is not None:
        epsilon_schedule = LinearSchedule(num_steps, eps_fraction=1.0, eps_start=0, eps_end=0)
    else:
        epsilon_schedule = LinearSchedule(num_steps, eps_fraction=0.15, eps_start=1, eps_end=0)
    return epsilon_schedule


def create_lexicographic_mo_agent(env_spec, epsilon_schedule, tensorboard_writer, checkpoint_path, agent_opts):
    num_objectives = agent_opts["num_objectives"]
    num_actions = agent_opts["num_actions"]
    discounts = agent_opts["discounts"]
    hidden_dim = agent_opts["hidden_dim"]
    learning_rate = agent_opts["learning_rate"]
    network = []
    for i in range(num_objectives):
        network.append(networks.DuellingMLP(num_actions, (hidden_dim, hidden_dim)))

    return tldqn.TLDQN(env_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=learning_rate,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)


def create_mo_agent(env_spec, epsilon_schedule, tensorboard_writer, checkpoint_path, agent_opts):
    num_actions = agent_opts["num_actions"]
    discounts = agent_opts["discounts"]
    hidden_dim = agent_opts["hidden_dim"]
    learning_rate = agent_opts["learning_rate"]
    network = networks.DuellingMLP(num_actions, (hidden_dim, hidden_dim))
    return MOdqn.MODQN(env_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=learning_rate,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)


def create_non_mo_agent(env_spec, epsilon_schedule, tensorboard_writer, checkpoint_path, agent_opts):
    num_actions = agent_opts["num_actions"]
    discounts = agent_opts["discounts"]
    hidden_dim = agent_opts["hidden_dim"]
    learning_rate = agent_opts["learning_rate"]
    network = networks.DuellingMLP(num_actions, (hidden_dim, hidden_dim))
    return dqn.DQN(env_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=learning_rate,
                  batch_size=256, samples_per_insert=256.0, tensorboard_writer=tensorboard_writer, n_step=5,
                  checkpoint=True, checkpoint_subpath=checkpoint_path, target_update_period=200)


def generate_agent(log_dir, tb_writer, env, agent_opts):
    multi_objective = agent_opts["multi_objective"]
    lexicographic = agent_opts["lexicographic"]
    pretrained = agent_opts["pretrained"]
    eval_only = agent_opts["eval_only"]
    num_steps = agent_opts["train_steps"]
    discounts = agent_opts["discounts"]

    assert not (multi_objective == True and len(discounts) <= 1), \
        "Discount size should be equals to the number of objectives"
    assert (not lexicographic or (multi_objective and lexicographic)), \
        "Lexicograhic should be used with Multiobjective"

    env = wrappers.Monitor_save_step_data(env, step_data_file=os.path.join(log_dir, "episode_data.csv"))
    # Make sure the environment obeys the dm_env.Environment interface.
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
    env_spec = specs.make_environment_spec(env)
    epsilon_schedule = get_epsilon_schedule(pretrained, num_steps)
    checkpoint_path = get_checkpoint_path(pretrained, log_dir)

    if multi_objective and lexicographic:
        agent = create_lexicographic_mo_agent(env_spec,  epsilon_schedule, tb_writer, checkpoint_path, agent_opts)
    elif multi_objective:
        agent = create_mo_agent(env_spec, epsilon_schedule, tb_writer, checkpoint_path, agent_opts)
    else:
        agent = create_non_mo_agent(env_spec, epsilon_schedule, tb_writer, checkpoint_path, agent_opts)

    if pretrained:
        agent.restore()
    if not eval_only:
        loop = acme.EnvironmentLoop(env, agent, tensorboard_writer=tb_writer)
        loop.run(num_steps=num_steps)
        agent.save_checkpoints(force=True)
    return agent
