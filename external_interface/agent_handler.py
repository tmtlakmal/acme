from external_interface.smarts_env import SMARTS_env
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
from external_interface.vehicle_env import Vehicle_env_mp_split
from external_interface.vehicle_gurobi_env import  Vehicle_gurobi_env_mp_split
from acme.agents import  agent
from acme.agents.gurobi import lp

import dm_env
import tensorflow as tf

def array_to_string(array):
    s = ''
    for i in array:
        s += 'x'+str(i)
    return  s

class AgentHandler():

    def __init__(self, env : SMARTS_env):
        self.env : SMARTS_env = env
        self.current_vehicles = set()
        self.env_loops = []

    def get_step_data(self):
        step_data = self.env.get_result()
        self.add_new_requests(step_data['vehicles'])

    def add_common_env(self):
        self.env_loops.clear()
        self.env_loops.append(self.create_env_loop(0, trained=True, gurobi=False))
        self.env_loops[0].load()

    def create_loop(self, id, trained : bool = False, gurobi=False):
        self.env_loops.append(self.create_env_loop(id, trained, gurobi))

    def add_new_requests(self, vehicles):
        for vehicle in vehicles:
            if not vehicle['vid'] in self.current_vehicles:
                if vehicle['externalControl']:
                    self.create_loop(vehicle['vid'])
                    self.current_vehicles.add(vehicle['vid'])

    def step_agents(self):
        for env_loop in self.env_loops:
            env_loop.run_step()

    def fetch_agent_data(self):
        for env_loop in self.env_loops:
            env_loop.fetch_data()

    def create_env_loop(self, id, trained = False, gurobi = False):
        train_summary_writer = self.createTensorboardWriter("./train/", "DQN")

        discounts = [1, 1, 0.9]
        extension = array_to_string(discounts)
        if trained:
            env = self.make_environment(id, env=self.env, front_vehicle=False, extension=extension, gurobi=gurobi)
            agent = self.create_agent(env, discounts, train_summary_writer=None, trained=trained)
            env_loop = acme.EnvironmentLoopSplit(env, agent, tensorboard_writer=None, id=id)
            return env_loop

        env = self.make_environment(id, env=self.env, extension=extension, gurobi=gurobi)
        agent = self.create_agent(env, discounts, train_summary_writer, gurobi=gurobi)
        env_loop = acme.EnvironmentLoopSplit(env, agent, tensorboard_writer=train_summary_writer, id=id)

        return env_loop



    def make_environment(self, id=1, env=None,  multi_objective=True, front_vehicle=False, extension='', gurobi=False) -> dm_env.Environment:

        if gurobi:
            environment = Vehicle_gurobi_env_mp_split(id, 3, front_vehicle=front_vehicle, multi_objective=multi_objective, env=env)
        else:
            environment = Vehicle_env_mp_split(id, 3, front_vehicle=front_vehicle, multi_objective=multi_objective, env=env)

        step_data_file = "episode_data_"+str(id)+"_"+extension+".csv" if multi_objective else "episode_data_single_"+str(id)+"_"+extension+".csv"
        #environment = wrappers.Monitor_save_step_data_split(environment, step_data_file=step_data_file)

        # Make sure the environment obeys the dm_env.Environment interface.
        environment = wrappers.GymWrapperSplit(environment)
        environment = wrappers.SinglePrecisionWrapper(environment)

        return environment

    def createTensorboardWriter(self, tensorboard_log_dir, suffix):
        id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
        train_log_dir = tensorboard_log_dir + suffix + "_" + str(id)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        return train_summary_writer

    def create_agent(self, env : dm_env, discounts : [int], train_summary_writer : None, trained : bool = False, gurobi=False):

        environment_spec =  specs.make_environment_spec(env)
        network = networks.DuellingMLP(3, (128, 128))

        #epsilon_schedule = LinearSchedule(400000, eps_fraction=0.3, eps_start=1, eps_end=0)
        if gurobi:
            agent = lp.LP()

        elif trained:
            epsilon_schedule = LinearSchedule(400000, eps_fraction=1.0, eps_start=0, eps_end=0)
            agent = MOdqn.MODQN(environment_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=1e-3,
                            batch_size=256, samples_per_insert=256.0, tensorboard_writer=train_summary_writer, n_step=5,
                            checkpoint=True, checkpoint_subpath='../examples/gym/checkpoints_single/', target_update_period=200)

        else:
            epsilon_schedule = LinearSchedule(400000, eps_fraction=0.3, eps_start=1, eps_end=0)
            agent = MOdqn.MODQN(environment_spec, network, discount=discounts, epsilon=epsilon_schedule, learning_rate=1e-3,
                            batch_size=256, samples_per_insert=256.0, tensorboard_writer=train_summary_writer, n_step=5,
                            checkpoint=True, checkpoint_subpath='../external_interface/checkpoints/', target_update_period=200)

        return  agent

    def close(self):
        for env_loop in self.env_loops:
            env_loop.close()


