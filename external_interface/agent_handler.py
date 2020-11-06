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
from acme.agents import  agent

import dm_env
import tensorflow as tf

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
        self.env_loops.append(self.create_env_loop(0))
        self.env_loops[0].load()

    def add_new_requests(self, vehicles):
        for vehicle in vehicles:
            if not vehicle['vid'] in self.current_vehicles:
                if vehicle['externalControl']:
                    self.env_loops.append(self.create_env_loop(vehicle['vid']))
                    self.current_vehicles.add(vehicle['vid'])

    def step_agents(self):
        for env_loop in self.env_loops:
            env_loop.run_step()

    def fetch_agent_data(self):
        for env_loop in self.env_loops:
            env_loop.fetch_data()

    def create_env_loop(self, id):
        train_summary_writer = self.createTensorboardWriter("./train/", "DQN")

        env = self.make_environment(id, env=self.env)
        agent = self.create_agent(env, train_summary_writer)

        env_loop = acme.EnvironmentLoopSplit(env, agent, tensorboard_writer=train_summary_writer)

        return env_loop



    def make_environment(self, id=1, env=None,  multi_objective=True) -> dm_env.Environment:
        environment = Vehicle_env_mp_split(id, 3, front_vehicle=False, multi_objective=multi_objective, env=env)
        step_data_file = "episode_data_"+str(id)+"_.csv" if multi_objective else "episode_data_single_"+str(id)+"_.csv"
        environment = wrappers.Monitor_save_step_data_split(environment, step_data_file=step_data_file)
        # Make sure the environment obeys the dm_env.Environment interface.
        environment = wrappers.GymWrapperSplit(environment)
        environment = wrappers.SinglePrecisionWrapper(environment)

        return environment

    def createTensorboardWriter(self, tensorboard_log_dir, suffix):
        id = paths.find_next_path_id(tensorboard_log_dir, suffix) + 1
        train_log_dir = tensorboard_log_dir + suffix + "_" + str(id)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        return train_summary_writer

    def create_agent(self, env : dm_env, train_summary_writer):

        environment_spec =  specs.make_environment_spec(env)
        network = networks.DuellingMLP(3, (128, 128))

        #epsilon_schedule = LinearSchedule(400000, eps_fraction=0.3, eps_start=1, eps_end=0)
        epsilon_schedule = LinearSchedule(4000, eps_fraction=1.0, eps_start=0, eps_end=0)
        agent = MOdqn.MODQN(environment_spec, network, discount=1, epsilon=epsilon_schedule, learning_rate=1e-3,
                            batch_size=256, samples_per_insert=256.0, tensorboard_writer=train_summary_writer, n_step=5,
                            checkpoint=True, checkpoint_subpath='../examples/gym/checkpoints_1/', target_update_period=200)

        return  agent

    def close(self):
        for env_loop in self.env_loops:
            env_loop.close()


