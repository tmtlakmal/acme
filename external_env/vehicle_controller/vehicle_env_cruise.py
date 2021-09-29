import gym
import zmq
import numpy as np
from gym import spaces
#from External_Interface.zeromq_client import ZeroMqClient
from external_env.vehicle_controller.vehicle_obj import  Vehicle
from external_env.vehicle_controller.simulator_interface import SmartsSimulator, PythonSimulator
from external_env.vehicle_controller.base_controller import *
from external_interface.zeromq_client import ZeroMqClient

class VehicleEnvCruise(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # num_actions : 3 accelerate, no change, de-acceleration
    # max_speed: max speed of vehicles in ms-1
    # time_to_reach: Time remaining to reach destination
    # distance: Distance to destination

    def __init__(self, id, num_actions, use_smarts, max_speed=22.0, time_to_reach=45.0, distance=500.0,
                 front_vehicle=False, multi_objective=False, lexicographic=False):
        super(VehicleEnvCruise, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_actions)
        # Example for using image as input:
        self.iter = 0
        self.sim_client = ZeroMqClient()
        self.is_front_vehicle = front_vehicle

        if not self.is_front_vehicle:
            self.observation_space = spaces.Box(low=np.array([0.0,0.0,0.0]),
                                                high=np.array([max_speed, time_to_reach, distance]), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, -1.0]),
                                                high=np.array([max_speed, distance, distance, max_speed, 1.0]), dtype=np.float32)

        if use_smarts:
            self.sim_interface = SmartsSimulator(vehicle_id=id, num_actions=num_actions, front_vehicle=front_vehicle)
        else:
            self.sim_interface = PythonSimulator(num_actions=num_actions, step_size=self.step_size, max_speed=max_speed,
                                                 time_to_reach=time_to_reach, distance=500.0, front_vehicle=False,
                                                 driver_type=HumanDriven)

        self.is_episodic = True
        self.is_simulator_used = True
        self.time_to_reach = time_to_reach
        self.step_size = 1
        self.id = id
        self.episode_num = 0
        self.correctly_ended = []
        self.default_headway = 2
        self.previous_headway  = 0
        self.previous_distance = 0
        self.previous_location = 0
        self.last_speed = 0

        self.control_length = 400
        self.multi_objective = multi_objective


        # if simulator not used
        self.vehicle = Vehicle()
        self.vehicle_front = Vehicle()
        self.vehicle_controller = HumanDriven(self.step_size, self.vehicle_front.max_acc, self.vehicle_front.max_acc)

    def set_id(self, id):
        self.id = id

    def step(self, action):
        self.iter += 1
        obs, done, info = self.sim_interface.get_step_data(action)
        observation, reward, done, info = self.decode_message(obs, done, info)
        return observation, reward, done, info

    def reset(self):

        obs, done, info = self.sim_interface.reset()
        observation, _, _, _ = self.decode_message(obs, done, info)
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
      # Simulation runs separately
      pass

    def close (self):
      print("Correctly ended episodes", self.correctly_ended)
      pass

    def decode_message(self, obs, done, add_info):

        speed, time, distance, gap, front_vehicle_speed, acc = obs
        # Init all step data
        reward = [0.0, 0.0]
        info = {'is_success':False}

        # sample observation
        obs = []
        if self.is_front_vehicle:
            obs = [speed, distance, gap, front_vehicle_speed, acc]

        reward[0] = -distance/self.control_length

        if done:
            if add_info['is_crashed']:
                info['is_success'] = False
                reward[1] = -400
            else:
                info['is_success'] = True
                reward[1] = 0
        else:
            if (gap < 20):
                reward[1] = (gap - 20)/20
            else:
                reward[1] = 0


        if self.multi_objective:
            reward = np.array(reward, dtype=np.float32)
        else:
            reward = sum(reward)# if reward_1 == 0.0 else reward_1

        return np.array(obs, dtype=np.float32), reward, done, info


    def map_to_paddle_command(self, action):
        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1
        return paddleCommand