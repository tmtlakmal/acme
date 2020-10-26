import gym
import numpy as np
from gym import spaces
#from External_Interface.zeromq_client import ZeroMqClient
from external_env.vehicle_controller.vehicle_obj import  Vehicle
from external_env.vehicle_controller.base_controller import *

class Vehicle_env_mp_split(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # num_actions : 3 accelerate, no change, de-acceleration
    # max_speed: max speed of vehicles in ms-1
    # time_to_reach: Time remaining to reach destination
    # distance: Distance to destination

    def __init__(self, id, num_actions, max_speed=22.0, time_to_reach=45.0, distance=400.0,
                 front_vehicle=False, multi_objective=True):
        super(Vehicle_env_mp_split, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_actions)
        # Example for using image as input:
        self.iter = 0
        #self.sim_client = ZeroMqClient()
        self.is_front_vehicle = front_vehicle

        if not self.is_front_vehicle:
            self.observation_space = spaces.Box(low=np.array([0.0,0.0,0.0]),
                                                high=np.array([max_speed, time_to_reach, distance]), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                                                high=np.array([max_speed, time_to_reach, distance, distance, max_speed]), dtype=np.float32)

        self.is_episodic = True
        self.is_simulator_used = False
        self.time_to_reach = time_to_reach
        self.step_size = 1
        self.id = 1
        self.episode_num = 0
        self.correctly_ended = []

        self.multi_objective = multi_objective


        # if simulator not used
        self.vehicle = Vehicle()
        self.vehicle_front = Vehicle()
        self.vehicle_controller = HumanDriven(self.step_size, self.vehicle_front.max_acc, self.vehicle_front.max_acc)
        self.action = 0

    def set_id(self, id):
        self.id = id



    def step(self, action):
        self.iter += 1

        self.action = action
        if self.is_front_vehicle:
            return np.array([0, 40, 400, 20, 380], dtype=np.float32), 0, False, {}
        else:
            return np.array([0, 40, 400], dtype=np.float32), 0, False, {}

        #print("Action", paddleCommand)

    def get_step(self):
        if self.action == 0:
            paddleCommand = -1
        elif self.action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1

        message_send = {'edges': [], 'vehicles':[{"index":self.id, "paddleCommand": paddleCommand}]}

        if self.is_simulator_used:
            message = self.sim_client.send_message(message_send)
        else:
            message = 0

        if self.is_front_vehicle:
            acc = self.vehicle_controller.compute_acceleration(self.vehicle_front)
            self.vehicle_front.step(self.map_to_paddle_command(acc))

        observation, reward, done, info = self.decode_message(message, paddleCommand)

        return observation, reward, done, info

    def reset(self):
        #self.sim_client.send_message({"Reset": []})
        if self.is_simulator_used:
            message_send = {'edges': [], 'vehicles': [{"index": self.id, "paddleCommand": 0}]}
            message = self.sim_client.send_message(message_send)
            observation, _, _, _ = self.decode_message(message, 0)
            #self.time_to_reach = np.random.randint(8,16)
            return observation # reward, done, info can't be included
        else:
            if self.is_front_vehicle:
                self.vehicle_controller = np.random.choice([
                    #HumanDriven(self.step_size, self.vehicle_front.max_acc, self.vehicle_front.max_acc),
                    HumanDriven_1(self.step_size, self.vehicle_front.max_acc, self.vehicle_front.max_acc),
                    HumanDriven_2(self.step_size, self.vehicle_front.max_acc, self.vehicle_front.max_acc)
                    ])

                l = self.vehicle.reset()
                time_to_reach = int(l[1])-1
                self.vehicle_front.assign_data()
                l.extend([self.vehicle.location - self.vehicle_front.location,
                            self.vehicle_front.speed])

                return np.array(l, dtype=np.float32)
            else:
                return  np.array(self.vehicle.reset(), dtype=np.float32)

    def render(self, mode='human'):
      # Simulation runs separately
      pass

    def close (self):
      print("Correctly ended episodes", self.correctly_ended)
      pass

    def decode_message(self, message, action):

        speed = 0
        time = 0
        distance = 0
        obs = [speed,time,distance]
        done = False
        reward = 0.0
        reward_1 = 0.0
        info = {'is_success':False}

        if self.is_simulator_used:
            #print(message["vehicles"])
            for vehicle in message["vehicles"]:
                if vehicle["vid"] == self.id:
                    speed   = int(round(vehicle["speed"]))
                    time    = int(vehicle["timeRemain"])
                    distance = int(round(vehicle["headPositionFromEnd"]))
                    done = vehicle["done"]

                    obs = [speed, time, distance]

                    if done:
                        self.episode_num += 1
                        if vehicle["is_success"]:
                            reward = 10+speed
                            info["is_success"] = True
                        else:
                            reward = -10
                    else:
                        reward = -distance/400

                    #print("Reward  given", reward)

        else:
            obs, reward, done, info = self.vehicle.step(action)

            if self.is_front_vehicle:
                obs.extend([self.vehicle.location - self.vehicle_front.location,
                            self.vehicle_front.speed])
                #reward += abs(abs(self.vehicle_front.location - self.vehicle.location) - 2)/400
                if done:
                    if abs(self.vehicle_front.location - self.vehicle.location) < 20 \
                            and reward == -10.0 and obs[1] < 2:
                        reward = 10.0 + self.vehicle.speed
                        info['is_success'] = True


                if (self.vehicle_front.location - self.vehicle.location) > 0 and self.vehicle_front.location > 0 :
                    reward_1 = -20.0
                    done = True
                else:
                    reward_1 = 0.0

            obs = [float(i) for i in obs]

            if done:
                self.episode_num += 1
                if info['is_success']:
                    self.correctly_ended.append(self.episode_num)

        if self.multi_objective:
            reward = np.array([reward, reward_1], dtype=np.float32)
        else:
            reward = reward if reward_1 == 0.0 else reward_1

        return np.array(obs, dtype=np.float32), reward, done, info


    def map_to_paddle_command(self, action):
        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1
        return paddleCommand