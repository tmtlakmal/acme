import gym
import zmq
import numpy as np
from gym import spaces
#from External_Interface.zeromq_client import ZeroMqClient
from external_env.vehicle_controller.vehicle_obj import  Vehicle
from external_env.vehicle_controller.base_controller import *
from external_interface.smarts_env import SMARTS_env

class Vehicle_gurobi_env_mp_split(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # num_actions : 3 accelerate, no change, de-acceleration
    # max_speed: max speed of vehicles in ms-1
    # time_to_reach: Time remaining to reach destination
    # distance: Distance to destination

    def __init__(self, id, num_actions, max_speed=22.0, time_to_reach=45.0, distance=500.0,
                 front_vehicle=False, multi_objective=True, env : SMARTS_env = None):
        super(Vehicle_gurobi_env_mp_split, self).__init__()
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
            self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                                high=np.array([max_speed, time_to_reach, distance, distance, max_speed, time_to_reach]), dtype=np.float32)

        self.is_episodic = True
        self.is_simulator_used = True
        self.time_to_reach = time_to_reach
        self.step_size = 1
        self.id = id
        self.episode_num = 0
        self.correctly_ended = []
        self.default_headway = 2
        self.previous_headway  = 0
        self.previous_location = 0
        self.last_speed = 0
        self.distance = 380
        self.multi_objective = multi_objective


        # if simulator not used
        self.vehicle = Vehicle()
        self.vehicle_front = Vehicle()
        self.vehicle_controller = HumanDriven(self.step_size, self.vehicle_front.max_acc, self.vehicle_front.max_acc)
        self.action = 0
        self.front_vehicle_end_time = 0
        self.env : SMARTS_env = env

    def set_id(self, id):
        self.id = id

    def get_step(self):
        message = self.env.get_result()
        observation, reward, done, info = self.decode_message(message)
        return observation, reward, done, info

    def step(self, action):
        self.iter += 1

        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1

        #print("Action", paddleCommand)
        message_send = {"index":self.id, "paddleCommand": paddleCommand}
        self.env.update_actions(self.id, message_send)

        if self.is_front_vehicle:
            return np.array([0, 40, 400, 20, 380, 0], dtype=np.float32), 0, False, {}
        else:
            return np.array([0, 40, 400], dtype=np.float32), 0, False, {}


    def reset(self):
        #self.sim_client.send_message({"Reset": []})
        if self.is_simulator_used:
            message_send = {"index": self.id, "paddleCommand": 0}
            self.distance = 380
            self.front_vehicle_end_time = 0
            self.env.update_actions(self.id, message_send)
            message = self.env.get_result()
            observation, _, _, _ = self.decode_message(message)
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

    def decode_message(self, message):

        # init variables
        speed = 0
        time = 0
        distance = 0
        obs = [speed,time,distance]
        if self.is_front_vehicle:
            obs = [speed,time,distance,0,0,0]

        done = False
        reward = [0.0, 0.0, 0.0]
        info = {'is_success':False}

        if self.is_simulator_used:
            for vehicle in message["vehicles"]:
                if vehicle["vid"] == self.id:
                    speed   = vehicle["speed"]
                    time    = vehicle["timeRemain"]
                    distance = vehicle["headPositionFromEnd"]
                    done = vehicle["done"]

                    obs = [speed, time, distance]

                    if done:
                        self.episode_num += 1
                        if vehicle["is_success"]:
                            reward[1] = 10.0+3*speed
                            info["is_success"] = True
                        else:
                            reward[1] = -10.0
                    else:
                        reward[0] = -distance/400 # max(self.distance - distance, 0)
                        self.distance = distance

                    if vehicle["isVirtual"]:
                        self.front_vehicle_end_time += 0.2

                    if self.is_front_vehicle:
                        front_vehicle_speed = vehicle['frontVehicleSpeed']
                        front_vehicle_time = vehicle['frontVehicleTimeRemain']
                        front_vehicle_distance = vehicle['frontVehicleDistance']

                        gap = vehicle['gap']

                        headway = gap / max(0.1, speed)
                        obs.extend([front_vehicle_speed, front_vehicle_time, front_vehicle_distance])

                        if done:
                            if gap < 20 and reward[1] == -10.0 and obs[1] < 2:
                                info["is_success"] = True
                                reward[1] = 10.0 + 3*speed

                            if distance < 20 and self.front_vehicle_end_time < 2 and obs[1] < 2:
                                info["is_success"] = True
                                reward[1] = 10.0 + 3*speed

                        if (gap > 6 and gap < 20):
                            reward[2] = 0.1

                        if (gap < 10):
                            reward[2] = (gap - 6)/6

                        if (vehicle['crashed']):
                            print("############### Crashed #############")
                            reward[2] = -400
                            reward[1] = 0
                            done = True

                        self.last_speed = front_vehicle_speed

                        # TODO: Remove this block after training



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
            reward = np.array(reward, dtype=np.float32)
        else:
            reward = sum(reward) # if reward_1 == 0.0 else reward_1

        for main_vehicle in message["vehicles"]:
            if main_vehicle['vid'] == 2 and main_vehicle['done'] and self.id != 2:
                info["is_success"] = True
                done = True


        return np.array(obs, dtype=np.float32), reward, done, info


    def map_to_paddle_command(self, action):
        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1
        return paddleCommand

