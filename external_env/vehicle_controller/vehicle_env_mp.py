import gym
from gym import spaces
from external_env.vehicle_controller.simulator_interface import SmartsSimulator, PythonSimulator
from external_env.vehicle_controller.base_controller import *
from external_interface.zeromq_client import ZeroMqClient

class Vehicle_env_mp(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, id, num_actions, max_speed=22.0, time_to_reach=45.0, distance=500.0,
                 front_vehicle=False, multi_objective=True, use_smarts=False):
        super(Vehicle_env_mp, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(num_actions)

        self.iter = 0
        self.sim_client = ZeroMqClient()
        self.is_front_vehicle = front_vehicle

        if not self.is_front_vehicle:
            self.observation_space = spaces.Box(low=np.array([0.0,0.0,0.0]),
                                                high=np.array([max_speed, time_to_reach, distance]), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
                                                high=np.array([max_speed, time_to_reach, distance, distance, max_speed, 1.0]), dtype=np.float32)

        self.is_episodic = True
        self.is_simulator_used = use_smarts

        self.step_size = 0.2

        if use_smarts:
            self.sim_interface = SmartsSimulator(vehicle_id=id, num_actions=num_actions, front_vehicle=front_vehicle)
        else:
            self.sim_interface = PythonSimulator(num_actions=num_actions, step_size=self.step_size, max_speed=max_speed,
                                                 time_to_reach=time_to_reach, distance=500.0, front_vehicle=False,
                                                 driver_type=HumanDriven)

        self.time_to_reach = time_to_reach

        self.id = id
        self.episode_num = 0
        self.correctly_ended = []
        self.default_headway = 2
        self.previous_headway  = 0
        self.previous_location = 0
        self.last_speed = 0
        self.distance = 0
        self.multi_objective = multi_objective
        self.control_length = 400



        # if simulator not used
        self.front_vehicle_end_time = 0

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
        return observation # reward, done, info can't be included

    def render(self, mode='human'):
      # Simulation runs separately
      pass

    def close (self):
      print("Correctly ended episodes", self.correctly_ended)
      pass

    def decode_message1(self, obs, done, add_info):
        speed, time, distance, gap, front_vehicle_speed, acc = obs

        if not self.is_front_vehicle:
            obs = [speed, time, distance]

        info = {'is_success':False}

        if done:
            reward = 20*((1-abs(distance)/self.control_length) + (1-time/5) + speed/25)
            if add_info["is_success"]:
                info["is_success"] = True
        else:
            reward = -distance/self.control_length

        return np.array(obs, dtype=np.float32), reward, done, info

    def decode_message(self, obs, done, add_info):

        speed, time, distance, gap, front_vehicle_speed, acc = obs

        if not self.is_front_vehicle:
            obs = [speed, time, distance]

        reward = [0.0, 0.0, 0.0]
        info = {'is_success':False}

        if done:
            self.episode_num += 1
            if add_info["is_success"]:
                reward[1] = 10.0*(1-time/10)+3*speed
                info["is_success"] = True
            else:
                reward[1] = -10
        else:
            reward[0] = -distance/self.control_length #self.distance - distance
            self.distance = distance

            if self.is_front_vehicle:
                if add_info["is_virtual"]:
                    self.front_vehicle_end_time += 0.2

                if done:
                    if gap < 20 and reward[1] == -10.0 and obs[1] < 2:
                        info["is_success"] = True
                        reward[1] = 10.0 + 3*speed

                    if distance < 20 and self.front_vehicle_end_time < 2 and obs[1] < 2:
                        info["is_success"] = True
                        reward[1] = 10.0 + 3*speed

                if (gap < 10):
                    reward[2] = (gap - 6)/6
                elif (gap < 20):
                    reward[2] = 0.1

                if (add_info['is_crashed']):
                    reward[2] = -400
                    reward[1] = 0
                    done = True

        """else:
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
                    self.correctly_ended.append(self.episode_num)"""

        if self.multi_objective:
            reward = np.array(reward, dtype=np.float32)
        else:
            reward = sum(reward)

        return np.array(obs, dtype=np.float32), reward, done, info


    def map_to_paddle_command(self, action):
        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1
        return paddleCommand