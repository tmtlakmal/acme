import json
import numpy as np
from gym import spaces

from zsim.envs.control_messages import VehicleControl, SimulatorExternalControlObjects
from zsim.envs.external_env import ExternalEnvironment
from zsim.envs.external_messages import VehicleExternal, TrafficData


class NoFollowEnvironment(ExternalEnvironment):
    def __init__(self, env_opts, lexicographic, multi_objective):
        super(NoFollowEnvironment, self).__init__(lexicographic, multi_objective)
        self.id = env_opts["training_vid"]
        self.control_length = env_opts["control_length"]
        self.max_speed = env_opts["max_speed"]
        self.time_to_reach = env_opts["time_to_reach"]
        self.distance = env_opts["distance"]
        self.num_rewards = env_opts["num_rewards"]
        self.action_space = spaces.Discrete(env_opts["num_actions"])
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]),
                                                high=np.array([self.max_speed, self.time_to_reach, self.distance]), dtype=np.float32)

    def encode_message(self, action):
        # Action is from [0,1,2] where paddle commands are [-1, 0, 1]
        paddle_command = action - 1
        vc = VehicleControl(self.id, paddle_command)
        sc = SimulatorExternalControlObjects([vc])
        return sc

    def decode_message(self, obs_message):
        traffic_data = TrafficData(**json.loads(obs_message))
        obs = [0, 0, 0]
        info = {'is_success': False}
        done = False
        reward = [0.0, 0.0]

        vehicle = None
        if len(traffic_data.vehicles) > 0:
            vehicle = VehicleExternal(**traffic_data.vehicles[0])

        if vehicle != None and vehicle.vid == self.id:
            done = vehicle.done
            speed = vehicle.speed
            time_remain = vehicle.timeRemain
            distance_remain = vehicle.headPositionFromEnd
            obs = [speed, time_remain, distance_remain]

            if done:
                self.episode_num += 1
                if vehicle.is_success:
                    reward[1] = 10.0 + 3 * speed
                    info["is_success"] = True
                else:
                    reward[1] = -10
            else:
                reward[0] = -(distance_remain / self.control_length)
                # self.distance = distance ( a line we had earlier looks like we can remove)
        if self.multi_objective:
            reward = np.array(reward, dtype=np.float32)
        else:
            reward = sum(reward)

        return np.array(obs, dtype=np.float32), reward, done, info


class RearFollowEnvironment(ExternalEnvironment):
    def __init__(self, env_opts, lexicographic, multi_objective):
        super(RearFollowEnvironment, self).__init__(lexicographic, multi_objective)
        self.id = env_opts["training_vid"]
        self.control_length = env_opts["control_length"]
        self.max_speed = env_opts["max_speed"]
        self.time_to_reach = env_opts["time_to_reach"]
        self.distance = env_opts["distance"]
        self.front_last_speed = 0
        self.front_vehicle_end_time = 0
        self.action_space = spaces.Discrete(env_opts["num_actions"])
        self.num_rewards = env_opts["num_rewards"]
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
                                                high=np.array([self.max_speed, self.time_to_reach, self.distance, self.distance, self.max_speed, 1.0]), dtype=np.float32)

    def encode_message(self, action):
        # Action is from [0,1,2] where paddle commands are [-1, 0, 1]
        paddle_command = action - 1
        vc = VehicleControl(self.id, paddle_command.item())
        sc = SimulatorExternalControlObjects([vc])
        return sc

    def decode_message(self, obs_message):
        traffic_data = TrafficData(**obs_message)
        obs = [0, 0, 0, 0, 0, 0]
        info = {'is_success': False, 'is_virtual': False, 'is_crashed': False}
        done = False
        reward = [0.0, 0.0, 0.0]

        vehicle = None
        if len(traffic_data.vehicles) > 0:
            vehicle = VehicleExternal(**traffic_data.vehicles[0])

        if vehicle != None and vehicle.vid == self.id:
            done = vehicle.done
            speed = vehicle.speed
            time_remain = vehicle.timeRemain
            distance_remain = vehicle.headPositionFromEnd
            gap = vehicle.gap
            pred_speed = vehicle.frontVehicleSpeed

            acc = 1 if pred_speed > self.front_last_speed else\
                 -1 if pred_speed == self.front_last_speed \
                 else -1
            self.front_last_speed = pred_speed

            obs = [speed, time_remain, distance_remain, gap, pred_speed, acc]

            # Additional info
            if vehicle.isVirtual:
                self.front_vehicle_end_time += 0.2
                info["is_virtual"] = True

            if done:
                self.episode_num += 1
                if vehicle.is_success:
                    reward[1] = 10.0 + 3 * speed
                    info["is_success"] = True
                else:
                    reward[1] = -10

                if gap < 20 and reward[1] == -10.0 and time_remain < 2:
                    info["is_success"] = True
                    reward[1] = 10.0 + 3*speed

                if distance_remain < 20 and self.front_vehicle_end_time < 2 and time_remain < 2:
                    info["is_success"] = True
                    reward[1] = 10.0 + 3*speed
            else:
                reward[0] = -(distance_remain / self.control_length)

            if gap < 10:
                reward[2] = (gap - 6)/6
            elif gap < 20:
                reward[2] = 0.1

            if vehicle.crashed:
                reward[2] = -400
                reward[1] = 0
                done = True
                info['is_crashed'] = True

        if self.multi_objective:
            reward = np.array(reward, dtype=np.float32)
        else:
            reward = sum(reward)

        return np.array(obs, dtype=np.float32), reward, done, info

    def reset(self):
        self.front_last_speed = 0
        self.front_vehicle_end_time = 0
        return super(RearFollowEnvironment, self).reset()


