import json
import math

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
        paddle_command = action.item() - 1
        vc = VehicleControl(self.id, paddle_command)
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


class SafeRearFollowEnvironment(ExternalEnvironment):
    def __init__(self, env_opts, lexicographic, multi_objective):
        super(SafeRearFollowEnvironment, self).__init__(lexicographic, multi_objective)
        self.ttc_mn = env_opts["ttc_mn"]
        self.ttc_thr = env_opts["ttc_thr"]
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

    @staticmethod
    def get_ttc(gap, speed, front_speed, ttc_thr):
        relative_dis = gap - 4.0
        ttc = ttc_thr
        if relative_dis <= 0:
            ttc = 0
        relative_vel = speed - front_speed
        if relative_vel > 0:
            ttc = relative_dis / relative_vel
        return min(ttc_thr, ttc)

    def encode_message(self, action):
        # Action is from [0,1,2] where paddle commands are [-1, 0, 1]
        paddle_command = action.item() - 1
        vc = VehicleControl(self.id, paddle_command)
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
            vehicle = traffic_data.get_vehicle_with_id(self.id)

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
            ttc = SafeRearFollowEnvironment.get_ttc(gap, speed, pred_speed, self.ttc_thr)
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

            if ttc < self.ttc_mn:
                reward[2] = -5
            elif ttc < self.ttc_thr:
                reward[2] = -math.pow((self.ttc_thr - ttc), 2.0)/100

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
        return super(SafeRearFollowEnvironment, self).reset()


class SafeRearFollowWithBackEnvironment(ExternalEnvironment):
    def __init__(self, env_opts, lexicographic, multi_objective):
        super(SafeRearFollowWithBackEnvironment, self).__init__(lexicographic, multi_objective)
        self.ttc_mn = env_opts["ttc_mn"]
        self.ttc_thr = env_opts["ttc_thr"]
        self.id = env_opts["training_vid"]
        self.control_length = env_opts["control_length"]
        self.max_speed = env_opts["max_speed"]
        self.time_to_reach = env_opts["time_to_reach"]
        self.distance = env_opts["distance"]
        self.front_last_speed = 0
        self.front_vehicle_end_time = 0
        self.back_last_speed = 0
        self.action_space = spaces.Discrete(env_opts["num_actions"])
        self.num_rewards = env_opts["num_rewards"]
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0,  0.0, 0.0, -1.0]),
                                                high=np.array([self.max_speed, self.time_to_reach, self.distance,
                                                               self.distance, self.max_speed, 1.0,
                                                               self.distance, self.max_speed, 1.0]), dtype=np.float32)

    @staticmethod
    def get_ttc(gap, speed, front_speed, ttc_thr):
        relative_dis = gap - 4.0
        ttc = ttc_thr
        if relative_dis <= 0:
            ttc = 0
        relative_vel = speed - front_speed
        if relative_vel > 0:
            ttc = relative_dis / relative_vel
        return min(ttc_thr, ttc)

    @staticmethod
    def get_acc(speed, last_speed):
        return 1 if speed > last_speed else \
            -1 if speed == last_speed \
                else -1

    def encode_message(self, action):
        # Action is from [0,1,2] where paddle commands are [-1, 0, 1]
        paddle_command = action.item() - 1
        vc = VehicleControl(self.id, paddle_command)
        sc = SimulatorExternalControlObjects([vc])
        return sc

    def decode_message(self, obs_message):
        traffic_data = TrafficData(**obs_message)
        obs = [0, 0, 0, 0, 0, 0,  0, 0, 0]
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
            acc = SafeRearFollowWithBackEnvironment.get_acc(pred_speed, self.front_last_speed)
            self.front_last_speed = pred_speed
            ttc = SafeRearFollowEnvironment.get_ttc(gap, speed, pred_speed, self.ttc_thr)

            backv_gap = self.distance
            backv_speed = 0
            back_acc = -1
            back_ttc = self.ttc_thr
            if not vehicle.backvVirtual:
                backv_gap = vehicle.backvGap
                backv_speed = vehicle.backvSpeed
                back_acc = SafeRearFollowWithBackEnvironment.get_acc(backv_speed, self.back_last_speed)
                self.back_last_speed = backv_speed
                back_ttc = SafeRearFollowEnvironment.get_ttc(backv_gap, backv_speed, speed, self.ttc_thr)

            obs = [speed, time_remain, distance_remain, gap, pred_speed, acc, backv_gap, backv_speed, back_acc]
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

            if ttc < self.ttc_mn:
                reward[2] = -5
            elif ttc < self.ttc_thr:
                reward[2] = -(math.pow((self.ttc_thr - ttc), 2.0))/100

            if back_ttc < self.ttc_mn:
                reward[2] += -5
            elif back_ttc < self.ttc_thr:
                reward[2] += -(math.pow((self.ttc_thr - back_ttc), 2.0))/100

            if vehicle.crashed or vehicle.backvCrashed:
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
        self.back_last_speed = 0
        self.front_vehicle_end_time = 0
        return super(SafeRearFollowWithBackEnvironment, self).reset()