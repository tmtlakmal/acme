from zsim.envs.vehicle_envs import SafeRearFollowEnvironment, SafeRearFollowWithBackEnvironment
import numpy as np


class SafeRearFollowTestEnvironment(SafeRearFollowEnvironment):
    def __init__(self, env_opts, lexicographic, multi_objective, env):
        super(SafeRearFollowTestEnvironment, self).__init__(env_opts, lexicographic, multi_objective)
        self.env = env
        self.iter = 0

    def set_id(self, id):
        self.id = id

    def get_step(self):
        message = self.env.get_result()
        observation, reward, done, info = self.decode_message(message)
        return observation, reward, done, info

    def step(self, action):
        self.iter += 1
        # message_send = {"index":self.id, "paddleCommand": action - 1}
        paddle_command = action.item() - 1
        self.env.update_actions(self.id, paddle_command)
        return np.array([0, 40, 400, 20, 380, 0], dtype=np.float32), 0, False, {}

    def reset(self):
        self.front_last_speed = 0
        self.front_vehicle_end_time = 0
        return super(SafeRearFollowEnvironment, self).reset()


class SafeRearFollowWithBackTestEnvironment(SafeRearFollowWithBackEnvironment):
    def __init__(self, env_opts, lexicographic, multi_objective, env):
        super(SafeRearFollowWithBackTestEnvironment, self).__init__(env_opts, lexicographic, multi_objective)
        self.env = env
        self.iter = 0

    def set_id(self, id):
        self.id = id

    def get_step(self):
        message = self.env.get_result()
        observation, reward, done, info = self.decode_message(message)
        return observation, reward, done, info

    def step(self, action):
        self.iter += 1
        # message_send = {"index":self.id, "paddleCommand": action - 1}
        paddle_command = action.item() - 1
        self.env.update_actions(self.id, paddle_command)
        return np.array([0, 40, 400, 20, 380, 0], dtype=np.float32), 0, False, {}

    def reset(self):
        self.front_last_speed = 0
        self.front_vehicle_end_time = 0
        return super(SafeRearFollowWithBackEnvironment, self).reset()