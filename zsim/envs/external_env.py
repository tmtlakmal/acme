import abc
import gym
from external_interface.zeromq_client import ZeroMqClient


class ExternalEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, lexicographic, multi_objective):
        super(ExternalEnvironment, self).__init__()
        self.iter = 0
        self.episode_num = 0
        self.is_episodic = True
        self.client = ZeroMqClient()
        self.reward_space = 3 if lexicographic else 1
        self.multi_objective = multi_objective
        self.lexicographic = lexicographic

    def step(self, action):
        self.iter += 1
        action_message = self.encode_message(action)
        obs_message = self.client.send_message(action_message)
        observation, reward, done, info = self.decode_message(obs_message)
        return observation, reward, done, info

    def reset(self):
        self.iter = 0
        self.episode_num = 0
        init_message = "Connection Reset"
        obs_message = self.client.send_message(init_message)
        observation, reward, done, info = self.decode_message(obs_message)
        return observation

    def render(self, mode="human"):
        pass

    @abc.abstractmethod
    def encode_message(self, action):
        """encode message with the action"""

    @abc.abstractmethod
    def decode_message(self, obs_message):
        """decode message to identify obs, reward, done, info"""
