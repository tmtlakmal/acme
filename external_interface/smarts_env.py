import gym
import numpy as np
from gym import spaces
from external_interface.zeromq_client import  ZeroMqClient

import zmq
import json
import random
import threading


class SMARTS_env():
    """Custom Environment that follows gym interface"""

    def __init__(self):
        # Zero Mq Client
        self.sim_client = ZeroMqClient()

        # Handle Threads
        self.thread_count_step = []
        self.thread_count_get_states = []
        self.g_step = threading.RLock()
        self.g_get_states = threading.RLock()
        self.step_complete = False
        self.get_state_complete = True
        self.data = []

        # Statistics
        self.iter = 0
        self.num_agents = 0

        #Graphs
        self.agent_actions = []

        #settings
        self.is_CLLA = False
        self.secPerStep = 60

        #####
        self.message_to_send = []
        self.result = []


    def init(self):
        print("Calling Init SMARTS...")
        self.result = self.sim_client.send_message({'settings':{"demandPerOneInterval":6}})

    def update_actions(self, vid : int, message : dict):
        self.message_to_send.append(message)

    def step(self):
        full_message= {'edges': [], 'vehicles': self.message_to_send}
        self.result = self.sim_client.send_message(full_message)
        self.message_to_send.clear()

    def get_result(self):
        return self.result

