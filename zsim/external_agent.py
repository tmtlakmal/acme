import dm_env

from acme.agents.agent import Agent
from external_interface.zeromq_client import ZeroMqClient
from enum import Enum
from dm_env import TimeStep, StepType
import numpy as np
from acme.wrappers.single_precision import _convert_value
from acme.utils.loggers.base import Logger
import threading

from zsim.dummy.zeromq_server import ZeroMQServer


def _convert_timestep(timestep: TimeStep) -> TimeStep:
    return timestep._replace(
        reward=_convert_value(timestep.reward),
        discount=_convert_value(timestep.discount),
        observation=_convert_value(timestep.observation))


class Action(str, Enum):
    SELECT_ACTION = "SELECT_ACTION"
    OBSERVE_FIRST = "OBSERVE_FIRST"
    OBSERVE = "OBSERVE"
    UPDATE = "UPDATE"
    ABORT = "ABORT"


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class ActionMessage:
    def __init__(self, action: Action, payload):
        self.action = action
        self.payload = payload


class StatusMessage:
    def __init__(self, status: Status, payload):
        self.status = status
        self.payload = payload


class ExternalEnvironmentAgent(threading.Thread):
    def __init__(self, agent: Agent, logger:Logger, address):
        threading.Thread.__init__(self)
        self.agent = agent
        self.logger = logger
        self.address = address
        self.server = None

    def run(self):
        self.server = ZeroMQServer(self.address)
        action = self.server.receive()
        while action is not None:
            status = self._get_status_message(action)
            action = self.server.send_and_receive(status)

    def _get_status_message(self, action_msg):
        action_msg = ActionMessage(**action_msg)
        action = action_msg.action

        if action == Action.SELECT_ACTION:
            observation = action_msg.payload["observation"]
            self.agent.select_action(_convert_value(np.array(observation, dtype=np.float32)))
            return StatusMessage(Status.SUCCESS, 1)

        elif action == Action.OBSERVE_FIRST:
            observation = action_msg.payload["timestep"]["observation"]
            timestep = dm_env.restart(np.array(observation, dtype=np.float32))
            self.agent.observe_first(_convert_timestep(timestep))
            return StatusMessage(Status.SUCCESS, None)

        elif action == Action.OBSERVE:
            action = action_msg.payload["action"]
            step_type = action_msg.payload["next_timestep"]["step_type"]
            reward = action_msg.payload["next_timestep"]["reward"]
            reward = np.array(reward, dtype=np.float32)
            observation = action_msg.payload["next_timestep"]["observation"]
            observation = np.array(observation, dtype=np.float32)
            next_timestep = dm_env.transition(reward=reward, observation=observation)
            if step_type is StepType.LAST:
                next_timestep = dm_env.termination(reward=reward, observation=observation)
            self.agent.observe(np.array(action, dtype=np.int), _convert_timestep(next_timestep))
            self.agent.update()
            return StatusMessage(Status.SUCCESS, None)

        elif action == Action.WRITE:
            log_data = action_msg.payload["log_data"]
            self.logger.write(log_data)
            return StatusMessage(Status.SUCCESS, None)
        elif action == Action.UPDATE:
            self.agent.update()
            return StatusMessage(Status.SUCCESS, None)

        else:
            return None


