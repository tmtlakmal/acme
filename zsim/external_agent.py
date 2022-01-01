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
    WRITE = "WRITE"


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
    def __init__(self, pretrained, agent: Agent, loggers:[Logger], address):
        threading.Thread.__init__(self)
        self.pretrained = pretrained
        self.agent = agent
        self.loggers = loggers
        self.address = address
        self.server = None

    def run(self):
        self.server = ZeroMQServer(self.address)
        while True:
            json_msg = self.server.receive()
            action_msg = ActionMessage(**json_msg)
            action = action_msg.action
            if action == Action.ABORT:
                break
            status = self._get_status_message(action_msg)
            self.server.send(status)
        if not self.pretrained:
            self.agent.save_checkpoints(force=True)
            print("Saved the Final checkpoint")
        self.server.close()

    def _get_status_message(self, action_msg):
        action = action_msg.action
        if action == Action.SELECT_ACTION:
            observation = action_msg.payload["observation"]
            action = self.agent.select_action(_convert_value(np.array(observation, dtype=np.float32)))
            return StatusMessage(Status.SUCCESS, action.item())

        elif action == Action.OBSERVE_FIRST:
            observation = action_msg.payload["timestep"]["observation"]
            timestep = dm_env.restart(np.array(observation, dtype=np.float32))
            self.agent.observe_first(_convert_timestep(timestep))
            return StatusMessage(Status.SUCCESS, None)

        elif action == Action.OBSERVE:
            action = action_msg.payload["timestep"]["action"]
            step_type = action_msg.payload["timestep"]["step_type"]
            reward = action_msg.payload["timestep"]["reward"]
            reward = np.array(reward, dtype=np.float32)
            observation = action_msg.payload["timestep"]["observation"]
            observation = np.array(observation, dtype=np.float32)
            next_timestep = dm_env.transition(reward=reward, observation=observation)
            if step_type == StepType.LAST.name:
                next_timestep = dm_env.termination(reward=reward, observation=observation)
            self.agent.observe(np.array(action, dtype=np.int32), _convert_timestep(next_timestep))
            self.agent.update()
            if step_type == StepType.LAST.name:
                log_data = action_msg.payload["log_data"]
                for logger in self.loggers:
                    logger.write(log_data)
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

