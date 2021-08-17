# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic actor implementation, using TensorFlow and Sonnet."""
from typing import Optional, Union

from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils.schedulers import Schedule

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree

tfd = tfp.distributions

import gurobipy as gp
from gurobipy import GRB
import numpy as np

class FeedForwardActor(core.Actor):
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      epsilon_scheduler: Optional[Union[Schedule, tf.Tensor]] = None,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      tensorboard_writer=None
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_network = tf.function(policy_network)
    self.epsilon_scheduler = epsilon_scheduler
    self.epsilon = tf.Variable(1.0, trainable=False) if isinstance(self.epsilon_scheduler, Schedule) \
                   else epsilon_scheduler

    self._tensorboard_writer = tensorboard_writer

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    if isinstance(self.epsilon_scheduler, Schedule):
      self.epsilon.assign(self.epsilon_scheduler.value())

    # Forward the policy network.
    policy_output = self._policy_network(batched_obs, self.epsilon)



    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    if self._variable_client:
      self._variable_client.update()

class TLForwardActor(core.Actor):
  """A set of feed-forward actors with lexicographic order.

  The actors based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. Each actor takes action one after the other based
  on a lexicographic order.It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_networks: [snt.Module],
      epsilon_scheduler: Optional[Union[Schedule, tf.Tensor]] = None,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      tensorboard_writer=None
  ):
    """Initializes the actor.

    Args:
      policy_networks: the policies to run in lexicographic order.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_networks = [tf.function(policy_network)  for policy_network in policy_networks]
    self.epsilon_scheduler = epsilon_scheduler
    self.epsilon = tf.Variable(1.0, trainable=False) if isinstance(self.epsilon_scheduler, Schedule) \
                   else epsilon_scheduler

    self._tensorboard_writer = tensorboard_writer

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    if isinstance(self.epsilon_scheduler, Schedule):
      self.epsilon.assign(self.epsilon_scheduler.value())

    # Forward the policy network.
    policy_output = []
    for policy_network in self._policy_networks:
      policy_output.append(policy_network(batched_obs, self.epsilon))

    #process order of acton and select

    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    if self._variable_client:
      self._variable_client.update()

class RecurrentActor(core.Actor):
  """A recurrent actor.

  An actor based on a recurrent policy which takes non-batched observations and
  outputs non-batched actions, and keeps track of the recurrent state inside. It
  also allows adding experiences to replay and updating the weights from the
  policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.RNNCore,
      epsilon_scheduler: Optional[Union[Schedule, tf.Tensor]] = None,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the (recurrent) policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """
    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._network = policy_network
    self._state = None
    self._prev_state = None

    self.epsilon_scheduler = epsilon_scheduler
    self.epsilon = tf.Variable(1.0, trainable=False) if isinstance(self.epsilon_scheduler, Schedule) \
                   else epsilon_scheduler

    # TODO(b/152382420): Ideally we would call tf.function(network) instead but
    # this results in an error when using acme RNN snapshots.
    self._policy = tf.function(policy_network.__call__)


  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    if isinstance(self.epsilon_scheduler, Schedule):
      self.epsilon.assign(self.epsilon_scheduler.value())

    # Initialize the RNN state if necessary.
    if self._state is None:
      self._state = self._network.initial_state(1)

    # Forward.
    policy_output, new_state = self._policy(batched_obs, self._state, self.epsilon)

    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    self._prev_state = self._state
    self._state = new_state

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if not self._adder:
      return

    numpy_state = tf2_utils.to_numpy_squeeze(self._prev_state)
    self._adder.add(action, next_timestep, extras=(numpy_state,))

  def update(self):
    if self._variable_client:
      self._variable_client.update()


# Internal class 1.
# Internal class 2.

class GurobiLpActor(core.Actor):

  def __init__(self, step_size, max_velocity):

    self.step_size = step_size

    self.model = None
    self.front_vehicle_model = None
    self.positions = None
    self.velocities = None
    self.accelerations = None
    self.max_velocity = max_velocity

    self.front_vehicle_positions = None

    # step managers
    self.current_index = 0
    self.last_remain_time = 0
    self.front_vehicle_remain_time = 0

  def compute_trajectory(self, observation, include_front_vehicle = False):

    env  = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("mip1", env=env)
    model.Params.BarConvTol = 0.05

    N = int(observation[1] / self.step_size)

    distance = np.float64(observation[2])

    positions = model.addVars(1, N, lb=-distance, ub=10.0, vtype='C', name='p')
    velocity = model.addVars(1, N, lb=0.0, ub=self.max_velocity, vtype='C', name='v')
    accelerations = model.addVars(1, N, lb=-1, ub=1, vtype='I', name='a')


    for i in range(N - 1):
      model.addConstr(positions[0, i + 1] == positions[0, i] + velocity[0, i + 1] * self.step_size)
      model.addConstr(velocity[0, i + 1] == velocity[0, i] + accelerations[0, i] * 2 * self.step_size)

    model.addConstr(positions[0, 0] == -distance)
    model.addConstr(positions[0, N - 1] >= -5)

    model.addConstr(velocity[0, 0] == round(np.float64(observation[0]), 2))
    # Constrain the end velocity between max velocity and max velocity - 4
    #model.addConstr(velocity[0, N - 1] <= self.max_velocity)
    #model.addConstr(velocity[0, N - 1] >= self.max_velocity - 4)

    if include_front_vehicle and self.front_vehicle_positions != None:
      for j in range(3, len(self.front_vehicle_positions)):
        model.addConstr(positions[0, j] <= self.front_vehicle_positions[0, j].x - 5)

    model.setObjective(positions.sum() / 400 + 3 * velocity[0, N - 1], GRB.MAXIMIZE)

    model.optimize()
    #print("Current length: ", N)
    return model, positions, velocity, accelerations

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:

    if (self.last_remain_time - observation[1] <= 0 or (observation.size > 4 and self.front_vehicle_remain_time - observation[4] < 0)):
      # recompute
      if observation[1] == 0:
        return 0
      #print("Observation: ", observation)
      self.last_remain_time = observation[1]

      if observation.size > 4:
        self.front_vehicle_remain_time = observation[4]
      self.current_index = 0

      if observation.size > 4 and not observation[5] == 0:
        self.front_vehicle_model, self.front_vehicle_positions, _, _ =  self.compute_trajectory(observation[3:6], include_front_vehicle=False)
        self.model, self.positions, self.velocities, self.accelerations = self.compute_trajectory(observation, include_front_vehicle=True)
      else:
        self.model, self.positions, self.velocities, self.accelerations = self.compute_trajectory(observation, include_front_vehicle=False)

      #for j in range(len(self.front_vehicle_positions)):
      #  print(j, self.positions[0, j].x, self.front_vehicle_positions[0, j].x, self.positions[0, j].x - self.front_vehicle_positions[0, j].x)
      action = np.array(self.accelerations[0,0].x+1)

      return action

    else:
      self.current_index += 1
      self.last_remain_time = observation[1]
      action = np.array(self.accelerations[0, self.current_index].x+1)
      return action

  def observe_first(self, timestep: dm_env.TimeStep):
     pass

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    pass

  def update(self):
    pass

  def to_array(self, data : gp.tupledict):
    list_data = np.zeros(len(data), dtype=np.float)
    for i in range(len(data)):
      list_data[i] = data[0, i].x

    return  list_data

class MultiGurobiLpActor(core.Actor):

  def __init__(self, step_size, max_velocity):

    self.step_size = step_size

    self.model = None
    self.front_vehicle_model = None
    self.positions = None
    self.velocities = None
    self.accelerations = None
    self.max_velocity = max_velocity

    self.front_vehicle_positions = None

    # step managers
    self.current_index = 0
    self.last_remain_time = 0
    self.front_vehicle_remain_time = 0

    self.id = -1

    self.trajectory_data = {}

  def set_id(self, id):
    self.id = id

  def compute_trajectory(self, observation, include_front_vehicle = False):

    env  = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model("mip1", env=env)
    model.Params.BarConvTol = 1e-5

    N = int(observation[1] / self.step_size)

    distance = np.float64(observation[2])

    positions = model.addVars(1, N, lb=-distance, ub=10.0, vtype='C', name='p')
    velocity = model.addVars(1, N, lb=0.0, ub=self.max_velocity, vtype='C', name='v')
    accelerations = model.addVars(1, N, lb=-1, ub=1, vtype='I', name='a')


    for i in range(N - 1):
      model.addConstr(positions[0, i + 1] == positions[0, i] + velocity[0, i + 1] * self.step_size)
      model.addConstr(velocity[0, i + 1] == velocity[0, i] + (2.5*accelerations[0, i]-0.5)  * self.step_size)


    model.addConstr(positions[0, 0] == -distance)
    model.addConstr(positions[0, N - 1] >= -5)
    model.addConstr(positions[0, N - 1] <= 2)

    model.addConstr(velocity[0, 0] == round(np.float64(observation[0]), 2))
    # Constrain the end velocity between max velocity and max velocity - 4
    #model.addConstr(velocity[0, N - 1] <= self.max_velocity)
    #model.addConstr(velocity[0, N - 1] >= self.max_velocity - 4)

    if include_front_vehicle and self.front_vehicle_positions != None:
      for j in range(3, len(self.front_vehicle_positions)):
        model.addConstr(positions[0, j] <= self.front_vehicle_positions[0, j].x - 5)

    model.setObjective(positions.sum() / 400 + 3 * velocity[0, N - 1], GRB.MAXIMIZE)

    model.optimize()
    #print("Current length: ", N)
    return model, positions, velocity, accelerations

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:

    if (self.id in self.trajectory_data):
      self.last_remain_time = self.trajectory_data[self.id].get_last_remain_time()
      self.front_vehicle_remain_time = 0

    if ((not self.id in self.trajectory_data) or abs(self.last_remain_time - observation[1]) >= 1 or (observation.size > 4 and self.front_vehicle_remain_time - observation[4] < 0)):
      # recompute
      if observation[1] == 0:
        return 0
      #print("Observation: ", observation)
      self.last_remain_time = observation[1]

      if observation.size > 4:
        self.front_vehicle_remain_time = observation[4]
      self.current_index = 0

      if observation.size > 4 and not observation[5] == 0:
        self.front_vehicle_model, self.front_vehicle_positions, _, _ =  self.compute_trajectory(observation[3:6], include_front_vehicle=False)
        self.model, self.positions, self.velocities, self.accelerations = self.compute_trajectory(observation, include_front_vehicle=True)
      else:
        self.model, self.positions, self.velocities, self.accelerations = self.compute_trajectory(observation, include_front_vehicle=False)

      #for j in range(len(self.front_vehicle_positions)):
      #  print(j, self.positions[0, j].x, self.front_vehicle_positions[0, j].x, self.positions[0, j].x - self.front_vehicle_positions[0, j].x)
      #print(observation)
      action = np.array(self.accelerations[0,0].x+1)

      if (self.id in self.trajectory_data):
        self.trajectory_data[self.id].update(self.model, self.accelerations, self.positions, self.velocities, observation[1])
      else:
        self.trajectory_data[self.id] = Trajectory(self.model, self.accelerations, self.positions, self.velocities, observation[1])

      return action

    else:
      self.trajectory_data[self.id].set_last_remain_time(observation[1])
      action = self.trajectory_data[self.id].get_acceleration()
      return action

  def observe_first(self, timestep: dm_env.TimeStep):
     pass

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    pass

  def update(self):
    pass

  def to_array(self, data : gp.tupledict):
    list_data = np.zeros(len(data), dtype=np.float)
    for i in range(len(data)):
      list_data[i] = data[0, i].x

    return  list_data
import copy

class Heuristic(core.Actor):

  def __init__(self, step_size, max_velocity):
      self.step_size = step_size
      self.max_velocity = max_velocity
      self.acceleration = 2

      self.early_stopping_distance = 200
      self.early_start_time = 15

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
      speed, time, distance = observation[0], observation[1], observation[2]

      temp_time = (self.max_velocity - speed)/self.acceleration
      extra_time = max(time-temp_time, 0)
      time_to_max_speed = time-extra_time

      predicted_distance = 0.5 * self.acceleration * (time_to_max_speed ** 2) \
                           + self.max_velocity * extra_time

      min_distance_to_stop = (speed**2)/(2*(self.acceleration+1))

      if time < self.early_start_time:
        if predicted_distance < 0.85*distance:
          action = 2
        elif predicted_distance < 1.2*distance:
          action = 1
        else:
          action = 0
      elif min_distance_to_stop > 0.9 * (distance - self.early_stopping_distance):
        action = 0
      else:
        action = 2

      return action


  def observe_first(self, timestep: dm_env.TimeStep):
     pass

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    pass

  def update(self):
    pass



class Trajectory:

  def __init__(self, model, accelerations, distance, velocity, last_remain_time):
    self.model = model
    self.accelerations =  accelerations
    self.distance = distance
    self.velocity = velocity
    self.current_index = 0
    self.last_remain_time = last_remain_time

  def get_index(self):
    return self.current_index

  def set_last_remain_time(self, last_remain_time):
    self.last_remain_time = last_remain_time

  def get_last_remain_time(self):
    return self.last_remain_time

  def get_acceleration(self):
    self.current_index += 1
    return np.array(self.accelerations[0, min(len(self.accelerations)-1, self.current_index)].x+1)

  def update(self, model, accelerations, distance, velocity, last_remain_time):
    self.model = model
    self.accelerations = accelerations
    self.distance = distance
    self.velocity = velocity
    self.current_index = 0
    self.last_remain_time = last_remain_time




import time
if __name__ == "__main__":
  grl = MultiGurobiLpActor(0.2, 22.0)
  #step_data = np.array([  6.52528,    53.8,       378.66312,     6.3916183,  41.4,       362.58725  ], dtype=np.float32)
  for i in range(2):
    #step_data = np.array([6.52528, 22+i, 378.66312, 6.3916183, 20+i, 362.58725], dtype=np.float32)
    grl.set_id(i)
    step_data = np.array([ 9.422222,  4.3,      55.93694], dtype=np.float32)
    s = time.time()
    grl.select_action(step_data)
    print(time.time()-s)
  print("Done")
  input()



