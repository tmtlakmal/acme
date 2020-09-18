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

"""A duelling network architecture, as described in [0].

[0] https://arxiv.org/abs/1511.06581
"""

from typing import Sequence

import sonnet as snt
import tensorflow as tf


class DuellingMLP(snt.Module):
  """A Duelling MLP Q-network."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Sequence[int],
  ):
    super().__init__(name='duelling_q_network')

    self._value_mlp = snt.nets.MLP([*hidden_sizes, 1])
    self._advantage_mlp = snt.nets.MLP([*hidden_sizes, num_actions])

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of the duelling network.

    Args:
      inputs: 2-D tensor of shape [batch_size, embedding_size].

    Returns:
      q_values: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Compute value & advantage for duelling.
    value = self._value_mlp(inputs)  # [B, 1]
    advantages = self._advantage_mlp(inputs)  # [B, A]

    # Advantages have zero mean.
    advantages -= tf.reduce_mean(advantages, axis=-1, keepdims=True)  # [B, A]

    q_values = value + advantages  # [B, A]

    return q_values

class SplitInputDuelling(snt.Module):

  def __init__(self,
               num_actions : int,
               hidden_size : Sequence[int],
               split: int,
               concat_hidden_size : Sequence[int]
  ):
    super(SplitInputDuelling, self).__init__(name="Split_Deulling_Network")

    self.duelling_network = DuellingMLP(num_actions, hidden_size)
    self.extra_network = snt.nets.MLP([*hidden_size, num_actions])
    self.concat_network = snt.nets.MLP([*concat_hidden_size, num_actions])
    self.split = split

  def __call__(self, inputs : tf.Tensor) -> tf.Tensor :
    # action values computed using a duelling network
    # then use a extra network to represent the extra weights
    #combine two network using concat_network
    input_1, input_2 = tf.split(inputs, [self.split,inputs.shape[1] - self.split],1)
    action_values = self.duelling_network(input_1)
    extra_values = self.extra_network(input_2)

    concat_values = tf.concat([action_values, extra_values], 1)
    return self.concat_network(concat_values)


