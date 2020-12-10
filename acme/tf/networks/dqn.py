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

"""A standard Deep-Q learning feed forward network.
"""

from typing import Sequence

import sonnet as snt
import tensorflow as tf


class DQN(snt.Module):
  """A Duelling MLP Q-network."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Sequence[int],
  ):
    super().__init__(name='q_network')

    self.q_network = snt.nets.MLP([*hidden_sizes, num_actions])

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of the duelling network.

    Args:
      inputs: 2-D tensor of shape [batch_size, embedding_size].

    Returns:
      q_values: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Compute value
    advantages = self.q_network(inputs)  # [B, A]

    q_values = advantages  # [B, A]

    return q_values


class DQNSplit(snt.Module):
  """A Duelling MLP Q-network."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Sequence[int],
      split: int = 3
  ):
    super().__init__(name='q_network_split')

    self.q_network1 = snt.nets.MLP([*hidden_sizes, num_actions])
    self.q_network2 = snt.nets.MLP([*hidden_sizes, num_actions])

    #self.attention = tf.Variable([1,1,1], trainable=False)
    self.split = split

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of the duelling network.

    Args:
      inputs: 2-D tensor of shape [batch_size, embedding_size].

    Returns:
      q_values: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Split input into two parts
    input_1, input_2 = tf.split(inputs, [self.split,inputs.shape[1] - self.split],1)
    q_out_1 = self.q_network1(input_1)  # [B, A]
    q_out_2 = self.q_network2(input_2)  # [B, A]

    q_values = q_out_1 + q_out_2

    return q_values

class DQNAttention(snt.Module):
    """A Duelling MLP Q-network."""

    def __init__(
            self,
            num_actions: int,
            hidden_sizes: Sequence[int],
            split: int = 1
    ):
      super().__init__(name='q_network_split')

      self.q_network1 = snt.nets.MLP([*hidden_sizes, num_actions])
      self.q_network2 = snt.nets.MLP([*hidden_sizes, num_actions])

      self.attention1 = tf.Variable([1,1,1], trainable=False)
      self.attention2 = tf.Variable([0,0,0], trainable=False)
      self.split = split
      self.threshold = 15

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
      """Forward pass of the duelling network.

      Args:
        inputs: 2-D tensor of shape [batch_size, embedding_size].

      Returns:
        q_values: 2-D tensor of action values of shape [batch_size, num_actions]
      """

      # Split input into two parts
      input_1, input_2 = tf.split(inputs, [self.split, inputs.shape[1] - self.split], 1)

      q_out_1 = self.q_network1(input_1)  # [B, A]
      q_out_2 = self.q_network2(input_2)  # [B, A]

      sliced = tf.slice(inputs, [0, 1], inputs.shape[1]-1, 1)
      condtion = tf.less(sliced, self.threshold)

      weights1 = tf.where(condtion, self.attention2, self.attention1)
      weights2 = tf.where(condtion, self.attention1, self.attention2)

      q_values = weights1 * q_out_1 + weights2 * q_out_2

      return q_values