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

"""Convenient base classes for custom networks."""

import abc
from typing import Tuple, TypeVar, Sequence

from acme import types
import sonnet as snt

State = TypeVar('State')


class Module(snt.Module, abc.ABC):
  """A base class for module with abstract __call__ method."""

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    """Forward pass of the module."""


class RNNCore(snt.RNNCore, abc.ABC):
  """An RNN core with a custom `unroll` function."""

  @abc.abstractmethod
  def unroll(self,
             inputs: types.NestedTensor,
             state: State,
             sequence_length: int,
             ) -> Tuple[types.NestedTensor, State]:
    """A custom function for doing static unrolls over sequences.

    This has the same API as `snt.static_unroll`, but allows the user to specify
    their own implementation to take advantage of the structure of the network
    for better performance, e.g. by batching the feed-forward pass over the
    whole sequence.

    Args:
      inputs: A nest of `tf.Tensor` in time-major format.
      state: The RNN core state.
      sequence_length: How long the static_unroll should go for.

    Returns:
      Nested sequence output of RNN, and final state.
    """
class R2D2Network(RNNCore):

  def __init__(self, num_actions: int,
                    lstm_layer_size: int,
                    feedforward_layers : Sequence[int]
  ):
    super().__init__(name='R2D2Network')
    self._net = snt.DeepRNN([
        snt.Flatten(),
        snt.LSTM(lstm_layer_size),
        snt.nets.MLP([*feedforward_layers, num_actions])
    ])

  def __call__(self, inputs, state):
    return self._net(inputs, state)

  def initial_state(self, batch_size: int, **kwargs):
    return self._net.initial_state(batch_size)

  def unroll(self, inputs, state, sequence_length):
    return snt.static_unroll(self._net, inputs, state, sequence_length)