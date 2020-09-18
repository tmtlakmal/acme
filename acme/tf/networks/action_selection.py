from typing import Any, Callable, Iterable
import sonnet as snt
import tensorflow as tf
import tree

### This module is used to run with adaptive epsilon
class GreedyEpsilonWithDecay(snt.Module):
    def __init__(self, layers: Iterable[Callable[..., Any]] = None
                 , name=None):
        super(GreedyEpsilonWithDecay, self).__init__(name=name)
        self._layers = list(layers) if layers is not None else []

    def __call__(self, inputs, epsilon, *args, **kwargs):
        outputs = inputs
        for idx, layer in enumerate(self._layers):
            if idx == 0:
                # Pass additional arguments to the first layer.
                outputs = layer(outputs, *args, **kwargs)
            elif idx == len(self._layers) - 1:
                outputs = layer(outputs, epsilon)
            else:
                outputs = layer(outputs)
        return outputs

class GreedyEpsilonWithDecayRNN(snt.DeepRNN):
    def __init__(self, layers: Iterable[Callable[..., Any]] = None
                 , name=None):
        super(GreedyEpsilonWithDecayRNN, self).__init__(layers, name=name)

    def __call__(self, inputs, prev_state, epsilon):

        current_inputs = inputs
        outputs = []
        next_states = []
        recurrent_idx = 0
        concat = lambda *args: tf.concat(args, axis=-1)
        for idx, layer in enumerate(self._layers):
            if self._skip_connections and idx > 0:
                current_inputs = tree.map_structure(concat, inputs, current_inputs)

            if isinstance(layer, snt.RNNCore):
                current_inputs, next_state = layer(current_inputs,
                                                   prev_state[recurrent_idx])
                next_states.append(next_state)
                recurrent_idx += 1
            elif idx == len(self._layers) - 1:
                current_inputs = layer(current_inputs, epsilon)
            else:
                current_inputs = layer(current_inputs)

            if self._skip_connections:
                outputs.append(current_inputs)

        if self._skip_connections and self._concat_final_output_if_skip:
            outputs = tree.map_structure(concat, *outputs)
        else:
            outputs = current_inputs

        return outputs, tuple(next_states)

