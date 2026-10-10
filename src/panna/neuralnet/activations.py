"""Compatibilty layer with old activation functions."""

import tensorflow as tf
from tensorflow.python.ops import math_ops

# def _rbf(b_shape):
#     def _act(in_tensor, weights, _):
#         reshaped_in = tf.tile(tf.expand_dims(in_tensor, -1), [1, 1, b_shape])
#         w_contrib = tf.subtract(reshaped_in, weights, name='w_contrib')
#         exp_arg = tf.reduce_sum(tf.square(w_contrib, name='exp_arg'), 1)
#         return tf.exp(tf.negative(exp_arg), name='activation')
#     return _act


def _gaussian(in_tensor):
    exp_arg = math_ops.square(in_tensor, name='exp_arg')
    return math_ops.exp(math_ops.negative(exp_arg), name='activation')


def _relu(in_tensor):
    return tf.nn.relu(in_tensor, name='relu')


def _tanh(in_tensor):
    return tf.nn.tanh(in_tensor, name='activation')


def get_activation(idx: int):
    """Recover activation from index."""
    if idx == 1:
        activation_name = 'gaussian'
        activation = _gaussian
    elif idx == 2:
        raise NotImplementedError
    elif idx == 3:
        activation_name = 'relu'
        activation = _relu
    elif idx == 4:
        activation_name = 'tanh'
        activation = _tanh
    elif idx == 0:
        activation_name = 'linear'
        activation = None
    else:
        # fallback on keras activations
        activation_name = 'ND'
        activation = idx
    return activation_name, activation
