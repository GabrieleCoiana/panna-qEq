###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name=name,
                                        shape=shape,
                                        initializer=initializer,
                                        dtype=dtype,
                                        trainable=trainable)
    return var


def _variable_random_uniform(name, shape, limit, trainable=True):
    """helper to create a random uniform variable
    """
    dtype = tf.float32
    var = _variable_on_cpu(name,
                           shape,
                           tf.random_uniform_initializer(minval=-limit,
                                                         maxval=limit,
                                                         dtype=dtype),
                           trainable=trainable)
    return var
