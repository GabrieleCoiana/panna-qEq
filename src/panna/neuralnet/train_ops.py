###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import tensorflow as tf


def train_neural_network(loss, global_step, lr, beta1, beta2, adam_eps, clip_value):
    """Train NN model, optimization step.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Parameters
    ----------
      loss: quantity to minimize
      global_step: Integer Variable counting the number of training steps
                   processed.
      lr : learning rate

    Returns
    -------
      train_op: op for training.
    """

    opt = tf.train.AdamOptimizer(
        learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=adam_eps)
    # emine: hardcoded the gradient clipping for now.
    grad_vars = opt.compute_gradients(loss)
    if clip_value != 0.0:
        capped_grad_vars = [(tf.clip_by_value(grad, -clip_value, clip_value),
                             var) for grad, var in grad_vars]
        apply_gradient_op = opt.apply_gradients(
            capped_grad_vars,
            global_step=global_step,
            name='gradient_application')
    else:
        apply_gradient_op = opt.apply_gradients(
            grad_vars, global_step=global_step, name='gradient_application')
    # side note:
    # "compute_gradient" and "apply_gradients" together form what we call
    # the "minimize" function; minimize just does both the operations together
    # For more info: AdamOptimizer inherited form Optimizer class.
    # There you can find all those methods.

    dep = [apply_gradient_op]

    with tf.control_dependencies(dep):
        train_op = tf.no_op(name='train')

    return train_op
