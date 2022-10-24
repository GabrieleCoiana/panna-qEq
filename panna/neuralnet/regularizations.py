###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import tensorflow as tf
from tensorflow.python.framework import ops


def l1l2_regularizations(wscale_l1, wscale_l2, bscale_l1, bscale_l2):
    """Apply required regularizator.

    Args:
        w : weights, b : biases
        l1 : norm one prefactor if zero nothing gets applyed
        l2 : norm two prefactor if zero nothing gets applyed

    Returns:
        A scalar representing the overall regularization penalty for W
        A scalar representing the overall regularization penalty for B
    """

    w_regularizer_op = tf.contrib.layers.l1_l2_regularizer(wscale_l1,
                                                           wscale_l2,
                                                           scope=None)
    b_regularizer_op = tf.contrib.layers.l1_l2_regularizer(bscale_l1,
                                                           bscale_l2,
                                                           scope=None)

    wcol = tf.get_collection(ops.GraphKeys.WEIGHTS)
    wcol = [x for x in wcol if not ('norm' in x.name)]
    w_l_norm_sum = tf.contrib.layers.\
        apply_regularization(w_regularizer_op, wcol)

    bcol = tf.get_collection(ops.GraphKeys.BIASES)
    bcol = [x for x in bcol if not ('norm' in x.name)]
    b_l_norm_sum = tf.contrib.layers.\
        apply_regularization(b_regularizer_op, bcol)

    all_norm_sum = tf.add(w_l_norm_sum,
                          b_l_norm_sum,
                          name="4.Regularization_loss")
    tf.add_to_collection('losses', all_norm_sum)

    return w_l_norm_sum, b_l_norm_sum
