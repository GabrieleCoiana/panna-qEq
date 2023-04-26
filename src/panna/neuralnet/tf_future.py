"""TF utility not available in older versions."""
# This is not available in TF2.4 (Marconi100)
# hence I copied the calls here

import tensorflow as tf

def scale_loss_for_distribution(loss_value):
    """Scales and returns the given loss value by the number of replicas."""
    num_replicas = (tf.distribute.get_strategy().num_replicas_in_sync)
    if num_replicas > 1:
        loss_value *= (1. / num_replicas)
    return loss_value


def cast_losses_to_common_dtype(losses):
    """Cast a list of losses to a common dtype.

    If any loss is floating-point, they will all be casted to the most-precise
    floating-point loss. Otherwise the losses are not casted. We also skip casting
    losses if there are any complex losses.

    Args:
      losses: A list of losses.

    Returns:
      `losses`, but they have been casted to a common dtype.
    """
    highest_float = None
    for loss in losses:
        if loss.dtype.is_floating:
            if highest_float is None or loss.dtype.size > highest_float.size:
                highest_float = loss.dtype
            elif {loss.dtype, highest_float} == {'bfloat16', 'float16'}:
                highest_float = 'float32'
        if loss.dtype.is_complex:
            return losses  # If we find any complex losses, do not cast any losses
    if highest_float:
        losses = [tf.cast(loss, highest_float) for loss in losses]
    return losses
