"""Force losses related operations."""

from typing import Union
from typing import Optional
from typing import Tuple
from typing import Callable

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.losses import Loss


def _compute_sum_q_diff(charges_true, charges_pred):
    r"""Quadratic difference per element of the batch.

    ..math:
       \sum_{nat} (q_{true} - q_{pred})^2
    """
    batch_delta_q = tf.reduce_sum((charges_true-charges_pred)**2,axis=-1)
    batch_q_true_sq = tf.reduce_sum(charges_true**2,axis=-1)
    return tf.where(batch_q_true_sq>1e-6, 
                   batch_delta_q, tf.zeros(tf.shape(batch_delta_q)))

def _compute_sum_exp_q_diff(charges_true, charges_pred):
    r"""Exponential of quadratic difference per element of the batch.

    ..math:
       \sum_{nat} exp ( (q_{true} - q_{pred})^2 )
    """
    batch_exp_delta_q = tf.reduce_sum(tf.exp((charges_pred-charges_true)**2),axis=-1)
    batch_q_true_sq = tf.reduce_sum(charges_true**2,axis=-1)
    return tf.where(batch_q_true_sq>1e-6, 
                   batch_exp_delta_q, tf.zeros(tf.shape(batch_exp_delta_q)))

def _resolve_weight_function(name: Union[str, None]) -> Callable:
    """Weighting possibilities for every example.

    Parameters
    ----------
    name: of the weigh system, possible values are
      - n_atoms, the weights are computed as 1/n_atoms
      - None, No weighting is applied
    Returns
    -------
    TF function to perform the required operation
    """
    if name == 'n_atoms':
        return lambda x: tf.divide(1, x)
    else:
        return None


def _safe_assign_weights(weight, name) -> Callable:
    if weight is not None:
        raise ValueError('charge_weights can not be set '
                         'with the current floss_function')
    return _resolve_weight_function(name)


def get_charge_loss(name: str, weight: Optional[str] = None,
                    reduction: Optional[str] = 'mean') -> Tuple[Loss, Callable]:
    """Recover the charge loss by combining the name and the correct weighting mechanism.

    Parameters
    ----------
    name: name of the loss function
    weight: charge a weighting mechanism
    reduction: sum or mean reduction strategy

    Returns
    -------
    Loss function, weighting function
    """
    weight = _resolve_weight_function(weight)
    if name == "quad":
        loss = _compute_sum_q_diff
    elif name == "exp_quad":
        loss = _compute_sum_exp_q_diff
    elif name == "quad_atom":
        loss = _compute_sum_q_diff
        weight = _safe_assign_weights(weight, 'n_atoms')
    else:
        raise ValueError(f'{name} charge loss not found')    

    if reduction=='mean':
        redkey = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    elif reduction=='sum':
        redkey = tf.keras.losses.Reduction.SUM
    else:
        raise ValueError(f'Unknown charges reduction strategy: {reduction}.')
        
    return LossFunctionWrapper(loss, reduction=redkey), weight
