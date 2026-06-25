"""Force losses related operations."""

from typing import Union
from typing import Optional
from typing import Tuple
from typing import Callable

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.losses import Loss


def _compute_sum_f_diff(forces_true, forces_pred):
    r"""Quadratic difference per element of the batch.

    ..math:
       \sum_{nat}\sum_1^3 (f_{true} - f_{pred})^2
    """
    batch_delta_f = tf.reduce_sum((forces_pred-forces_true)**2,axis=[1])
    return batch_delta_f

def _compute_sum_exp_f_diff(forces_true, forces_pred):
    r"""Exponential of quadratic difference per element of the batch.

    ..math:
       \sum_{nat} \sum_1^3 exp ( (f_{true} - f_{pred})^2 )
    """
    batch_exp_delta_f = tf.reduce_sum(tf.exp((forces_pred-forces_true)**2),axis=[1])
    return batch_exp_delta_f

def _resolve_weight_function(name: Union[str, None]) -> Callable:
    """Weighting possibilities for every example.

    Parameters
    ----------
    name: of the weigh system, possible values are
      - n_atoms, the weights are computed as 1/n_atoms
      - n_atoms2, the weights are computed as 1/n_atoms**2
      - 3n_atoms2, the weights are computed as 1/(3 * n_atoms**2)
      - None, No weighting is applied
    Returns
    -------
    TF function to perform the required operation
    """
    if name == 'n_atoms':
        return lambda x: tf.divide(1, x)
    elif name == 'n_atoms2':
        return lambda x: tf.divide(1, tf.square(x))
    elif name == '3n_atoms2':
        return lambda x: tf.divide(1, tf.square(3 * x))
    elif name == '3n_atoms':
        return lambda x: tf.divide(1, 3 * x)
    else:
        return None


def _safe_assign_weights(weight, name) -> Callable:
    if weight is not None:
        raise ValueError('force_weights can not be set '
                         'with the current floss_function')
    return _resolve_weight_function(name)


def get_force_loss(name: str, weight: Optional[str] = None,
                    reduction: Optional[str] = 'mean') -> Tuple[Loss, Callable]:
    """Recover the force loss by combining the name and the correct weighting mechanism.

    Parameters
    ----------
    name: name of the loss function
    weight: force a weighting mechanism
    reduction: sum or mean reduction strategy

    Returns
    -------
    Loss function, weighting function
    """
    weight = _resolve_weight_function(weight)
    if name == "quad":
        loss = _compute_sum_f_diff
    elif name == "exp_quad":
        loss = _compute_sum_exp_f_diff
    elif name == "quad_atom":
        loss = _compute_sum_f_diff
        weight = _safe_assign_weights(weight, 'n_atoms')
    elif name == "quad_percomp":
        loss = _compute_sum_f_diff
        weight = _safe_assign_weights(weight, '3n_atoms')
    else:
        raise ValueError(f'{name} force loss not found')    

    if reduction=='mean':
        redkey = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    elif reduction=='sum':
        redkey = tf.keras.losses.Reduction.SUM
    else:
        raise ValueError(f'Unknown forces reduction strategy: {reduction}.')
        
    return LossFunctionWrapper(loss, reduction=redkey), weight
