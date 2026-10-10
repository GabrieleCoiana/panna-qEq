"""Energy losses related operations."""

from typing import Optional
from typing import Tuple
from typing import Callable

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.losses import Loss

def quad(energies_true, energies_pred, **kwargs):
    r"""Quadratic difference per element of the batch.

    ..math:
       \sum_{batch} (e_{true} - e_{pred})^2
    """
    batch_delta_e = energies_pred - energies_true
    return tf.math.square(batch_delta_e)

def exp_quad(energies_true, energies_pred, **kwargs):
    r"""Quadratic difference per element of the batch.

    ..math:
       \sum_{batch} \exp(prefactor * (e_{true} - e_{pred})^2 )
    """
    if kwargs['prefact']:
        pref = kwargs['prefact']
    else:
        pref = 1.0
    batch_delta_e = energies_pred - energies_true
    return tf.exp(pref*tf.math.square(batch_delta_e))


# Creating out own loss class to modify the call.
# This is needed to weight the exponential losses per atom,
# because we cannot use the standard weight which is applied after the function,
# but only y_true and y_pred are passed to a standard loss,
# So instead if needed we preprocess them here before the call
class MyLossFunctionWrapper(LossFunctionWrapper):
    def __init__(self, fn,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name=None,
                 **kwargs):
        super().__init__(fn, reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        if self._fn_kwargs['pre_normalize']:
            if self._fn_kwargs['pre_normalize']=='n_atoms':
                norm = 1./tf.cast(kwargs['n_atoms'], tf.float32)
            else:
                norm = 1.
            y_true *= norm
            y_pred *= norm 
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

def _resolve_weight_function(name):
    if name == 'n_atoms':
        return lambda x: tf.divide(1, x)
    elif name == 'n_atoms2':
        return lambda x: tf.divide(1, tf.square(x))
    else:
        return None


def _safe_assign_weights(weight, name):
    if weight is not None:
        raise ValueError('energy_weights can not be set '
                         'with the current loss_function')
    return _resolve_weight_function(name)


def get_energy_loss(name: str, weight: Optional[str] = None,
                    reduction: Optional[str] = 'mean',
                    **kwargs) -> Tuple[Loss, Callable]:
    """Recover the energy loss by combining the name and the correct weighting mechanism.
    Where possible, weights are introduced through the "weights_function" in the standard TF way
    For other cases (e.g. exp) where the weighting happens within the function, the number
    or atoms needs to be passed to the function itself.

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
    kwargs['pre_normalize'] = None
    if name == 'quad':
        loss = quad
    elif name == "exp_quad":
        loss = exp_quad
    elif name == 'quad_atom':
        loss = quad
        weight = _safe_assign_weights(weight, 'n_atoms2')
    elif name == "quad_std_atom":
        loss = quad
        weight = _safe_assign_weights(weight, 'n_atoms')
    elif name == "exp_quad_atom":
        loss = exp_quad
        kwargs['pre_normalize'] = 'n_atoms'
    elif name == "quad_exp_tanh_atom":
        raise NotImplementedError("TODO")
    elif name == "quad_exp_tanh":
        raise NotImplementedError("TODO")
    elif name == "quad_atom_exp_tanh_atom":
        raise NotImplementedError("TODO")
    else:
        raise ValueError(f'{name} energy loss not found')

    if reduction=='mean':
        redkey = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    elif reduction=='sum':
        redkey = tf.keras.losses.Reduction.SUM
    else:
        raise ValueError(f'Unknown energy reduction strategy: {reduction}.')

    return MyLossFunctionWrapper(loss,reduction=redkey,**kwargs), weight
