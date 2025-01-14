import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial
from panna.jax.MLP_jax import PANNA_MLP
# from panna.jax.MLP2_jax import PANNA_MLP2

class make_model:
    def __init__(self, parameters, pre, **kwargs):
        if parameters['model_type']=='MLP':
            self.model = PANNA_MLP
        else:
            raise ValueError('Unknown model type.')
        # Creating a list of variables to be kept fixed
        if parameters['fixed_descr']:
            parameters['fixed_variables'].append(self.model.name+'/~/h_kpre')
        self.parameters = parameters
        self.pre = pre

    def __call__(self, x):
        model = self.model(self.parameters, self.pre)
        return model(x)


