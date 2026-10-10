import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial
from panna.jax.MLP_jax import PANNA_MLP
from panna.jax.MLP2_jax import PANNA_MLP2
from panna.jax.MLP_jax_LR_mod import PANNA_MLP_LR_mod

from mendeleev import element


class make_model:
    def __init__(self, parameters, pre, **kwargs):
        if parameters['model_type']=='MLP':
            self.model = PANNA_MLP
        elif parameters['model_type']=='MLP2':
            self.model = PANNA_MLP2
        elif parameters['model_type']=='MLP_LR_mod':
             self.model = PANNA_MLP_LR_mod
        else:
            raise ValueError('Unknown model type.')
        # Creating a list of variables to be kept fixed
        if parameters['fixed_descr']:
            parameters['fixed_variables'].append(self.model.name+'/~/h_kpre')
        self.parameters = parameters
        self.pre = pre
        
        print('INFO - Model chosen: ', parameters['model_type'])
        if 'LR' in parameters['model_type']: # then it's a long range model
            print('INFO - El kernels are precomputed ', parameters['precompute_ewald_sums'])
            print('INFO - PBC is ', parameters['pbc'])  
            
            print('INFO - Gauss charges is ', *parameters['gaussian_charges'])
            species_list = parameters['species']
            if parameters['gaussian_charges'][0] == 'covalent':
                radii = [element(x).covalent_radius * 0.01 * np.sqrt(2.0) for x in species_list]
            elif parameters['gaussian_charges'][0] == 'ionic':
                radii = [element(x).ionic_radii[0].ionic_radius * 0.01 * np.sqrt(2.0) for x in species_list]
            elif parameters['gaussian_charges'][0] in ['False', b'False']:
                radii = [0. for x in species_list]
            else:
                radii = [float(x) for x in parameters['gaussian_charges']]
            print('INFO -   therefore the radii are ', np.round(radii, 3))
            
            print('INFO - chi offsets ', *parameters['chi0_offsets'])
            print('INFO - J offsets ',  *parameters['J0_offsets'])
            print('INFO - kappa offsets ',  *parameters['kappa0_offsets'])
        
        
    def __call__(self, x):
        model = self.model(self.parameters, self.pre)
        
        return model(x)


