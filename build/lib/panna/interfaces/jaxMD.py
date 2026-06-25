###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

from panna.jax.parser_jax import trainjax_parameter_parser
from panna.jax.model_jax import make_model
from panna.gvector import GvectLATTE
import numpy as np
import jax_md
import jax
import jax.numpy as jnp
import haiku as hk
import os
import pickle

def PANNAJAX_neighbor_list(displacement_fn, config, species, box,
                           capacity_multiplier=1.25,
                           dr_threshold=0.0):
    # Load configuration
    parameters = trainjax_parameter_parser(config)
    # Create descriptor
    preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
    preprocess.parse_parameters(parameters['descriptor_params'])
    # Make model and apply function
    raw_model = make_model(parameters, pre=preprocess)
    model = hk.without_apply_rng(hk.transform(raw_model))
    modapp = jax.jit(model.apply)
    # Load parameters
    with open(os.path.dirname(config)+'/'+parameters['weights_file'], 'rb') as f:
        model_state = pickle.load(f)
    # Create useful quantities
    nats = len(species)
    nsp = len(parameters['species'])
    mapping = dict(zip(parameters['species'], range(nsp)))
    internal_species = jnp.asarray([mapping[s] for s in species]+[0],dtype=jnp.int32)
    rcut = parameters['cutoff']
    # Create neighbor function, requesting sparse list
    neighbor_fn = jax_md.partition.neighbor_list(displacement_fn, box, rcut, \
                                                 capacity_multiplier=capacity_multiplier, \
                                                 dr_threshold=dr_threshold, \
                                                 format=jax_md.partition.Sparse)

    # The main function calling the model
    def energy_force_fn(coords, neighbors):
        # Creating the input dict
        displ = jax.vmap(displacement_fn)(coords[neighbors.idx[1]],coords[neighbors.idx[0]])
        rads = jnp.sqrt(jnp.sum(displ**2,axis=1))
        inda, indb = neighbors.idx
        data = {
            'species': internal_species[:nats],
            'nats': jnp.asarray([nats]),
            'ntot': nats,
            'inda': inda,
            'sp_a': internal_species[inda],
            'indb': indb,
            'sp_b': internal_species[indb],
            'inde': jnp.zeros(nats,dtype=jnp.int32),
            'nn_vecs': displ,
            'nn_r': rads}
        # Model call, out is [E, F]
        out = modapp(model_state, data)
        return out

    # Compiled function to get only energy
    @jax.jit
    def energy_fn(coords, neighbors):
        return energy_force_fn(coords, neighbors)[0][0]

    # Compiled function to get only force
    @jax.jit
    def force_fn(coords, neighbors):
        return energy_force_fn(coords, neighbors)[1]

    return neighbor_fn, energy_fn, force_fn
