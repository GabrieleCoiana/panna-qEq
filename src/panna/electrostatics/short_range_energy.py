# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:05:43 2025

@author: Gabriele Coiana
"""
import numpy as np
import jax.numpy as jnp
import jax

def one_over_ralpha(batch, Aij, alpha=6):
    real_nats_tot = jnp.shape(batch['indf'])[0]
    nats = jnp.shape(batch['species'])[0]
    bsize = jnp.shape(batch['nats'])[0]
    nn_r = batch['nn_r']
    nn_vecs = batch['nn_vecs']
    inda = jnp.expand_dims(batch['inda'],-1)
    indb = jnp.expand_dims(batch['indb'],-1)
    inde = jnp.expand_dims(batch['inde'][0:real_nats_tot], 1)
    dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,),scatter_dims_to_operand_dims=(0,))

    
    E0ij = 1/2 * Aij * 1/nn_r**alpha
    
    # this first scatter_add takes a central atom i and sums up all pairwise 
    # contributions to the sr energy coming from atoms j within the cutoff.
    # The 1/Rij are contained in the flat nn_r array, and inda contains the index
    # of the central atom. For example
    #  nn_r    inda    indb
    #  1/R00     0       0
    #  1/R01     0       1
    #   ...     ...     ...
    # so, the first scatter_add does Ei = \sum_j 1/Rij^{12}, for all i in the batch.
    E0i = jax.lax.scatter_add(jnp.zeros(nats),inda,E0ij,
                            dimnums,mode='drop') # shape: (Natspad,)
    
    
    # # The second scatter_add decides which atoms belong to which configuration,
    # # and adds the contributions of all i's within the configuration.
    # # E^{sr} = 1/2 \sum_i Ei = 1/2 \sum_{ij} 1/Rij^{12}
    # E0 = jax.lax.scatter_add(jnp.zeros(bsize),inde,E0i,
    #                         dimnums,mode='drop') # shape: (bsize,)
    # # E0 is the sr energy of the configurations in the batch.
    
    # The forces are computed by using Franco's code. Consider the x-component of 
    # the force on atom k. Fxk = - d / d x_k E^{sr}. Now the force kernel contains
    # terms like 1/2 * 12 * x_{kj} / Rkj^{14}. I call these terms Fterms. 
    Fterms0 = alpha/2 * Aij[:,None] * nn_vecs / nn_r[:,None]**(alpha+2) # shape: (Natspad, 3)
    
    # # Now, the derivative contains one term x_{kj} with the plus, and one term 
    # # -x_{jk} with the minus. Hence, the Fterms can be given to the function
    # # gather_forces, which is also used for the Fterms coming from the NN.
    # operand = jnp.zeros((nats,3)) # shape: (Natspad, 3)
    # F0 = gather_forces(operand, inda, indb, Fterms0) # shape: (Natspad, 3)

    return E0i, Fterms0


def morse_stretch(batch, Dsisj, gsisj, r0sisj, rcut3):
    nats = jnp.shape(batch['species'])[0]
    spa, spb, inda, indb  = batch['sp_a'], batch['sp_b'], batch['inda'], batch['indb']  # shape: (Npairs,)
    rij, xyzij = batch['nn_r'], -batch['nn_vecs'] # (Npairs,), (Nparis,3)
    # NOTICE: nn_vecs are defined as posj - posi, but we want posi - posj!
    dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,),scatter_dims_to_operand_dims=(0,))


    spa_plus_spb = spa + spb
    Dij = Dsisj[spa_plus_spb] # shape: (Npairs,)
    gij = gsisj[spa_plus_spb] # shape: (Npairs,)
    r0ij = r0sisj[spa_plus_spb] # shape: (Npairs,)

    # exponents
    alpha = gij * (1 - rij / r0ij)
    beta = gij  * (1 - rcut3 / r0ij)

    # energy
    Eij = 1/2 * Dij * (jnp.exp(alpha) - 2*jnp.exp(0.5 * alpha))
    Elimij = 1/2 *  Dij * (jnp.exp(beta) - 2*jnp.exp(0.5 * beta))
    # E = 1/2 * np.sum(Eij - Elimij)
    # this first scatter_add takes a central atom i and sums up all pairwise 
    # contributions to the sr energy coming from atoms j within the cutoff.
    # The 1/Rij are contained in the flat nn_r array, and inda contains the index
    # of the central atom. For example
    #  nn_r    inda    indb
    #  1/R00     0       0
    #  1/R01     0       1
    #   ...     ...     ...
    # so, the first scatter_add does Ei = \sum_j 1/Rij^{12}, for all i in the batch.
    E0i = jax.lax.scatter_add(jnp.zeros(nats),inda[:,None],Eij-Elimij,
                            dimnums,mode='drop') # shape: (Natspad,)
    

    # forces
    kernelij = Dij* (gij / r0ij * jnp.exp(alpha) - gij / r0ij * jnp.exp(0.5*alpha) )# shape: (Npairs,)
    drij_dxk = xyzij / rij[:,None] # shape: (Npairs, 3)
    Fterms0 = 0.5 * kernelij[:,None] * drij_dxk

    # operand = np.zeros((batch['ntot'],3)) # shape: (Natspad, 3)
    # F = gather_forces(operand, inda[:,None], indb[:,None], Fterms) # shape: (Natspad, 3)
    # F = np.array(F)
    return E0i, Fterms0

