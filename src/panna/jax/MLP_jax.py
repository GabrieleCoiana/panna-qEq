import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial
from .neural_network import HKpre, atom_Linear, gather_forces


def E_func(x, sp, layers):
    for i,l in enumerate(layers):
        x = l(sp,x)
    return x[0]

def pairs2E(disp, batch, dfunc, layers):
    des = dfunc(disp, batch)
    Ei = jax.vmap(E_func, in_axes=(0,0,None))(des,batch['species'],layers)
    return jnp.sum(Ei), Ei

# Species resolved MLP
# Handling pairs of neighs as input
class PANNA_MLP(hk.Module):
    def __init__(self, parameters, pre, **kwargs):
        super().__init__()
        self.name = 'panna_mlp'
        # Disabling derivative in the gvect function
        pre.compute_dgvect = False
        self.dfunc = HKpre(pre, parameters, self.name)
        self.arch = [self.dfunc.dsize]+parameters['mlp_arch']
        self.nsp = len(parameters['species'])
        self.layers = []
        self.offsets = jnp.array(parameters['offsets'])
        self.forces = parameters['forces_cost']>0.0
        self.stress = parameters['stress_cost']>0.0
        # To handle species specific/common mlp weights
        sp_weights = parameters['mlp_sp_weights']
        if sp_weights:
            sp_weights = [s.strip() for s in sp_weights.split(':')]
        else:
            sp_weights = ['s']*len(self.arch)
        # To handle frozen layers
        trainable = parameters['mlp_trainable']
        if trainable:
            trainable = [{'T':True, 'F':False}[t.strip()] for t in trainable.split(':')]
        else:
            trainable = [True]*len(self.arch)

        # Special case if we just need the descriptor
        self.descr_only = parameters['task']=='descr'

        for i in range(len(self.arch)-1):
            if i<len(self.arch)-2:
                act = parameters['mlp_act']
            else:
                act = 'lin'
            self.layers.append(atom_Linear(self.nsp,self.arch[i],self.arch[i+1],act,
                               sp_weights[i],parameters['sp_embedding']))
            if not trainable[i]:
                fixvar = self.name+'/~/'+self.layers[-1].name
                if fixvar not in parameters['fixed_variables']:
                    parameters['fixed_variables'].append(fixvar)


    def __call__(self, batch):
        if self.descr_only:
            return self.dfunc(batch['nn_vecs'], batch)

        if self.forces:
            # Computing directly E and dE/dpair
            (Esum, Ei), Fterms = jax.value_and_grad(pairs2E, has_aux=True)(
                                 batch['nn_vecs'], batch, self.dfunc, self.layers)
        else:
            Esum, Ei = pairs2E(batch['nn_vecs'], batch, self.dfunc, self.layers)
        nats = jnp.shape(batch['species'])[0]
        bsize = jnp.shape(batch['nats'])[0]

        Ei += self.offsets[batch['species']]
        # Summing the right elements (indices inde in the batch)
        dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), 
                                                  inserted_window_dims=(0,),
                                                  scatter_dims_to_operand_dims=(0,))
        E = jax.lax.scatter_add(jnp.zeros(bsize),jnp.expand_dims(batch['inde'],-1),Ei,
                                dimnums,mode='drop')
        
        # jax.debug.print('{} {} {}', batch['inda'], batch['indb'], batch['nn_vecs'][0])
        # jax.debug.print('{} ', batch['nn_vecs'])


        if not self.forces:
            return [E]

        dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,), 
                                                  inserted_window_dims=(0,),
                                                  scatter_dims_to_operand_dims=(0,))
        # Gathering forces on atoms a of pairs
        F = jax.lax.scatter_add(jnp.zeros((nats,3)),
                                jnp.expand_dims(batch['inda'],-1),Fterms,
                                dimnums,mode='drop')
        # Gathering forces on atoms b of pairs, with other sign
        F -= jax.lax.scatter_add(jnp.zeros((nats,3)),
                                 jnp.expand_dims(batch['indb'],-1),Fterms,
                                 dimnums,mode='drop')
        
        # jax.debug.print('F {} \n', F[0])
        
        # print(jnp.zeros((nats,3)).shape, 'operand')
        # print(batch['indb'], 'indices')
        # print(Fterms, 'updates')
        # print(F, 'iii')
        # des = self.dfunc(batch['nn_vecs'], batch)
        if self.stress:
            dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,2),
                                                      inserted_window_dims=(0,),
                                                      scatter_dims_to_operand_dims=(0,))

            virial_pairs = jnp.einsum("pa,pb->pab", batch["nn_vecs"], Fterms)
            virial = jax.lax.scatter_add(
                                    jnp.zeros((bsize,3,3)),
                                    jnp.expand_dims(batch['inds'],-1),
                                    virial_pairs,
                                    dimnums,
                                    mode='drop'
                                    )
            volumes = jnp.linalg.det(batch['latvecs'])[:,None,None] # (bsize,1,1)
            stress = -virial / volumes
            return [E, F, stress]
        
            
        
        return [E, F, [0]]

