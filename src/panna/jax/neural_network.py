# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:09:36 2026

@author: Gabriele Coiana
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial
import haiku as hk

# Class to handle preprocessing with trainable parameters
class HKpre(hk.Module):
    def __init__(self, pre, parameters, name):
        super().__init__()
        self.dfunc = pre.jax_g
        self.dfunc_postg = pre.jax_g_sum
        self.varinfo = pre.var_info_jax()
        self.name = name
        # If species cutoff is requested, we build here the matrix updating terms
        if parameters['sp_cutoff']:
            spl = parameters['species']
            nsp = len(spl)
            sp_Rc = parameters['cutoff']*np.ones((nsp,nsp,1))
            for s in parameters['sp_cutoff'].split(':'):
                sp1, sp2, rc = [t.strip() for t in s.split(',')]
                sp_Rc[spl.index(sp1),spl.index(sp2)] = rc
        parameters['clipping_vars'] = []
        if not parameters['fixed_descr']:
            # Here we create a list of parameters to be clipped
            # We report name1 and name2 to identify the variable to be clipped
            # We have different types of clipping:
            # type1 has structure: [1, name1, name2, min, max]
            #       and will keep name1.name2 in [min, max]
            # type2 has structure: [2, name1, name2, name3, max]
            #       and will keep name1.name2 in [0, max-name1.name3]
            #       (on a per-element basis)
            # type3 has structure: [3, name1, name2, name3, name4, max_matrix]
            #       where max_matrix has [nsp, nsp] shape
            #       and will set name1.name2 to 0 if name1.name3+name1.name4 > max_matrix
            #       (on a per element and per species basis)
            for i,v in enumerate(self.varinfo):
                # If sigma is const, we just set min/max for clipping
                if not parameters['learnsig']:
                    if v['name'].split('_')[0]=='cent':
                        parameters['clipping_vars'].append([1, self.name+'/~/h_kpre',
                                    'DV_'+v['name'],v['min'],v['max']])
                # Otherwise, we set centers to depend on sigma, and sigma to its limits
                else:
                    if v['name'].split('_')[0]=='cent':
                        parameters['clipping_vars'].append([2, self.name+'/~/h_kpre',
                                    'DV_'+v['name'],'DV_'+v['name'].replace('cent','sig'),parameters['cutoff']])
                    if v['name'].split('_')[0]=='sig':
                        parameters['clipping_vars'].append([1, self.name+'/~/h_kpre',
                                    'DV_'+v['name'],v['min'],v['max']])
                    if parameters['sp_cutoff']:
                        if v['name'].split('_')[0]=='spw':
                            parameters['clipping_vars'].append([3, self.name+'/~/h_kpre',
                                        'DV_'+v['name'],'DV_'+v['name'].replace('spw','cent'),
                                        'DV_'+v['name'].replace('spw','sig'), sp_Rc])
                        

        # Handling species embedding
        if parameters['sp_emb_file']:
            self.sp_emb_size = parameters['sp_emb_size']
            self.sp_embedding = jnp.asarray(parameters['sp_embedding'])
        else:
            self.sp_embedding = None

        self.dsize = pre.gsize
        self.deriv = parameters['forces_cost'] > 0.0

    def __call__(self, nn_vecs, inputs):
        dvars = []
        for i,v in enumerate(self.varinfo):
            if v['type']=='norm':
                dvars.append(hk.get_parameter('DV_'+v['name'],
                    v['shape'],init=hk.initializers.RandomNormal(v['std'],v['mean'])))
            elif v['type']=='unif':
                dvars.append(hk.get_parameter('DV_'+v['name'],
                    v['shape'],init=hk.initializers.RandomUniform(v['min'],v['max'])))
        if self.sp_embedding is not None:
            dvars.append(self.sp_embedding)

        # We call the function first to set the params, then vmap over the inputs
        dwithvars = partial(self.dfunc, dvars)
        modinputs = {'nn_vecs':nn_vecs,
                      'nn_r':jnp.sqrt(jnp.sum(nn_vecs**2,axis=-1)),
                      'sp_a':inputs['sp_a'],
                      'sp_b':inputs['sp_b'],
                      }

        # We vmap over pairs, obtain dicts of g and dg contributions from that pair
        gpairs = jax.vmap(dwithvars, in_axes=({i:0 for i in modinputs},))(modinputs)
        nats = jnp.shape(inputs['inde'])[0]

        gpatom = {}
        for gpk in gpairs.keys():
            up_dim = list(jnp.shape(gpairs[gpk]))[1:]
            up_tuple = tuple(np.arange(1,len(up_dim)+1))
            dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=up_tuple, 
                                                      inserted_window_dims=(0,),
                                                      scatter_dims_to_operand_dims=(0,))
            gpatom[gpk] = jax.lax.scatter_add(jnp.zeros([nats]+up_dim),
                                jnp.expand_dims(inputs['inda'],-1),gpairs[gpk],
                                dimnums,mode='drop')

        # We vmap over atoms, get g contribution for that atom
        g = jax.vmap(self.dfunc_postg, 
                      in_axes=({i:0 for i in gpatom},))(gpatom)
        
        return g

class atom_Linear(hk.Module):
    def __init__(self, nsp, insize, outsize, act, sp_w, sp_emb=None):
        super().__init__()
        self.nsp = nsp
        self.insize = insize
        self.outsize = outsize
        self.sp_w = sp_w
        if sp_w=='e':
            self.emb = jnp.asarray(sp_emb)
            self.emb_size = sp_emb.shape[1]
        if act=='lin':
            self.act = None
        elif act=='exp':
            self.act = lambda x: jnp.exp(-x*x)
        elif act=='tanh':
            self.act = jnp.tanh
        elif act=='silu':
            self.act = jax.nn.silu
        elif act=='lorentz':
            self.act = lambda x: 1./(1.+x*x)
        elif act=='relu':
            self.act = jax.nn.relu
        elif act=='gelu':
            self.act = jax.nn.gelu
        else:
            raise ValueError('Unknown activation')

    def __call__(self, species, inputs):
        if self.sp_w=='s':
            # Weights for each species, Glorot initialization
            w = hk.get_parameter('w',(self.nsp,self.outsize,self.insize),
                    init=hk.initializers.RandomNormal(jnp.sqrt(2./(self.insize+self.outsize)),0.0))
            # RandomNormal(1e-2, 0.0))#
            b = hk.get_parameter('b',(self.nsp,self.outsize),init=hk.initializers.Constant(0.0))
            out = jnp.matmul(w[species],inputs) + b[species] 
        elif self.sp_w=='c':
            # Not selecting for species
            w = hk.get_parameter('w',(self.outsize,self.insize),
                    init=hk.initializers.RandomNormal(jnp.sqrt(2./(self.insize+self.outsize)),0.0))
            b = hk.get_parameter('b',(self.outsize,),init=hk.initializers.Constant(0.0))
            out = jnp.matmul(w,inputs) + b
        elif self.sp_w=='e':
            # Using second dim for the embedding
            w = hk.get_parameter('w',(self.outsize,self.emb_size,self.insize),
                    init=hk.initializers.RandomNormal(jnp.sqrt(2./(self.insize+self.outsize)),0.0))
            b = hk.get_parameter('b',(self.emb_size,self.outsize,),init=hk.initializers.Constant(0.0))
            out = jnp.matmul(jnp.matmul(self.emb[species],w),inputs) + jnp.matmul(self.emb[species],b)
        if self.act:
            out = self.act(out)
        return out


def gather_forces(operand, indicesa, indicesb, Fterms):
    dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,), 
                                              inserted_window_dims=(0,),
                                              scatter_dims_to_operand_dims=(0,))
    # Gathering forces on atoms a of pairs
    F = jax.lax.scatter_add(operand,
                            indicesa,Fterms,
                            dimnums,mode='drop')
    # Gathering forces on atoms b of pairs, with other sign
    F -= jax.lax.scatter_add(operand,
                             indicesb,Fterms,
                             dimnums,mode='drop')
    return F


# def pairs2E(disp, batch, dfunc, layers):
#     # jax.debug.print(' {disp.shape}', disp=disp)
#     des = dfunc(disp, batch)
#     Ei = jax.vmap(E_func, in_axes=(0,0,None))(des,batch['species'],layers)
#     Ei0, Ei1, Ei2 = Ei[:,0], Ei[:,1], Ei[:,2]
#     print('AAAA')
#     return (jnp.sum(Ei0), jnp.sum(Ei1)), (Ei0, Ei1, Ei2)