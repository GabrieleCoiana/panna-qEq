###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import itertools
import logging
from functools import partial

import numpy as np
import tensorflow as tf
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pass

from .gvect_base import GvectBase, requires_parameters

# logger
logger = logging.getLogger('panna')


class GvectLATTE(GvectBase):
    """ 
      Local Atomic Tensors Trainable Expansion descriptor
      For now implemented for usage with JAX
    """

    name = 'LATTE'
    doi = ''

    _gvect_parameters = {
        # RADIAL_COMPONENTS
        'Rc': lambda self, x: x * self.unit2A,
        'sig': lambda self, x: x * self.unit2A,
        'RsNs': lambda self, x: x,
    }

    def parse_parameters(self, gv_param):
        self.units = gv_param.get('gvect_parameters_unit', 'angstrom')

        Rc = gv_param.getfloat('Rc')
        self.update_parameter('Rc', np.float32(Rc))

        dshape = gv_param.get('descriptor_shape', None)
        dshapel = []
        RsNs = []
        totsh = [0]
        for part in dshape.split(':'):
            elem = part.split(',')
            elem = [e.strip() for e in elem]
            # The number of terms
            RsNs.append(int(elem[0]))
            # The body order (technically this+1)
            m = len(elem)-1
            # The list of possible indices
            inds = list(set([i for ii in elem[1:] for i in ii]))
            # The number of indices
            c = len(inds)
            # The cumulative number of single elements
            if elem[1]=='-':
                totsh.append(totsh[-1]+int(elem[0]))
            else:
                totsh.append(totsh[-1]+int(elem[0])*m*(3**c))
            # For each index of each body, we compute a list of indices
            # wrt which we need to expand a versor to obtain a term of the tensor
            eeinds = []
            # Loop over bodies
            for j,ee in enumerate(elem[1:]):
              einds = []
              # Loop over indices of that body
              for e in ee:
                ti = inds.index(e)
                ai = list(range(c))
                ai.remove(ti)
                # List up to max c, with the index removed
                einds.append(ai)
              eeinds.append(einds)
            # Saving body order, num indices, signature, 
            # and list of expansion indices
            dshapel.append([m,c,','.join(elem[1:]),eeinds])
        self.update_parameter('RsNs', RsNs)
        self.dshape = dshapel
        self.totsh = totsh

        sig = gv_param.get_comma_list_floats('sig')
        self.update_parameter('sig', np.asarray(sig).astype(np.float32))
        self.learnsig = gv_param.getboolean('learnsig', False)
        if self.learnsig and len(sig)!=2:
            raise ValueError('If learnsig, sig should have 2 values (min, max).')

        self.spec_weights = gv_param.get('spec_weights', 'specific')
        self.sp_emb = gv_param.getboolean('neigh_emb', False)

        if gv_param.get('sp_emb_file', None):
            self.sp_emb_size = gv_param.getint('sp_emb_size', 0)
            # Supporting early notation
            if gv_param.getboolean('center_atom_emb', False):
                print("Warning: center_atom_emb is a deprecated keyword")
                print("in the future, please set spec_weights to embedded.")
                self.spec_weights = 'embedded'
            if not self.sp_emb:
                print("Warning: embedding file present, but neigh_emb is False")
                print("currently, this does not enable neighbors embedding.")
        else:
            if self.sp_emb:
                raise ValueError('neigh_emb True requires an embedding file.')
            if self.spec_weights=='embedded':
                raise ValueError('spec_weights embedded requires an embedding file.')
            # self.center_emb = False

        self.pre = gv_param.get_comma_list_floats('pref', np.ones(len(RsNs)))

    @property
    @requires_parameters
    def gsize(self):
        return np.sum(self.RsNs)

    @property
    @requires_parameters
    def gvect(self):
        parameters = {
            'Rc': self.Rc,
            'sig': self.sig,
            'RsNs': self.RsNs
        }
        return parameters

    @requires_parameters
    def __call__(self, key, positions, species, lattice_vectors, **kwargs):
        """
          Not implemented for now, we use this with JAX...
        """
        raise NotImplementedError()

    def var_info_jax(self):
        """Create a list of specifications to create variables
        So that we can create them inside the model for tracking purposes
        """
        RsNs = self.RsNs
        nsp = self.number_of_species
        Rc = self.Rc
        sig = self.sig
        var_dat = []

        if self.sp_emb:
            emb = self.sp_emb_size
        else:
            emb = nsp
        if self.spec_weights=='specific':
            for n, sh in zip(RsNs,self.dshape):
                var_dat.append({'type': 'norm', 'mean':0.0, 'std':1.0,'shape':[nsp,emb,n*sh[0]],'name': 'spw_'+sh[2]})
                var_dat.append({'type': 'unif', 'min':sig[0], 'max':Rc-sig[0], 'shape':[nsp,n*sh[0]],'name': 'cent_'+sh[2]})
                if self.learnsig:
                    var_dat.append({'type': 'unif', 'min':sig[0], 'max':sig[1], 'shape':[nsp,n*sh[0]],'name': 'sig_'+sh[2]})
        elif self.spec_weights=='common':
            for n, sh in zip(RsNs,self.dshape):
                var_dat.append({'type': 'norm', 'mean':0.0, 'std':1.0,'shape':[emb,n*sh[0]],'name': 'spw_'+sh[2]})
                var_dat.append({'type': 'unif', 'min':sig[0], 'max':Rc-sig[0], 'shape':[n*sh[0]],'name': 'cent_'+sh[2]})
                if self.learnsig:
                    var_dat.append({'type': 'unif', 'min':sig[0], 'max':sig[1], 'shape':[n*sh[0]],'name': 'sig_'+sh[2]})
        elif self.spec_weights=='embedded':
            for n, sh in zip(RsNs,self.dshape):
                var_dat.append({'type': 'norm', 'mean':0.0, 'std':1.0,'shape':[2,emb,n*sh[0]],'name': 'spw_'+sh[2]})
                var_dat.append({'type': 'unif', 'min':sig[0], 'max':Rc-sig[0], 'shape':[n*sh[0]],'name': 'cent_'+sh[2]})
                if self.learnsig:
                    var_dat.append({'type': 'unif', 'min':sig[0], 'max':sig[1], 'shape':[n*sh[0]],'name': 'sig_'+sh[2]})
        return var_dat

    # @partial(jax.jit, static_argnums=[0])
    def jax_g(self, weights, x):
        """Operating on one pair at a time.
        x elements is a dict
        """
        gsize = self.gsize
        Rc = self.Rc
        nsp = self.number_of_species

        # THese are now for a single atom
        sp_a = x['sp_a']
        sp_b = x['sp_b']
        nn_vecs = x['nn_vecs']
        rij = x['nn_r']

        G_terms = {}
        dG_terms = {}
        inn_r = 1/(rij+1e-20)
        nn_vers = nn_vecs*inn_r
        for i, (n, (m, c, sign, eeinds)) in enumerate(zip(self.RsNs,self.dshape)):
            sig = self.sig[0]
            if self.learnsig:
                varnum = 3
            else:
                varnum = 2
            if self.spec_weights=='specific':
                # centers need a 1 after for "extra species" in padding
                # otherwise the derivative can explode
                # Species factors get a 0, since we're using direct indexing (can't use OOB)
                c_W = jnp.concatenate([weights[varnum*i+1],jnp.ones((1,m*n))],axis=0)
                if self.sp_emb:
                    sp_W = jnp.concatenate([weights[varnum*i],jnp.zeros((1,self.sp_emb_size,m*n))],axis=0)
                    # If we embed, the emb matrix is passed as the last weight
                    sp_cont = self.pre[i]*(weights[-1][sp_b]@sp_W[sp_a])
                else:
                    sp_W = jnp.concatenate([weights[varnum*i],jnp.zeros((1,nsp,m*n))],axis=0)
                    sp_cont = self.pre[i]*sp_W[sp_a,sp_b]
                spcent = c_W[sp_a]
                if self.learnsig:
                    sig_W = jnp.concatenate([weights[varnum*i+2],jnp.ones((1,m*n))],axis=0)
                    sig = sig_W[sp_a]
            elif self.spec_weights=='common':
                if self.sp_emb:
                    sp_cont = self.pre[i]*(weights[-1][sp_b]@weights[varnum*i])
                else:
                    sp_cont = self.pre[i]*weights[varnum*i][sp_b]
                spcent = weights[varnum*i+1]
                if self.learnsig:
                    sig = weights[varnum*i+2]
            elif self.spec_weights=='embedded':
                if self.sp_emb:
                    emb_pad = jnp.concatenate([weights[-1],jnp.zeros((1,self.sp_emb_size))],axis=0)
                    sp_cont = self.pre[i]*(emb_pad[sp_a]@weights[varnum*i][0])*(emb_pad[sp_b]@weights[varnum*i][1])
                else:
                    sp_cont = self.pre[i]*weights[varnum*i][sp_b]
                spcent = weights[varnum*i+1]
                if self.learnsig:
                    sig = weights[varnum*i+2]
            rbf_cont = jax_RBF(rij,spcent,sig)
            # Special case for 2 body without versor
            if m==1 and sign[0]=='-':    
                G_terms['-'] = sp_cont*rbf_cont
            else:
                # For each body, we multiply by versor for all required indices
                # i.e., for every letter in a given body contribution
                # eeinds has all the other indices, so we expand nn_vers there
                # then we take the product and we stack in tens
                t = []
                for ee in eeinds:
                    v = jnp.ones([3]*c)
                    for e in ee:
                        v *= jnp.expand_dims(nn_vers,e)
                    t.append(v)
                tens = jnp.expand_dims(jnp.stack(t,axis=0),1)
                # We reshape the radial function for each possible tensor dimension
                # Then contract with the tensors
                shape = [m,n]+[1]*c
                frad = sp_cont*rbf_cont
                G_terms[sign] = jnp.reshape(frad,shape)*tens

        return G_terms

    def jax_g_sum(self, x):
        """Doing the summation over indices for each term for g
        """
        Gs = []

        for i, (n, (m, c, sign, eeinds)) in enumerate(zip(self.RsNs,self.dshape)):
            # Special case for 2 body without versor
            if m==1 and sign[0]=='-':
                Gs.append(x['-'])
                continue
            # The common case: product of terms, sum over tensor indices
            Gs.append(jnp.sum(jnp.prod(x[sign],axis=0),axis=np.arange(c)+1)) 

        Gs = jnp.concatenate(Gs, axis=0)
        return Gs

#@jax.jit
def jax_RBF(x,xc,sig):
    """ A simple RBF function of x, centered in x_c and with width sigma,
        with 0 value, 1st and 2nd derivative when (x-x_c) == sigma,
        defined as:
        (1-((x-x_c)/sig)^2)^3 / x_c^2
        and zero outside
    """
    y = (x-xc)/sig
    t = jax.nn.relu(1-y**2)
    return t*t*t/(xc*xc)


#@jax.jit
def jax_dRBF(x,xc,sig):
    """ A simple RBF function of x, centered in x_c and with width sigma,
        with 0 value, 1st and 2nd derivative when (x-x_c) == sigma,
        defined as:
        (1-((x-x_c)/sig)^2)^3 / x_c^2
        and zero outside
        Also returns the derivative.
    """
    y = (x-xc)/sig
    t = jax.nn.relu(1-y**2)
    q = t*t/(xc*xc)
    return t*q, -6*y*q/sig

