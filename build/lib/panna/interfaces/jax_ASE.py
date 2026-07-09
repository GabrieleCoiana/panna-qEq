###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import os
import pickle
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from ase.neighborlist import neighbor_list
from panna.jax.parser_jax import trainjax_parameter_parser
from panna.jax.model_jax import make_model
from panna.gvector import GvectLATTE

class PANNAJAXCalculator(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        config = None,
        skin = 5.0,
        weights_file=-1,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        self.nl = None
        self.skin = skin
        # Parsing config file
        parameters = trainjax_parameter_parser(config, train=False)
        parameters['stress'] = True # forcing model to compute stress
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        
        # wfolder = parameters['train_dir']+'/models/'
        wfolder = os.path.dirname(config)+'/models/'
        if weights_file == -1:
            last_file = max(
            (f for f in os.listdir(wfolder) if f.startswith("epoch_")),
            key=lambda f: int(f.split("_")[1])
            )
            wfile = wfolder + last_file
        else:
            wfile = wfolder + weights_file
        print('INFO - Getting model', wfile)
        with open(wfile, 'rb') as f:
            self.model_state = pickle.load(f)
        raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(raw_model))
        self.modapp = jax.jit(model.apply)
        
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']#/2.0 + self.skin
        # self.max_pairs = 10
        # self.inda = np.zeros(self.max_pairs,dtype=np.int64)
        # self.indb = np.zeros(self.max_pairs,dtype=np.int64)
        # self.indc = np.zeros((self.max_pairs,3),dtype=np.int64)
        
    def atoms2dict(self, nats, pos, cell, specs, inda, indb, cells):
        data = {}
        data['species'] = specs[:nats]
        data['nats'] = np.asarray([nats])
        data['ntot'] = nats
        data['inda'] = inda
        data['sp_a'] = specs[data['inda']]
        data['indb'] = indb
        data['sp_b'] = specs[data['indb']]
        data['inde'] = np.zeros(nats,dtype=np.int32)
        data['inds'] = np.zeros(len(inda),dtype=np.int32)
        # data['mask'] = np.ones(len(inda),dtype=np.int32)
        cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        data['latvecs'] = cell[None] # (1, 3, 3)
        return data
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms)
        
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)
        
        # if not self.nl:
        #     self.nl = NeighborList(self.rcut*np.ones(nats),
        #                            skin=self.skin,
        #                            bothways=True,
        #                            primitive=NewPrimitiveNeighborList)
        # self.nl.update(atoms)
        # # Creating input data
        # # get_neigh gives us (index_nn, (cell_indices))
        # this_pairs = 0
        # for i in range(nats):
        #     b,c = self.nl.get_neighbors(i)
        #     # Removing b==i and cells=[0,0,0]
        #     safeinds = np.logical_or(b!=i,np.sum(np.abs(c),axis=1)!=0)
        #     b = b[safeinds]
        #     c = c[safeinds]
        #     pp = len(b)
        #     if pp>0:
        #         new_pairs = this_pairs + pp
        #         # Reshaping the arrays, this should progressively happen less and less frequently
        #         if new_pairs>self.max_pairs:
        #             self.max_pairs = new_pairs
        #             self.inda.resize(new_pairs)
        #             self.indb.resize(new_pairs)
        #             self.indc.resize((new_pairs,3))
        #         self.inda[this_pairs:new_pairs] = b
        #         self.indb[this_pairs:new_pairs] = i
        #         self.indc[this_pairs:new_pairs] = c
        #         this_pairs = new_pairs
        # # Padding for safety
        # if this_pairs<self.max_pairs:
        #     self.inda[this_pairs:] = nats
        #     self.indb[this_pairs:] = nats
        #     self.indc[this_pairs:] = 0
            
        inda, indb, nn_r, nn_vecs, indc = neighbor_list('ijdDS', atoms, self.rcut)
        
        data = self.atoms2dict(nats,pos,cell,specs,self.inda,self.indb,self.indc)
        
        out = self.modapp(self.model_state, data)
        self.results = {
            "energy": out[0][0],
            "forces": out[1],
            "stress": -out[2][0]# ASE has opposite stress convention
        }
