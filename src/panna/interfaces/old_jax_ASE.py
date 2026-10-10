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
import os, glob
import pickle
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from panna.jax.parser_jax import trainjax_parameter_parser
from panna.jax.model_jax import make_model
from panna.gvector import GvectLATTE
import time
from mendeleev import element




# class PANNAJAXCalculator_ASEnn(Calculator):
#     """PANNA JAX ASE Calculator"""
#     implemented_properties = ["energy", "forces"]
    
#     def __init__(
#         self,
#         config = None,
#         skin = 5.0,
#         **kwargs):

#         Calculator.__init__(self, **kwargs)
#         self.nl = None
#         self.skin = skin
#         # Parsing config file
#         parameters = trainjax_parameter_parser(config)
#         preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
#         preprocess.parse_parameters(parameters['descriptor_params'])
#         raw_model = make_model(parameters, pre=preprocess)
#         model = hk.without_apply_rng(hk.transform(raw_model))
#         self.modapp = jax.jit(model.apply)
#         with open(os.path.dirname(config)+'/'+parameters['weights_file'], 'rb') as f:
#             self.model_state = pickle.load(f)
#         self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
#         self.rcut = parameters['cutoff']/2.0 + self.skin
#         self.max_pairs = 10
#         self.inda = np.zeros(self.max_pairs,dtype=np.int64)
#         self.indb = np.zeros(self.max_pairs,dtype=np.int64)
#         self.indc = np.zeros((self.max_pairs,3),dtype=np.int64)
        
#     def atoms2dict(self, nats, pos, cell, specs, inda, indb, cells):
#         data = {}
#         data['species'] = specs[:nats]
#         data['nats'] = np.asarray([nats])
#         data['ntot'] = nats
#         data['inda'] = inda
#         data['sp_a'] = specs[data['inda']]
#         data['indb'] = indb
#         data['sp_b'] = specs[data['indb']]
#         data['inde'] = np.zeros(nats,dtype=np.int32)
#         # data['mask'] = np.ones(len(inda),dtype=np.int32)
#         cellsh = cells@cell
#         # We do not recheck for cutoff.. with skin it's good enough
#         data['nn_vecs'] = pos[data['indb']]+cellsh-pos[data['inda']]
#         data['nn_r'] = np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
#         return data
        
#     def calculate(self, atoms=None, properties=['energy'],
#                   system_changes=all_changes):
#         # t1 = time.time()
#         Calculator.calculate(self, atoms)
#         nats = atoms.get_global_number_of_atoms()
#         pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
#         cell = np.asarray(atoms.get_cell())
#         specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)
#         if not self.nl:
#             self.nl = NeighborList(self.rcut*np.ones(nats),
#                                    skin=self.skin,
#                                    bothways=True,
#                                    primitive=NewPrimitiveNeighborList)
#         self.nl.update(atoms)
#         # print("Calc called!")
#         # Creating input data
#         # t2 = time.time()
#         # get_neigh gives us (index_nn, (cell_indices))
#         this_pairs = 0
#         for i in range(nats):
#             b,c = self.nl.get_neighbors(i)
#             # Removing b==i and cells=[0,0,0]
#             safeinds = np.logical_or(b!=i,np.sum(np.abs(c),axis=1)!=0)
#             b = b[safeinds]
#             c = c[safeinds]
#             pp = len(b)
#             if pp>0:
#                 new_pairs = this_pairs + pp
#                 # Reshaping the arrays, this should progressively happen less and less frequently
#                 if new_pairs>self.max_pairs:
#                     self.max_pairs = new_pairs
#                     self.inda.resize(new_pairs)
#                     self.indb.resize(new_pairs)
#                     self.indc.resize((new_pairs,3))
#                 self.inda[this_pairs:new_pairs] = b
#                 self.indb[this_pairs:new_pairs] = i
#                 self.indc[this_pairs:new_pairs] = c
#                 this_pairs = new_pairs
#         # Padding for safety
#         if this_pairs<self.max_pairs:
#             self.inda[this_pairs:] = nats
#             self.indb[this_pairs:] = nats
#             self.indc[this_pairs:] = 0
#         # t3 = time.time()
#         data = self.atoms2dict(nats,pos,cell,specs,self.inda,self.indb,self.indc)
#         # print([(k,np.asarray(data[k]).shape) for k in data.keys()])
#         # t4 = time.time()
#         out = self.modapp(self.model_state, data)
#         self.results = {
#             "energy": np.array(out[0][0]),
#             "forces": np.array(out[1])
#         }
#         # t5 = time.time()
#         # print('nnup {}, nnnei {}, a2d {}, net {}'.format(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)))
        
class PANNAJAXCalculator_ASEnn(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        config = None,
        skin = 5.0,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        self.nl = None
        self.skin = skin
        # Parsing config file
        parameters = trainjax_parameter_parser(config)
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(raw_model))
        self.modapp = jax.jit(model.apply)
        with open(os.path.dirname(config)+'/'+parameters['weights_file'], 'rb') as f:
            self.model_state = pickle.load(f)
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']/2.0 + self.skin
        self.max_pairs = 10
        self.inda = np.zeros(self.max_pairs,dtype=np.int64)
        self.indb = np.zeros(self.max_pairs,dtype=np.int64)
        self.indc = np.zeros((self.max_pairs,3),dtype=np.int64)
        
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
        # data['mask'] = np.ones(len(inda),dtype=np.int32)
        cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        
        # # Because of skin, the neighbor list may include extra pairs outside rcut.
        # # ASE docs explicitly note this.
        # mask = data['nn_r'] < self.rcut
        # print(np.where(mask==True))
        # if not np.all(mask):
        #     data['inda'] = inda[mask]
        #     data['indb'] = indb[mask]
        #     # data['indc'] = indc[mask]
        #     data['nn_vecs'] = data['nn_vecs'][mask]
        #     data['nn_r'] = data['nn_r'][mask]
        #     data['sp_a'] = specs[inda[mask]]
        #     data['sp_b'] = specs[indb[mask]]
        return data
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms)
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)
        if not self.nl:
            self.nl = NeighborList(self.rcut*np.ones(nats),
                                   skin=self.skin,
                                   bothways=True,
                                   primitive=NewPrimitiveNeighborList)
        self.nl.update(atoms)
        # Creating input data
        # get_neigh gives us (index_nn, (cell_indices))
        this_pairs = 0
        for i in range(nats):
            b,c = self.nl.get_neighbors(i)
            # Removing b==i and cells=[0,0,0]
            safeinds = np.logical_or(b!=i,np.sum(np.abs(c),axis=1)!=0)
            b = b[safeinds]
            c = c[safeinds]
            pp = len(b)
            if pp>0:
                new_pairs = this_pairs + pp
                # Reshaping the arrays, this should progressively happen less and less frequently
                if new_pairs>self.max_pairs:
                    self.max_pairs = new_pairs
                    self.inda.resize(new_pairs)
                    self.indb.resize(new_pairs)
                    self.indc.resize((new_pairs,3))
                self.inda[this_pairs:new_pairs] = b
                self.indb[this_pairs:new_pairs] = i
                self.indc[this_pairs:new_pairs] = c
                this_pairs = new_pairs
        # Padding for safety
        if this_pairs<self.max_pairs:
            self.inda[this_pairs:] = nats
            self.indb[this_pairs:] = nats
            self.indc[this_pairs:] = 0
        data = self.atoms2dict(nats,pos,cell,specs,self.inda,self.indb,self.indc)
        out = self.modapp(self.model_state, data)
        self.results = {
            "energy": out[0][0],
            "forces": out[1]
        }


class PANNAJAXCalculator(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        config = None,
        skin =  10,#0.01, #2.0,
        e_field = 0.0,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        print('PANNA JAX sr ASE Calculator')
        
        # Parsing config file
        parameters = trainjax_parameter_parser(config, train=False)
        parameters['precompute_ewald_sums'] = False
        parameters['e_field'] = e_field
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        folder = '/'.join(config.split('/')[0:-1])
        # model
        if parameters['weights_file'] is None: # then take the last model!
            print('INFO - taking the last model of training')
            files = glob.glob(parameters['train_dir']+'/models/*pkl')
            namest = [(f,int(f.split('step_')[-1].split('.')[0])) \
                      for f in files \
                      if f.split('step_')[-1].split('.')[0].isnumeric()]
            weif = namest[np.argmax([n[1] for n in namest])][0]
            weif = weif.split('/')[1:][-1]
        else:
            weif = parameters['weights_file']
        
        print('INFO - reading model ', weif)
        with open(folder+'/models/'+weif, 'rb') as f:
            self.model_state = pickle.load(f)

        self.raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(self.raw_model))
        self.modapp = jax.jit(model.apply)
        
        
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']#/2.0 + self.skin
        self.skin = skin
        # self.previous_positions = [0.,0.,0.]
        
        # needed for LR
        # self.acc_factor, self.Rc, self.Kc_sc, self.gaussian_smearing = parameters['ewald_input_params']
        # self.alpha = 1/(np.sqrt(2)*self.gaussian_smearing) if not self.gaussian_smearing == 0.0 else 0.0
        
        self.count = 0
        self.verbose = False
        
        
    def atoms2dict(self, nats, pos, cell, specs, inda, indb, indf,nn_vecs, nn_r):
        data = {}
        data['species'] = specs[:nats]
        data['nats'] = np.asarray([nats])
        data['ntot'] = nats
        data['inda'] = inda
        data['sp_a'] = specs[data['inda']]
        data['indb'] = indb
        data['sp_b'] = specs[data['indb']]
        data['inde'] = np.zeros(nats,dtype=np.int32)
        # cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = nn_vecs #pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = nn_r #np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        data['indf'] = indf
        pos[-1] = np.inf # for ewald things
        # data['pos'] = pos[jnp.newaxis]
        # data['latvecs'] = cell[jnp.newaxis]
        # data['ewald_input_realspace'] = ewald_params[0]
        # data['ewald_input_alphas'] = ewald_params[1]
        # data['ewald_input_recspace'] = ewald_params[2]
        # data['linsys_masks'] = linsys_mask
        # data['gaussian_radii'] = np.zeros(nats) # point charges
        # # print(data['pos'].shape, data['latvecs'], cellsh.shape,  'ssss')
        # # print(data['species'],'ssss')
        return data
        
    def calculate(self, atoms=None, properties=['energy', 'forces', 'charges'],
                  system_changes=all_changes):
        t1 = time.time()
        Calculator.calculate(self, atoms)
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        # self.max_displ = np.max(np.linalg.norm(pos - self.previous_positions, axis=1))
        # coords = np.pad(pos[0:-1], ((0,100-64),(0,0)) , constant_values=np.inf )
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()],dtype=np.int32)#np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)

        # needed for LR
        linsys_mask = create_masks((nats,),nats+1) #100)# 
        
        # ewald_params = set_ewald_parameters(pos[0:-1], cell, self.acc_factor, Rc=self.Rc, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
        # print('aaaaa', ewald_params[0].shape, ewald_params[1], ewald_params[2].shape)
        
        
        # print(self.max_displ, self.skin)
        # if self.max_displ > self.skin or self.count == 0:
        #     print('Ricalculating NN list')
        positions, replicas = make_replicas(pos[0:-1], cell, self.rcut)
        # replicas2 = get_Scells(SCell, R0)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,-1,3))
        rij_vec = posj-posi  +rep
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))
        radial_mask = np.logical_and(rij < self.rcut, rij > 1e-8)
        indall = np.where(radial_mask)

        nn_vecs = rij_vec[indall]
        nn_r = rij[indall]
        inda = indall[0]
        indb = indall[1]
        
        # self.previous_positions = pos.copy()
            
        indf = np.arange(nats) # atom index within structure 
            
        # t2 = time.time()
        data = self.atoms2dict(nats,pos,cell,specs,inda,indb,indf,nn_vecs,nn_r)
        # print([(k,np.asarray(data[k]).shape) for k in data.keys()])
        
        
        out = self.modapp(self.model_state, data)
        
        
        self.results = {
            "energy": np.array(out[0][0]), #self.nl,#np.array(out[2]),#
            "forces": np.array(out[1])
        }
        t3 = time.time()
        # print('sec for NN prep', t2-t1)
        print('sec for ff', t3-t1)
        
        # # print('nnup {}, nnnei {}, a2d {}, net {}'.format(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)))
        
        


from panna.electrostatics.Electrostatics_jax_qq import create_masks, set_ewald_parameters
from panna.electrostatics.Electrostatics_jax_qq_gc import set_ewald_parameters_gaussians   
        
class PANNAJAXCalculatorLR_ASEnn(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces", "charges"]
    
    def __init__(
        self,
        config = None,
        skin =  10,#0.01, #2.0,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        self.nl = None
        self.skin = skin
        # Parsing config file
        parameters = trainjax_parameter_parser(config)
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        self.raw_model = make_model(parameters, pre=preprocess)
        
        model = hk.without_apply_rng(hk.transform(self.raw_model))
        self.modapp = jax.jit(model.apply)
        with open(os.path.dirname(config)+'/'+parameters['weights_file'], 'rb') as f:
            self.model_state = pickle.load(f)
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']/2.0 + self.skin
        self.max_pairs = 10
        self.inda = np.zeros(self.max_pairs,dtype=np.int64)
        self.indb = np.zeros(self.max_pairs,dtype=np.int64)
        self.indc = np.zeros((self.max_pairs,3),dtype=np.int64)
        # needed for LR
        # parameters['gaussian_smearing'] = 2.42
        self.acc_factor, self.Rc, self.Kc_sc, self.gaussian_smearing = parameters['ewald_input_params']
        self.alpha = 1/(np.sqrt(2)*self.gaussian_smearing) if not np.isnan(self.gaussian_smearing) else np.nan
        # parameters['ewald_input_params'][3]) if 
        self.raw_model.parameters['task'] = 'val'
        self.count = 0
        self.verbose = False
        
    def atoms2dict(self, nats, pos, cell, specs, inda, indb, cells, indf, ewald_params, linsys_mask):
        data = {}
        data['species'] = specs[:nats]
        data['nats'] = np.asarray([nats])
        data['ntot'] = nats
        data['inda'] = inda
        data['sp_a'] = specs[data['inda']]
        data['indb'] = indb
        data['sp_b'] = specs[data['indb']]
        data['inde'] = np.zeros(nats,dtype=np.int32)
        # data['mask'] = np.ones(len(inda),dtype=np.int32)
        cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        data['indf'] = indf
        pos[-1] = np.inf
        data['pos'] = pos[jnp.newaxis]
        data['latvecs'] = cell[jnp.newaxis]
        data['ewald_params'] = ewald_params
        data['linsys_masks'] = linsys_mask
        # print(data['pos'].shape, data['latvecs'], cellsh.shape,  'ssss')
        # print(data['species'],'ssss')
        return data
        
    def calculate(self, atoms=None, properties=['energy', 'forces', 'charges'],
                  system_changes=all_changes):
        # t1 = time.time()
        Calculator.calculate(self, atoms)
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        # coords = np.concatenate([atoms.get_positions(),[[np.inf,np.inf,np.inf]]],axis=0)
        # coords = np.pad(pos[0:-1], ((0,100-64),(0,0)) , constant_values=np.inf )
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)

        # needed for LR
        linsys_mask = create_masks((nats,),nats+1) #100)# 
        if self.count == 0:
            self.verbose = True
        ewald_params = set_ewald_parameters(pos[0:-1], cell, self.acc_factor, Rc=self.Rc, alpha=self.alpha, Kc_sc=self.Kc_sc, verbose=self.verbose)
        self.count += 1
        # this sets the ewald parameters in the PANNA LR model
        # self.raw_model.parameters['ewald_params'] = ewald_params
        # self.raw_model.parameters['linsys_masks'] = linsys_masks

        
        if not self.nl:
            self.nl = NeighborList(self.rcut*np.ones(nats),
                                    skin=self.skin,
                                    bothways=True,
                                    primitive=NewPrimitiveNeighborList)
        self.nl.update(atoms)
        # print("Calc called!")
        # Creating input data
        # t2 = time.time()
        # get_neigh gives us (index_nn, (cell_indices))
        this_pairs = 0
        for i in range(nats):
            b,c = self.nl.get_neighbors(i)
            # Removing b==i and cells=[0,0,0]
            safeinds = np.logical_or(b!=i,np.sum(np.abs(c),axis=1)!=0)
            b = b[safeinds]
            c = c[safeinds]
            pp = len(b)
            if pp>0:
                new_pairs = this_pairs + pp
                # Reshaping the arrays, this should progressively happen less and less frequently
                if new_pairs>self.max_pairs:
                    self.max_pairs = new_pairs
                    self.inda.resize(new_pairs)
                    self.indb.resize(new_pairs)
                    self.indc.resize((new_pairs,3))
                self.inda[this_pairs:new_pairs] = b
                self.indb[this_pairs:new_pairs] = i
                self.indc[this_pairs:new_pairs] = c
                this_pairs = new_pairs
        
        self.indf = np.arange(nats) # atom index within structure 

        # Padding for safety
        if this_pairs<self.max_pairs:
            self.inda[this_pairs:] = nats
            self.indb[this_pairs:] = nats
            self.indc[this_pairs:] = 0
            
        # t3 = time.time()
        data = self.atoms2dict(nats,pos,cell,specs,self.inda,self.indb,self.indc,self.indf, ewald_params, linsys_mask)
        # print([(k,np.asarray(data[k]).shape) for k in data.keys()])
        # t4 = time.time()
        
        
        out = self.modapp(self.model_state, data)
        
        
        self.results = {
            "energy": np.array(out[0][0]), #self.nl,#np.array(out[2]),#
            "forces": np.array(out[1]),
            "charges": np.array(out[2])
        }
        # t5 = time.time()
        # print('nnup {}, nnnei {}, a2d {}, net {}'.format(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)))
        

from panna.gvector.pbc import make_replicas

class PANNAJAXCalculatorLR(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces", "charges", "dipole", "magmom"]
    
    def __init__(
        self,
        config = None,
        skin =  10,#0.01, #2.0,
        weights_file = None,
        verbose = False,
        phi_ext = 0.0, E_ext = 0.0,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        print('PANNA JAX Long Range ASE Calculator')
        
        # Parsing config file
        parameters = trainjax_parameter_parser(config, train=False)
        parameters['precompute_ewald_sums'] = False
        parameters['phi_ext'] = phi_ext
        parameters['Efield_ext'] = E_ext
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        folder = '/'.join(config.split('/')[0:-1])
        # model
        if weights_file is None: # then take the last model!
            print('INFO - taking the last model of training')
            # files = glob.glob(parameters['train_dir']+'/models/*pkl')
            files = glob.glob(folder+'/models/*pkl')
            namest = [(f,int(f.split('step_')[-1].split('.')[0])) \
                      for f in files \
                      if f.split('step_')[-1].split('.')[0].isnumeric()]
            weif = namest[np.argmax([n[1] for n in namest])][0]
            weif = weif.split('/')[-1:][0]            
        else:
            # weif = parameters['weights_file']
            weif = weights_file
            
        
        print('INFO - reading model ', weif)
        with open(folder+'/models/'+weif, 'rb') as f:
            self.model_state = pickle.load(f)
            
        self.raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(self.raw_model))
        self.modapp = jax.jit(model.apply)
        
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']#/2.0 + self.skin
        self.skin = skin
        # self.previous_positions = [0.,0.,0.]
        
        # needed for LR
        self.acc_factor, self.Rc, self.Kc_sc, self.gaussian_smearing = parameters['ewald_input_params']
        # self.alpha = 1/(np.sqrt(2)*self.gaussian_smearing) if not self.gaussian_smearing == 0.0 else 0.0
        
        self.count = 0
        self.verbose = verbose
        
        # gaussian charges or point charges?
        gaussian_charges = parameters['gaussian_charges']
        species_list = parameters['species']
        if gaussian_charges[0] == 'covalent':
            radii = [element(x).covalent_radius * 0.01 * np.sqrt(2.0) for x in species_list]
        elif gaussian_charges[0] == 'ionic':
            radii = [element(x).ionic_radii[0].ionic_radius * 0.01 * np.sqrt(2.0) for x in species_list]
        elif gaussian_charges[0] in [b'False', 'False']:
            radii = [0. for x in species_list]
        else:
            radii = [float(x) for x in gaussian_charges]
        self.radii = np.array(radii) # (nspecies,)
        self.gaussian_charges = np.any(self.radii > 0)
        print('INFO - Gaussian charges is ', self.gaussian_charges)
        if self.gaussian_charges:
            print('\t Gauss radii', np.round(self.radii,3))

        
        

        
        
    def atoms2dict(self, nats, pos, cell, specs, inda, indb, indf,nn_vecs, nn_r, ewald_params, linsys_mask):
        data = {}
        data['species'] = specs[:nats]
        data['nats'] = np.asarray([nats])
        data['ntot'] = nats
        data['inda'] = inda
        data['sp_a'] = specs[data['inda']]
        data['indb'] = indb
        data['sp_b'] = specs[data['indb']]
        data['inde'] = np.zeros(nats,dtype=np.int32)
        # cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = nn_vecs #pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = nn_r #np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        data['indf'] = indf
        pos[-1] = np.inf # for ewald things
        data['pos'] = pos[jnp.newaxis]
        data['latvecs'] = cell[jnp.newaxis]
        data['ewald_input_Gcuts'] = ewald_params[0][None]
        data['ewald_input_rcuts'] = ewald_params[1][None]
        data['ewald_input_realspace'] = ewald_params[2]
        data['ewald_input_alphas'] = ewald_params[3]
        data['ewald_input_recspace'] = ewald_params[4]
        data['linsys_masks'] = linsys_mask
        # assigns radius to each atom, depending on its specie
        gauss_radii = self.radii[specs] # shape: (Nats,)
        data['gaussian_radii'] = np.concatenate((gauss_radii,[0]))[None] # shape: (1, Nats+1)
        # print(data['pos'].shape, data['latvecs'], cellsh.shape,  'ssss')
        # print(data['species'],'ssss')
        return data
        
    def calculate(self, atoms=None, properties=['energy', 'forces', 'charges', 'dipole', 'extra'],
                  system_changes=all_changes):
        # t1 = time.time()
        Calculator.calculate(self, atoms)
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        # self.max_displ = np.max(np.linalg.norm(pos - self.previous_positions, axis=1))
        # coords = np.pad(pos[0:-1], ((0,100-64),(0,0)) , constant_values=np.inf )
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()],dtype=np.int32)#np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)
        
        # needed for LR
        linsys_mask = create_masks((nats,),nats+1) #100)# 
        if not self.gaussian_charges: # then it's point charges
            ewald_params = set_ewald_parameters(pos[0:-1], cell, self.acc_factor, Rc=self.Rc, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
        else: # it's gaussian charges
            ewald_params = set_ewald_parameters_gaussians(pos[0:-1], cell, self.radii, self.acc_factor, Rc=self.Rc, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
        
        
        # print(self.max_displ, self.skin)
        # if self.max_displ > self.skin or self.count == 0:
        #     print('Ricalculating NN list')
        positions, replicas = make_replicas(pos[0:-1], cell, self.rcut)
        # replicas2 = get_Scells(SCell, R0)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,-1,3))
        rij_vec = posj-posi  +rep
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))
        radial_mask = np.logical_and(rij < self.rcut, rij > 1e-8)
        indall = np.where(radial_mask)
        nn_vecs = rij_vec[indall]
        nn_r = rij[indall]
        inda = indall[0]
        indb = indall[1]
        indf = np.arange(nats) # atom index within structure 

        # t2 = time.time()
        data = self.atoms2dict(nats,pos,cell,specs,inda,indb,indf,nn_vecs,nn_r, ewald_params, linsys_mask)
        # print([(k,np.asarray(data[k]).shape) for k in data.keys()])
        out = self.modapp(self.model_state, data)
        self.results = {
            "energy": np.array(out[0][0]), #self.nl,#np.array(out[2]),#
            "forces": np.array(out[1]),
            "charges": np.array(out[3]) ,
            # "dipole": np.array(out[3]), # this is the short range energy
            "magmom": out[5]  # this is the pure lr energy
        }
        t3 = time.time()
        # print('sec for NN prep', t2-t1)
        # print('sec for ff', t3-t2)
        
        # # print('nnup {}, nnnei {}, a2d {}, net {}'.format(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)))
        
        

        
from panna.electrostatics.Electrostatics_jax_pol import create_masks_pol    
class PANNAJAXCalculatorLR_pol(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces", "charges", "dipole", "magmom"]
    
    def __init__(
        self,
        config = None,
        skin =  10,#0.01, #2.0,
        weights_file=None,
        verbose=False,
        phi_ext = 0.0, E_ext = 0.0,
        **kwargs):
        Calculator.__init__(self, **kwargs)
        print('PANNA JAX Long Range Pol ASE Calculator')
        
        
        # Parsing config file
        parameters = trainjax_parameter_parser(config, train=False)
        parameters['precompute_ewald_sums'] = False
        parameters['phi_ext'] = phi_ext
        parameters['Efield_ext'] = E_ext
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        folder = '/'.join(config.split('/')[0:-1])
        # model
        if weights_file is None: # then take the last model!
            print('INFO - taking the last model of training')
            files = glob.glob(folder+'/models/*pkl')
            namest = [(f,int(f.split('step_')[-1].split('.')[0])) \
                      for f in files \
                      if f.split('step_')[-1].split('.')[0].isnumeric()]
            weif = namest[np.argmax([n[1] for n in namest])][0]
            weif = 'models/'+weif.split('/')[-1:][0] 
        else:
            # weif = parameters['weights_file']
            weif = 'models/'+weights_file
            
        
        print('INFO - reading model ', weif)
        with open(folder+'/'+weif, 'rb') as f: # os.path.dirname(config)+
            self.model_state = pickle.load(f)
            
        # model
        self.raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(self.raw_model))
        self.modapp = jax.jit(model.apply)
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']#/2.0 + self.skin
        self.skin = skin
        # self.previous_positions = [0.,0.,0.]
        
        # needed for LR
        self.acc_factor, self.Rc, self.Kc_sc, self.gaussian_smearing = parameters['ewald_input_params']
        # self.alpha = 1/(np.sqrt(2)*self.gaussian_smearing) if not self.gaussian_smearing == 0.0 else 0.0
        
        self.count = 0
        self.verbose = verbose
        
        # gaussian charges or point charges?
        gaussian_charges = parameters['gaussian_charges']
        species_list = parameters['species']
        if gaussian_charges[0] == 'covalent':
            radii = [element(x).covalent_radius * 0.01 * np.sqrt(2.0) for x in species_list]
        elif gaussian_charges[0] == 'ionic':
            radii = [element(x).ionic_radii[0].ionic_radius * 0.01 * np.sqrt(2.0) for x in species_list]
        elif gaussian_charges[0] in [b'False', 'False']:
            radii = [0. for x in species_list]
        else:
            radii = [float(x) for x in gaussian_charges]
        self.radii = np.array(radii) # (nspecies,)
        self.gaussian_charges = np.any(self.radii > 0)
        print('INFO - Gaussian charges is ', self.gaussian_charges)
        if self.gaussian_charges:
            print('\t Gauss radii', np.round(self.radii,3))
        
        
    def atoms2dict(self, nats, pos, cell, specs, inda, indb, indf, indg, nn_vecs, nn_r, ewald_params, linsys_mask):
        data = {}
        data['species'] = specs[:nats]
        data['nats'] = np.asarray([nats])
        data['ntot'] = nats
        data['inda'] = inda
        data['sp_a'] = specs[data['inda']]
        data['indb'] = indb
        data['sp_b'] = specs[data['indb']]
        data['inde'] = np.zeros(nats,dtype=np.int32)

        # cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = nn_vecs #pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = nn_r #np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        data['indf'] = indf
        data['indg'] = indg
        pos[-1] = np.inf # for ewald things
        data['pos'] = pos[jnp.newaxis]
        data['latvecs'] = cell[jnp.newaxis]
        data['ewald_input_realspace'] = ewald_params[0]
        data['ewald_input_alphas'] = ewald_params[1]
        data['ewald_input_recspace'] = ewald_params[2]
        data['linsys_masks'] = linsys_mask
        gauss_radii = self.radii[specs] # shape: (Nats,)
        data['gaussian_radii'] = np.concatenate((gauss_radii,[0]))[None] # shape: (1, Nats+1)
        # print(data['pos'].shape, data['latvecs'], cellsh.shape,  'ssss')
        # print(data['species'],'ssss')
        return data
        
    def calculate(self, atoms=None, properties=['energy', 'forces', 'charges', 'dipole', 'magmom'],
                  system_changes=all_changes):
        t1 = time.time()
        Calculator.calculate(self, atoms)
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        # self.max_displ = np.max(np.linalg.norm(pos - self.previous_positions, axis=1))
        # coords = np.pad(pos[0:-1], ((0,100-64),(0,0)) , constant_values=np.inf )
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()],dtype=np.int32)#np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)

        # needed for LR
        linsys_mask = create_masks_pol( (nats,), nats+1, n=4)
        
        if not self.gaussian_charges: # then it's point charges
            ewald_params = set_ewald_parameters(pos[0:-1], cell, self.acc_factor, Rc=self.Rc, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
        else: # it's gaussian charges
            ewald_params = set_ewald_parameters_gaussians(pos[0:-1], cell, self.radii, self.acc_factor, Rc=self.Rc, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)

        # ewald_params = set_ewald_parameters(pos[0:-1], cell, self.acc_factor, Rc=self.Rc, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
        
        
        # print(self.max_displ, self.skin)
        # if self.max_displ > self.skin or self.count == 0:
        #     print('Ricalculating NN list')
        positions, replicas = make_replicas(pos[0:-1], cell, self.rcut)
        # replicas2 = get_Scells(SCell, R0)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,-1,3))
        rij_vec = posj-posi  +rep
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))
        radial_mask = np.logical_and(rij < self.rcut, rij > 1e-8)
        indall = np.where(radial_mask)

        nn_vecs = rij_vec[indall]
        nn_r = rij[indall]
        inda = indall[0]
        indb = indall[1]
        
        # self.previous_positions = pos.copy()
            
        indf = np.arange(nats) # atom index within structure 
        indg = np.arange(nats+1, nats*4+1) # atom index within structure 


        
            
        t2 = time.time()
        data = self.atoms2dict(nats,pos,cell,specs,inda,indb,indf,indg,nn_vecs,nn_r, ewald_params, linsys_mask)

        
        out = self.modapp(self.model_state, data)
        
        
        self.results = {
            "energy": np.array(out[0][0]), 
            "forces": np.array(out[1]),
            "charges": np.array(out[2]),
            "dipole": np.array(out[3]),
            "magmom": out[4]
        }
        t3 = time.time()
        # print('sec for NN prep', t2-t1)
        # print('sec for ff', t3-t2)
        
        # # print('nnup {}, nnnei {}, a2d {}, net {}'.format(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)))
        

class PANNAJAXCalculator_original(Calculator):
    """PANNA JAX ASE Calculator"""
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        config = None,
        skin = 5.0,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        self.nl = None
        self.skin = skin
        # Parsing config file
        parameters = trainjax_parameter_parser(config)
        preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(raw_model))
        self.modapp = jax.jit(model.apply)
        with open(os.path.dirname(config)+'/'+parameters['weights_file'], 'rb') as f:
            self.model_state = pickle.load(f)

        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']/2.0 + self.skin
        self.max_pairs = 10
        self.inda = np.zeros(self.max_pairs,dtype=np.int64)
        self.indb = np.zeros(self.max_pairs,dtype=np.int64)
        self.indc = np.zeros((self.max_pairs,3),dtype=np.int64)
        
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
        cellsh = cells@cell
        # We do not recheck for cutoff.. with skin it's good enough
        data['nn_vecs'] = pos[data['indb']]+cellsh-pos[data['inda']]
        data['nn_r'] = np.sqrt(np.sum(data['nn_vecs']**2,axis=1))
        return data
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        # t1 = time.time()
        Calculator.calculate(self, atoms)
        nats = atoms.get_global_number_of_atoms()
        pos = np.concatenate([atoms.get_positions(),[[0,0,0]]],axis=0)
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)
        if not self.nl:
            self.nl = NeighborList(self.rcut*np.ones(nats),
                                   skin=self.skin,
                                   bothways=True,
                                   primitive=NewPrimitiveNeighborList)
        self.nl.update(atoms)
        # print("Calc called!")
        # Creating input data
        # t2 = time.time()
        # get_neigh gives us (index_nn, (cell_indices))
        this_pairs = 0
        for i in range(nats):
            b,c = self.nl.get_neighbors(i)
            pp = len(b)
            new_pairs = this_pairs + pp
            # Reshaping the arrays, this should progressively happen less and less frequently
            if new_pairs>self.max_pairs:
                self.max_pairs = new_pairs
                self.inda.resize(new_pairs)
                self.indb.resize(new_pairs)
                self.indc.resize((new_pairs,3))
            self.inda[this_pairs:new_pairs] = b
            self.indb[this_pairs:new_pairs] = i
            self.indc[this_pairs:new_pairs] = c
            this_pairs = new_pairs
        # Padding for safety
        if this_pairs<self.max_pairs:
            self.inda[this_pairs:] = nats
            self.indb[this_pairs:] = nats
            self.indc[this_pairs:] = 0
        # t3 = time.time()
        data = self.atoms2dict(nats,pos,cell,specs,self.inda,self.indb,self.indc)
        # print([(k,np.asarray(data[k]).shape) for k in data.keys()])
        # t4 = time.time()
        out = self.modapp(self.model_state, data)

        self.results = {
            "energy": out[0][0],
            "forces": out[1]
        }
        # t5 = time.time()
        # print('nnup {}, nnnei {}, a2d {}, net {}'.format(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t5-t4)))
