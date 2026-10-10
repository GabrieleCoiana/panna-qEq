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
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList, PrimitiveNeighborList
from ase.neighborlist import neighbor_list 
from panna.jax.parser_jax import trainjax_parameter_parser
from panna.jax.model_jax import make_model
from panna.gvector import GvectLATTE
from mendeleev import element
import time
# needed for LR models
from panna.electrostatics.Electrostatics_jax_pol import create_masks_pol    
from panna.electrostatics.Electrostatics_jax_qq import create_masks, set_ewald_parameters
from panna.electrostatics.Electrostatics_jax_qq_gc import set_ewald_parameters_gaussians   
from panna.electrostatics.Electrostatics_jax_pol_gc import prepare_Ewald_sum


class PANNAJAXCalculator_new(Calculator):
    implemented_properties = ["energy", "forces", "stress", "charges", "dipole", "magmom"]
    
    def __init__(self,
            config = None,
            skin = 0.5,
            weights_file = None,
            verbose = False,
            timeit = False,
            pad_things = True,
            phi_ext = [], E_ext = [],
            ewald_acc_factor = 0,
            force_nl_creation = False,
            rcut_LR = 10, # default for long range cutoff
            **kwargs):
        
        Calculator.__init__(self, **kwargs)
        print('Unified PANNA-JAX ASE Calculator')
        
        self.phi_ext = phi_ext
        self.E_ext = E_ext
        self.skin = skin
        self.timeit = timeit
        self.pad_things = pad_things
        self.ewald_acc_factor = ewald_acc_factor
        self.rcut_LR = rcut_LR
        
        # getting input parameters
        parameters = trainjax_parameter_parser(config, train=False)
        # forcing some parameters
        parameters['task'] = 'validation'
        parameters['precompute_ewald_sums'] = False
        parameters['BORN'] = ''
        parameters['stress_cost'] = 1 # forcing stress calculation
        
        
        
        # getting model state (weights) and parameters
        self.model_state, preprocess = self.parse_configfile(config, weights_file, parameters['species_str'], parameters['descriptor_params'])
        
        # model
        raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(raw_model))
        self.modapp = jax.jit(model.apply)
        
        # miscellaneous
        self.mapping = dict(zip(parameters['species'], range(len(parameters['species']))))
        self.rcut = parameters['cutoff']#/2.0 + self.skin
        
        
        # needed for LR
        self.verbose = verbose
        
        self.is_LR_model = 'LR' in parameters['model_type']
        self.long_range_pol = 'pol' in parameters['model_type'] # parameters['model_type'] in ['MLP_LR_pol' ,'MLP_LR_fq_pol', 'MLP_LR_fq_pol_modsr']
        
        if self.is_LR_model:
            self.rcut_LR = parameters['ewald_input_params'][1] if not parameters['ewald_input_params'][1] == 0 else self.rcut_LR
            self.initialize_LR_stuff(parameters['gaussian_charges'], parameters['species'], parameters['ewald_input_params'])
        
        self.nl = None
        self.force_nl_creation = force_nl_creation
        self.Npad = 0
        self.Npad_LR = 0
        # self.max_pairs = 10
        # self.inda = np.zeros(self.max_pairs,dtype=np.int64)
        # self.indb = np.zeros(self.max_pairs,dtype=np.int64)
        # self.indc = np.zeros((self.max_pairs,3),dtype=np.int64)
        
    def calculate(self, atoms, properties=['energy', 'forces', 'charges', 'dipole', 'magmom'],
                  system_changes=all_changes):
        t0 = time.time()
        
        pos = atoms.get_positions()
        # vol = atoms.get_volume()
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()]+[0],dtype=np.int32)
        nats = len(pos)
        indf, indg, nn_r_LR, nn_vecs_LR = 0, 0, 0, 0 #◙ defaults, in case is SR model
        
        # ASE base calculator call
        Calculator.calculate(self, atoms, properties, system_changes)
        
        t1 = time.time()
        # In principle, ASE should rebuild the NN only once in a while ...
        # Recreate NL only if topology-relevant things changed
        # rebuilt_nl = False
        if self.nl is None or self.force_nl_creation == True: # nl object built only at start
            self.nl = NeighborList(self.rcut/2*np.ones(nats),
                                    skin=self.skin,
                                    bothways=True,
                                    self_interaction=False,
                                    primitive=NewPrimitiveNeighborList)
            if self.is_LR_model:
                self.nl_LR = NeighborList(self.rcut_LR/2*np.ones(nats),
                                        skin=self.rcut_LR / 10,
                                        bothways=True,
                                        self_interaction=False,
                                        primitive=NewPrimitiveNeighborList)
                
            # rebuilt_nl = True
        # Checking to rebuild SR NN lists
        rebuilt_nl_SR = self.nl.update(atoms) #or rebuilt_nl
        if rebuilt_nl_SR: # then NN list has been rebuilt, we need to update indices
            print('ASE - Rebuilding SR NN list!')
            # pad the cutoff used for the cached list so it stays valid
            # until the *next* rebuild, matching self.nl's effective cutoff
            cutoff_padded = self.rcut + self.skin
            self.inda, self.indb, self.indc = neighbor_list('ijS', atoms, cutoff_padded)
            # self.newnew_rebuild_pair_indices(atoms, cutoff_padded)
            
        nn_r, nn_vecs = self.compute_distances(pos, cell, self.inda, self.indb, self.indc)
        # otherwise use the same indices as before!

        # # ... In practice, I found that this algorithm is much faster
        # # inda, indb, indc = neighbor_list('ijS', atoms, self.rcutt)
        # inda, indb, nn_r, nn_vecs, indc = neighbor_list('ijdDS', atoms, self.rcut)
        t2 = time.time()
        if self.timeit:
            print('TIME - ASE SR NN ', t2-t1)
        
        # LR stuff
        if self.is_LR_model: 
            # Then we need the NN list for the real space Ewald sum
            # Checking to rebuild LR NN lists
            rebuilt_nl_LR = self.nl_LR.update(atoms) #or rebuilt_nl
            if rebuilt_nl_LR: # then NN list has been rebuilt, we need to update indices
                print('ASE - Rebuilding LR NN list!')
                # We need to set the Ewald parameters
                if not self.gaussian_charges: # then it's point charges
                    self.ewald_params = set_ewald_parameters(nats, cell, self.acc_factor, Rc=self.rcut_LR, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
                else: # it's gaussian charges
                    self.ewald_params = set_ewald_parameters_gaussians([nats], cell, self.radii, self.acc_factor, Rc=self.rcut_LR, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
                # self.rcut_LR = self.ewald_params[1]
                                
                # pad the cutoff used for the cached list so it stays valid
                # until the *next* rebuild, matching self.nl's effective cutoff
                cutoff_padded = self.rcut_LR + self.rcut_LR / 10
                self.inda_LR, self.indb_LR, self.indc_LR = neighbor_list('ijS', atoms, cutoff_padded)
                

            nn_r_LR, nn_vecs_LR = self.compute_distances(pos, cell, self.inda_LR, self.indb_LR, self.indc_LR)
            nn_vecs_LR = - nn_vecs_LR # I define Rij = Ri - Rj, PANNA defines the opposite 
            
            # then we need additional two sets of indices
            indf = np.arange(nats)             # atom index within structure 
            indg = np.arange(nats+1, nats*4+1) # atom index within structure 
            
            # and we need linsys masks (the + 1 is for padding)
            if self.long_range_pol:
                self.linsys_mask = np.array(create_masks_pol( (nats,), nats+1, n=4))
            else: # it's a charge-only model
                self.linsys_mask = create_masks((nats,), nats+1)
            
            # tt = time.time()
            # print(self.ewald_params[1])
            # self.ewald_prep = prepare_Ewald_sum(pos, cell, self.ewald_params[2], self.ewald_params[4], self.ewald_params[1], self.ewald_params[0])
            # ttt = time.time()
            # if self.timeit:
            #     print('ewald prep [s]', ttt-tt)
        t3 = time.time()
        if self.timeit:
            print('TIME - LR NN and preparation', t3-t2)
        
        # create input dictionary for feeding the model
        data = self.atoms2dict(nats,pos,cell, nn_r,nn_vecs, nn_r_LR, nn_vecs_LR, specs,indf,indg)
        # tt = time.time()
        # calling the model
        out = self.modapp(self.model_state, data)
        # ff = np.array(out[1])
        # ttt = time.time()
        # if self.timeit:
        #     print(ttt - tt, 'sec aaa')
        

        self.results = {
            "energy": np.array(out[0][0]), 
            "forces": np.array(out[1]),
            "stress": -np.array(out[2][0]) #/ vol # virial / volume
        }
        if self.is_LR_model:
            self.results.update({
                "charges": np.array(out[3]),
                "dipole":  np.array(out[4]),
                "magmom":  {key:np.array(item) for (key,item) in out[5].items() }  # here I store some extra parameters
                })
        t4 = time.time()
        if self.timeit:
            print('TIME - model call ', t4-t3)
            print('TIME - Total ', t4 - t0)
      
    
    def atoms2dict(self, nats, pos, cell, nn_r, nn_vecs, nn_r_LR, nn_vecs_LR, specs, indf, indg):
        # Because of skin, the neighbor list may include extra pairs outside rcut.
        # ASE docs explicitly note this.
        mask = nn_r < self.rcut
        if not np.all(mask):
            inda = self.inda[mask]
            indb = self.indb[mask]
            indc = self.indc[mask]
            nn_vecs = nn_vecs[mask]
            nn_r = nn_r[mask]
            # sp_a = specs[inda[mask]]
            # sp_b = specs[indb[mask]]
        else:
            inda = self.inda
            indb = self.indb
            indc = self.indc
            
        if self.pad_things:
            # the following is to avoid recompiling every time
            Npairs = len(inda)
            # round Npairs to the next hundred
            Npad_this = np.ceil(Npairs / 1000).astype(int) * 1000 # e.g. 15636 -> 15700
            Npad = np.max([Npad_this, self.Npad])
            # pad the arrays
            inda = np.pad(inda, (0,Npad-Npairs), constant_values=nats+1)
            indb = np.pad(indb, (0,Npad-Npairs), constant_values=nats+1)
            indc = np.pad(indc, (0,Npad-Npairs), constant_values=nats+1)
            nn_r = np.pad(nn_r, (0,Npad-Npairs), constant_values=nats+1)
            nn_vecs = np.pad(nn_vecs, ((0,Npad-Npairs),(0,0)), constant_values=Npad)
            specs = np.pad(specs, (0,1), constant_values=specs.max()+1)
            self.Npad = Npad
            # print(Npairs, Npad)
        
        # let's put everything in a dictionary
        data = {}
        data['species'] = specs[:nats]
        data['nats'] = np.asarray([nats]) 
        data['ntot'] = nats
        data['inda'] = inda
        data['sp_a'] = specs[inda]
        data['indb'] = indb
        data['sp_b'] = specs[indb]
        data['inde'] = np.zeros(nats,dtype=np.int32)
        data['inds'] = np.zeros(len(inda),dtype=np.int32) # for stress
        data['latvecs'] = cell[None] # for stress
        data['nn_vecs'] = nn_vecs
        data['nn_r'] = nn_r
        
  
        # LR stuff
        if self.is_LR_model:
            data['indf'] = indf
            data['indg'] = indg
            # you pad positions            
            data['pos'] = np.concatenate([pos,[[np.inf,np.inf,np.inf]]],axis=0)[None]
            # data['pos'] = np.pad(pos, ((0,45),(0,0)))[None]
            data['latvecs'] = cell[None]
            data['ewald_input_Gcuts'] = self.ewald_params[0][None]
            data['ewald_input_rcuts'] = self.ewald_params[1][None]
            data['ewald_input_realspace'] = self.ewald_params[2]
            data['ewald_input_alphas'] = self.ewald_params[3]
            data['ewald_input_recspace'] = self.ewald_params[4]
            data['linsys_masks'] = self.linsys_mask
            gauss_radii = self.radii[specs[:nats]] # shape: (Nats,)
            data['gaussian_radii'] = np.concatenate((gauss_radii,[np.inf]))[None] # shape: (1, Nats+1)
            # data['gaussian_radii'] = np.concatenate((gauss_radii,np.repeat(np.inf, 45)))[None] # shape: (1, Nats+1)
            data['Efield_ext'] = np.array(self.E_ext)
            data['phi_ext'] = np.array(self.phi_ext)
            
            mask = nn_r_LR < self.rcut_LR
            if not np.all(mask):
                inda_LR = self.inda_LR[mask]
                indb_LR = self.indb_LR[mask]
                nn_vecs_LR = nn_vecs_LR[mask]
                nn_r_LR = nn_r_LR[mask]
            else:
                inda_LR = self.inda_LR
                indb_LR = self.indb_LR
            
            if self.pad_things:
                # the following is to avoid recompiling every time
                Npairs = len(inda_LR)
                # round Npairs to the next 2*thousand
                Npad_this = np.ceil(Npairs / 2000).astype(int) * 2000 # e.g. 15636 -> 15700
                Npad = np.max([Npad_this, self.Npad_LR])
                # pad the arrays
                inda_LR = np.pad(inda_LR, (0,Npad-Npairs), constant_values=nats+1)
                indb_LR = np.pad(indb_LR, (0,Npad-Npairs), constant_values=nats+1)
                nn_r_LR = np.pad(nn_r_LR, (0,Npad-Npairs), constant_values=1e19)
                nn_vecs_LR = np.pad(nn_vecs_LR, ((0,Npad-Npairs),(0,0)), constant_values=0)
                self.Npad_LR = Npad

            data['ewald_prep'] = (inda_LR, indb_LR, nn_r_LR, nn_vecs_LR)
        return data
        
    def newnew_rebuild_pair_indices(self, atoms, cutoff_padded):
        """Vectorized rebuild — replaces the Python-loop version."""
        inda, indb, indc = neighbor_list('ijS', atoms, cutoff_padded)
        self.inda = inda.astype(np.int32)
        self.indb = indb.astype(np.int32)
        self.indc = indc.astype(np.int32)
        

    def compute_distances(self, pos, cell, inda, indb, indc):
        """Cheap per-step refresh — no cell list, just linear algebra."""
        nn_vecs = pos[indb] - pos[inda] + indc @ cell
        nn_r = np.linalg.norm(nn_vecs, axis=1)
        return nn_r, nn_vecs
    
    def new_rebuild_pair_indices(self):
        """Flatten NeighborList contents into global pair arrays.

        This is the only place with a Python loop over atoms, and it runs
        only when the neighbor list is rebuilt.
        """
        print('new rebuilding indicesss')
        pair_i = []
        pair_j = []
        pair_C = []

        natoms = len(self.atoms)
        for a in range(natoms):
            neigh, offsets = self.nl.get_neighbors(a)
            if len(neigh) == 0:
                continue

            pair_i.append(np.full(len(neigh), a, dtype=np.int32))
            pair_j.append(np.asarray(neigh, dtype=np.int32))
            pair_C.append(np.asarray(offsets, dtype=np.int32))
            
        self.inda = np.concatenate(pair_i)
        self.indb = np.concatenate(pair_j)
        self.indc = np.concatenate(pair_C, axis=0)
        
                
            
    def old_rebuild_pair_indices(self):
        print('old rebuildiiiing')
        nats = len(self.atoms)
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
        

    def parse_configfile(self, config, weights_file, species_str, descriptor_params):
        """
        Just reads the mytrain.ini file to get info on the model

        """
        
        preprocess = GvectLATTE(species=species_str,compute_dgvect=True)
        preprocess.parse_parameters(descriptor_params)
        
        
        # model: which epoch?
        folder = '/'.join(config.split('/')[0:-1])
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
            model_state = pickle.load(f)
            
        return model_state, preprocess
    
    def initialize_LR_stuff(self, gaussian_charges, species, ewald_input_params):
        # gaussian charges or point charges?
        gaussian_charges = gaussian_charges
        species_list = species
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
            
        self.acc_factor, self.Rc, self.Kc_sc, self.gaussian_smearing = ewald_input_params
        if self.ewald_acc_factor != 0: # in case you want to increase 
            self.acc_factor = self.ewald_acc_factor
            
        
            
    
        



