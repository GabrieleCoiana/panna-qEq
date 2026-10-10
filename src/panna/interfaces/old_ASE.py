###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import numpy as np
import os
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
from panna.neuralnet.config_parser_train import parse_config, parameter_file_parser,parse_network, ModelParams, example_param_parser
from panna.neuralnet.panna_model import create_panna_model
# from panna.neuralnet.LongRange_electrostatic_model import create_panna_model_with_electrostatics

class PANNACalculator(Calculator):
    """PANNA ASE Calculator"""
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        config = None,
        skin = 0.2,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        self.nl = None
        self.skin = skin
        train_config = parameter_file_parser(config)
        default_nn_config, networks_config, _ = parse_network(train_config)
        default_nn_config[1].compute_jacobian = True
        for _, nn_config in networks_config:
            nn_config.compute_jacobian = True
        g_size = networks_config[0][1].feature_size
        preprocess, extra_data = example_param_parser(config, True)
        model = os.path.dirname(config)+'/'+train_config['IO_INFORMATION'].get('network_file', None)
        self.mincut = min(preprocess.gvect['Rc_rad'],preprocess.gvect['Rc_ang'])
        self.maxcut= max(preprocess.gvect['Rc_rad'],preprocess.gvect['Rc_ang'])
        self.mapping = preprocess.species_str_2idx
        self.nsp = preprocess.number_of_species
        self.rcut = (self.maxcut)/2.0 + self.skin
        long_range_el = False

        
        # model_params, compile_params, fit_params, validation_params = \
        #     parse_config(train_config)    


        model_params = ModelParams(default_nn_config,
                                networks_config,
                                g_size,
                                False,
                                True,
                                input_format='example',
                                preprocess=preprocess,
                                long_range_el=long_range_el)
        
        

        
        self.panna_model = create_panna_model(model_params)

        dummy_batch = [np.zeros((1,1)),np.zeros((1,g_size)),np.zeros((1,1,g_size,1,3))]
        _ = self.panna_model(dummy_batch)
            
        
        self.panna_model.load_weights(model).expect_partial()
        
    def atoms2dict(self, nats, pos, cell, specs, nndata):
        data = {}
        data['species'] = specs
        data['nats'] = np.asarray([nats],dtype=np.int32)
        data['positions'] = np.asarray([pos])

        # Reshaping to the species, and padding
        nnoi = [np.asarray(n[0]) for n in nndata]
        nnos = [np.asarray(n[1]) for n in nndata]
        nnsp = [[np.where(specs[0][ni]==s)[0] for s in range(self.nsp)] for ni in nnoi]
        lenn = [[len(sp) for sp in sps] for sps in nnsp]
        nnmax = np.max(lenn)
        nni = np.asarray([[np.pad(ni[sp],(0,nnmax-n),'constant',constant_values=aa) for n,sp in zip(ns,sps)] \
                                                         for aa,(ns,sps,ni) in enumerate(zip(lenn,nnsp,nnoi))])
        nnsh = np.asarray([[np.concatenate((nsh[sp],np.zeros((nnmax-n,3))),axis=0) for n,sp in zip(ns,sps)] \
                                                                        for ns,sps,nsh in zip(lenn,nnsp,nnos)])
        
        # Vecs to nn (nats,nsp,nnmax,3), and r
        vec = pos[nni]+nnsh@cell-np.reshape(pos,[nats,1,1,3])
        rij = np.linalg.norm(vec,axis=3)
        # Compute actual nn within both cutoffs, but we keep the whole stack, since skin in thin
        rmin = rij < self.mincut
        nnm1 = np.logical_and(rmin, rij > 1e-8)
        nnum1 = np.sum(nnm1,axis=2)
        # Mask for each cutoff (1,nats,nsp,nmax)
        data['mask1'] = np.expand_dims(nnm1.astype(np.float32),axis=0)
        if self.mincut != self.maxcut:
            nnm2 = np.logical_and(rij < self.maxcut, np.logical_not(rmin))
            nnum = nnum1+np.sum(nnm2,axis=2)
            data['nn_num'] = np.expand_dims(np.stack((nnum1,nnum),axis=-1),axis=0).astype(np.int32)
            data['mask2'] = np.expand_dims(np.logical_or(nnm1,nnm2).astype(np.float32),axis=0)
        else:
            data['nn_num'] = np.expand_dims(np.stack((nnum1,nnum1),axis=-1),axis=0).astype(np.int32)
            data['mask2'] = data['mask1']

        # Indices of actual nn (first and second cutoff) (1,nats,nsp,nmax)
        data['nn_inds'] = np.expand_dims(nni.astype(np.int32),axis=0)
        # Vectors (1,nats,nsp,nmax,3) and radii (1,nats,nsp,nmax)
        data['nn_vecs'] = np.expand_dims(vec,axis=0)
        data['nn_r'] = np.expand_dims(rij,axis=0)

        # Dummies for the dict
        data['energy'] = np.zeros((1,1))
        data['forces'] = np.zeros((1,nats,3))
        return data
        
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms)
        nats = np.asarray(atoms.get_global_number_of_atoms())
        pos = np.asarray(atoms.get_positions())
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([[self.mapping[a] for a in atoms.get_chemical_symbols()]],dtype=np.int32)
        if not self.nl:
            self.nl = NeighborList(self.rcut*np.ones(nats),
                                   skin=self.skin,
                                   bothways=True,
                                   primitive=NewPrimitiveNeighborList)
        self.nl.update(atoms)
        # print("Calc called!")
        # Creating input data, adding extra dim for batch (=1)
        nndata = [self.nl.get_neighbors(i) for i in range(nats)]
        data = self.atoms2dict(nats,pos,cell,specs,nndata)
        out = self.panna_model(data)
            
        self.results = {
            "energy": out[1][0],
            "forces": np.reshape(out[2],(nats,3))
        }


from panna.gvector.LR_electrostatic_term import Ewald_summation as Ewald_panna
from panna.gvect_calculator import extract_gvect_func
import configparser
from panna.lib.parser_callable import converters
from panna.neuralnet.LongRange_electrostatic_model_gabri import create_panna_model_with_electrostatics


        
class PANNACalculatorLR(Calculator):
    """PANNA ASE Calculator"""
    implemented_properties = ["energy", "forces", "charges"]
    
    def __init__(
        self,
        config = None,
        PBC=True,
        long_range_el = True,
        **kwargs):

        Calculator.__init__(self, **kwargs)
        train_config = parameter_file_parser(config)
        default_nn_config, networks_config, _ = parse_network(train_config)
        default_nn_config[1].compute_jacobian = True
        for _, nn_config in networks_config:
            nn_config.compute_jacobian = True
        g_size = networks_config[0][1].feature_size
        preprocess, extra_data = example_param_parser(config, True)
        model = os.path.dirname(config)+'/'+train_config['IO_INFORMATION'].get('network_file', None)
        self.mapping = preprocess.species_str_2idx

        self.e_field = 0.
        self.PBC = PBC
        self.long_range_el = long_range_el
        self.covalent_radii = train_config['LONG_RANGE'].get_comma_list_floats('covalent_radii', None)
        print('ASE - cov radii', self.covalent_radii)
        
        
        # model_params, compile_params, fit_params, validation_params = \
        #     parse_config(train_config)    

        
        model_params = ModelParams(default_nn_config,
                                networks_config,
                                g_size,
                                False,
                                True,
                                input_format='example',
                                preprocess=preprocess,
                                long_range_el=long_range_el)
        

        
        # Parsing gvector parameters
        # config = '../panna/inputs/mygvect_forces_lr_test.ini'
        configg = configparser.ConfigParser(converters=converters)
        configg.read(config)
        self.gvect_func = extract_gvect_func(configg)


        # self.gvect_func = GvectmBP(compute_dgvect=True,
        #                               species=['Mg', 'O'],
        #                               sparse_dgvect=False,
        #                               pbc_directions=[True, True, True])
        # # Every gvect type may have different param set, needs its own parser to parse the config
        # self.gvect_func.parse_parameters(configg['GVECT_PARAMETERS'])
        

        self.panna_model = create_panna_model_with_electrostatics(model_params)
        batch_size = 1
        Nats = 1
        dummy_batch = [[np.zeros((batch_size,Nats)), np.zeros((batch_size,Nats,g_size)), np.zeros((batch_size, Nats*Nats*g_size*3))],
                        [np.zeros((batch_size, Nats**2)), np.zeros((batch_size)), np.zeros((batch_size,Nats**2*3))]]
        
        _ = self.panna_model(dummy_batch)

        self.panna_model.load_weights(model).expect_partial()
        
        return 
    
    def set_e_field(self, e_field):
        self.e_field = e_field
    
    
        
    def calculate(self, atoms=None, properties=['energy', 'forces', 'charges'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms)
        
        key = ''
        # nats = np.asarray(atoms.get_global_number_of_atoms())
        pos = np.asarray(atoms.get_positions())
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()],dtype=np.int32)
        specs_str = atoms.get_chemical_symbols()
        
        
        output_net = self.gvect_func(key, pos, specs, cell)
        gvect, dgvect = output_net['Gvect'], output_net['dGvect']
        
        
        accuracy = 1e-06
        output_el = Ewald_panna(key,
                      pos, 
                      cell, 
                      accuracy, 
                      specs_str,
                      PBC=self.PBC,
                      kmax=None, 
                      rs=None,
                      compute_forces=True,
                      covalent_radii=self.covalent_radii)
        
        
        

        V, Vp = output_el['el_energy_kernel'], output_el['el_force_kernel']
        tot_charge = np.zeros((1))

        
        
        gvect = np.expand_dims(gvect, 0)
        dgvect = np.expand_dims(dgvect, 0).reshape((1, -1))
        V = np.expand_dims(V, 0).reshape((1,-1))
        Vp = np.expand_dims(Vp, 0).reshape((1,-1))
        specs = np.expand_dims(specs, 0)


        data = [ [specs,gvect, dgvect] , [V, tot_charge, Vp] ]
    
        out = self.panna_model(data, e_field=self.e_field)
        
        
            
        self.results = {
            "energy": out[1][0].numpy(),
            "forces": out[2].numpy().reshape((-1,3)),
            "charges": out[3].numpy().flatten()
        }
        
        
        
        
from panna.gvector.LR_electrostatic_term_gabri_pointcharges import Ewald_summation as Ewald_gabri
from panna.gvect_calculator import extract_gvect_func
import configparser
from panna.lib.parser_callable import converters
from panna.neuralnet.LongRange_electrostatic_model_gabri import create_panna_model_with_electrostatics


        
class PANNACalculatorLRgabri(Calculator):
    """PANNA ASE Calculator"""
    implemented_properties = ["energy", "forces", "charges"]
    
    def __init__(
        self,
        config = None,
        PBC=True,
        long_range_el = True,
        kmax = None,
        gaussian_std = None,
        Rc = None,
        verbose=True,
        **kwargs):

        Calculator.__init__(self, **kwargs)

        train_config = parameter_file_parser(config)
        default_nn_config, networks_config, _ = parse_network(train_config)
        default_nn_config[1].compute_jacobian = True
        for _, nn_config in networks_config:
            nn_config.compute_jacobian = True
        g_size = networks_config[0][1].feature_size
        preprocess, extra_data = example_param_parser(config, True)
        model = os.path.dirname(config)+'/'+train_config['IO_INFORMATION'].get('network_file', None)
        self.mapping = preprocess.species_str_2idx

        self.e_field = 0.
        self.PBC = PBC
        self.long_range_el = long_range_el
        self.kmax = kmax
        # self.gaussian_std = gaussian_std
        self.alpha = 1/(np.sqrt(2) * gaussian_std) if gaussian_std is not None else None
        self.Rc = Rc
        self.verbose = verbose
        
        
        # model_params, compile_params, fit_params, validation_params = \
        #     parse_config(train_config)    

        
        model_params = ModelParams(default_nn_config,
                                networks_config,
                                g_size,
                                False,
                                True,
                                input_format='example',
                                preprocess=preprocess,
                                long_range_el=long_range_el)
        

        # Parsing gvector parameters
        # config = '../panna/inputs/mygvect_forces_lr_test.ini'
        configg = configparser.ConfigParser(converters=converters)
        configg.read(config)
        self.gvect_func = extract_gvect_func(configg)


        self.panna_model = create_panna_model_with_electrostatics(model_params)
        batch_size = 1
        Nats = 1
        dummy_batch = [[np.zeros((batch_size,Nats)), np.zeros((batch_size,Nats,g_size)), np.zeros((batch_size, Nats*Nats*g_size*3))],
                        [np.zeros((batch_size, Nats**2)), np.zeros((batch_size)), np.zeros((batch_size,Nats**2*3))]]
        
        print('IIIIIIIIIIIIIII')
        # _ = self.panna_model(dummy_batch)

        self.panna_model.load_weights(model).expect_partial()
        
        return 
    
    def set_e_field(self, e_field):
        self.e_field = e_field
    
        
    def calculate(self, atoms=None, properties=['energy', 'forces', 'charges'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms)
        
        key = ''
        # nats = np.asarray(atoms.get_global_number_of_atoms())
        pos = np.asarray(atoms.get_positions())
        cell = np.asarray(atoms.get_cell())
        specs = np.asarray([self.mapping[a] for a in atoms.get_chemical_symbols()],dtype=np.int32)
        specs_str = atoms.get_chemical_symbols()
        
        
        output_net = self.gvect_func(key, pos, specs, cell)
        gvect, dgvect = output_net['Gvect'], output_net['dGvect']
        
        
        accuracy = 1e-06
        # Rc = np.max(np.linalg.norm(cell, axis=1))

        
        output_el = Ewald_gabri(key,
                      pos, 
                      cell, 
                      accuracy, 
                      specs_str,
                      PBC=self.PBC,
                      kmax=self.kmax, 
                      rs=None,
                      compute_forces=True,
                      alpha=self.alpha, 
                      Rc=self.Rc,
                      verbose=self.verbose)
        

        V, Vp = output_el['el_energy_kernel'], output_el['el_force_kernel']
        tot_charge = np.zeros((1))

                
        gvect = np.expand_dims(gvect, 0)
        dgvect = np.expand_dims(dgvect, 0).reshape((1, -1))
        V = np.expand_dims(V, 0).reshape((1,-1))
        Vp = np.expand_dims(Vp, 0).reshape((1,-1))
        specs = np.expand_dims(specs, 0)
        data = [ [specs,gvect, dgvect] , [V, tot_charge, Vp] ]
        
        
        out = self.panna_model(data, e_field=self.e_field)
            
        self.results = {
            "energy": out[1][0].numpy(),
            "forces": out[2].numpy().reshape((-1,3)),
            "charges": out[3].numpy().flatten()
        }