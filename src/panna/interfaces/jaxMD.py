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
import jax_md
import jax
import jax.numpy as jnp
import haiku as hk
import os
import pickle
from mendeleev import element
from panna.electrostatics.Electrostatics_jax_pol import create_masks_pol    
from panna.electrostatics.Electrostatics_jax_qq import set_ewald_parameters, create_masks
from panna.electrostatics.Electrostatics_jax_qq_gc import set_ewald_parameters_gaussians



class PANNAJAXMdCalculator:

    def __init__(self,
                 config,
                 box,
                 species,
                 capacity_multiplier=1.25,
                 skin=0.5,
                 weights_file=-1,
                 verbose=True,
                 rcut_LR = 0, # default for LR cutoff
                 ewald_acc_factor = 0, 
                 phi_ext=[], E_ext=[]):
        # ---- attributes ----
        self.box = box
        self.species = species
        self.verbose = verbose
        self.phi_ext = phi_ext
        self.E_ext = E_ext
        self.rcut_LR = rcut_LR
        self.ewald_acc_factor = ewald_acc_factor
        # --------------------
        
        # ---- load config / model ----
        parameters = trainjax_parameter_parser(config, train=False)
        # forcing some parameters
        parameters['task'] = 'validation'
        parameters['precompute_ewald_sums'] = False
        parameters['BORN'] = ''
        preprocess = GvectLATTE(species=parameters['species_str'], compute_dgvect=True)
        preprocess.parse_parameters(parameters['descriptor_params'])
        raw_model = make_model(parameters, pre=preprocess)
        model = hk.without_apply_rng(hk.transform(raw_model))
        modapp = jax.jit(model.apply)

        wfolder = os.path.dirname(config) + '/models/'
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
            model_state = pickle.load(f)
        # -----------------------------
        
        # ---- NN function ----
        nats = len(species)
        nsp = len(parameters['species'])
        mapping = dict(zip(parameters['species'], range(nsp)))
        internal_species = jnp.asarray([mapping[s] for s in species] + [0], dtype=jnp.int32)
        rcut = parameters['cutoff']
        self.neighbor_fn = jax_md.custom_partition.neighbor_list_multi_image(
            None,
            box,
            rcut,
            capacity_multiplier=capacity_multiplier,
            dr_threshold=skin,
            format=jax_md.partition.Sparse,
            fractional_coordinates=False)
        # --------------------
        
        # ---- LR stuff ----
        indf, indg = 0, 0
        self.neighbor_fn_LR = 0
        self.is_LR_model = 'LR' in parameters['model_type']
        self.long_range_pol = 'pol' in parameters['model_type'] # parameters['model_type'] in ['MLP_LR_pol' ,'MLP_LR_fq_pol', 'MLP_LR_fq_pol_modsr']
        if self.is_LR_model:
            self.initialize_LR_stuff(parameters['gaussian_charges'], parameters['species'], parameters['ewald_input_params'])

            # rcut_LR = parameters['ewald_input_params'][1] if parameters['ewald_input_params'][1] != 0 else jnp.linalg.det(box)**(1/3)
            rcut_LR = self.rcut_LR if self.rcut_LR != 0 else jnp.linalg.det(box)**(1/3)
            self.neighbor_fn_LR = jax_md.custom_partition.neighbor_list_multi_image(
                None,
                box,
                rcut_LR,
                capacity_multiplier=capacity_multiplier,
                dr_threshold=rcut_LR/10,
                format=jax_md.partition.Sparse,
                fractional_coordinates=False)
 
        
            # then we need additional two sets of indices
            indf = jnp.arange(nats)             # atom index within structure 
            indg = jnp.arange(nats+1, nats*4+1) # atom index within structure 
            
            if self.long_range_pol:
                self.linsys_mask = jnp.array(create_masks_pol( (nats,), nats+1, n=4))
            else: # it's a charge-only model
                self.linsys_mask = jnp.array(create_masks((nats,), nats+1))
                
            
            # and we need to set the Ewald parameters
            if not self.gaussian_charges: # then it's point charges
                self.ewald_params = set_ewald_parameters(nats, box, self.acc_factor, Rc=rcut_LR, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
            else: # it's gaussian charges
                self.ewald_params = set_ewald_parameters_gaussians(nats, box, self.radii, self.acc_factor, Rc=rcut_LR, gaussian_smearing=self.gaussian_smearing, Kc_sc=self.Kc_sc, verbose=self.verbose)
        # -------------------   
        
        

        # --- Closures over local variables only (no `self` in the traced call) ---
        def energy_force_fn(pos, box, neighbors, neighbors_LR):
            #pos = jnp.concatenate((pos,jnp.array([[1e19, 1e19, 1e19]])),axis=0)
            inda, indb = neighbors.idx
            shifts_real = neighbors.shifts @ box.T
            
            nn_vecs = pos[indb] + shifts_real - pos[inda]
            nn_r = jnp.linalg.norm(nn_vecs, axis=1)

            data = {
                'species': internal_species[:nats],
                'nats': jnp.asarray([nats]),
                'ntot': nats,
                'inda': inda,
                'sp_a': internal_species[inda],
                'indb': indb,
                'sp_b': internal_species[indb],
                'inde': jnp.zeros(nats, dtype=jnp.int32),
                'inds': jnp.zeros(len(inda), dtype=jnp.int32),
                'nn_vecs': nn_vecs,
                'nn_r': nn_r,
                'latvecs': box[None]}
            
            # LR stuff
            if self.is_LR_model:
                pos = jnp.concatenate((pos,jnp.array([[1e19, 1e19, 1e19]])),axis=0)
                data['indf'] = indf
                data['indg'] = indg
                # you pad positions            
                data['pos'] = pos[None]
                # data['pos'] = np.pad(pos, ((0,45),(0,0)))[None]
                data['latvecs'] = box[None]
                data['ewald_input_Gcuts'] = self.ewald_params[0][None]
                data['ewald_input_rcuts'] = self.ewald_params[1][None]
                data['ewald_input_realspace'] = self.ewald_params[2]
                data['ewald_input_alphas'] = self.ewald_params[3]
                data['ewald_input_recspace'] = self.ewald_params[4]
                data['linsys_masks'] = self.linsys_mask
                gauss_radii = self.radii[internal_species[:nats]] # shape: (Nats,)
                data['gaussian_radii'] = jnp.concatenate((gauss_radii,jnp.array([1e19])))[None] # shape: (1, Nats+1)
                # data['gaussian_radii'] = np.concatenate((gauss_radii,np.repeat(np.inf, 45)))[None] # shape: (1, Nats+1)
                data['Efield_ext'] = jnp.array(self.E_ext)
                data['phi_ext'] = jnp.array(self.phi_ext)

                inda_LR, indb_LR = neighbors_LR.idx
                shifts_real = neighbors_LR.shifts @ box.T
                xyzij = pos[inda_LR] - shifts_real - pos[indb_LR]   
                rij = jnp.linalg.norm(xyzij, axis=1)

                mask = rij < rcut_LR
                indi = jnp.where(mask, size=mask.size)
                rij = rij[indi]
                xyzij = xyzij[indi]
                inda_LR = inda_LR[indi]      
                indb_LR = indb_LR[indi]      

                n_valid = jnp.sum(mask)
                valid = jnp.arange(mask.size) < n_valid
                rij = jnp.where(valid, rij, 1e18)
                xyzij = jnp.where(valid[:, None], xyzij, 0.)
                rij = jnp.where(rij < 1e-09, 1e19, rij)
                inda_LR = jnp.where(valid, inda_LR, nats)   # send padding slots to the dummy/sentinel atom
                indb_LR = jnp.where(valid, indb_LR, nats)   # so their scatter contributions land on the padded row

                data['ewald_prep'] = (inda_LR, indb_LR, rij, xyzij)

            out = modapp(model_state, data)
            return out

        @jax.jit
        def energy_fn(coords, box, neighbors, neighbors_LR):
            print('jaxMD - Energy compilation')
            return energy_force_fn(coords, box, neighbors, neighbors_LR)[0][0]

        @jax.jit
        def force_fn(coords, box, neighbors, neighbors_LR):
            print('jaxMD - Force compilation')
            return energy_force_fn(coords, box, neighbors, neighbors_LR)[1]

        self.energy_fn = energy_fn
        self.force_fn = force_fn
        self.energy_force_fn = energy_force_fn  # unjitted, exposed if you want [E, F] together
        self.rcut = rcut

    def initialize(self):
        return self.neighbor_fn, self.neighbor_fn_LR, self.energy_fn, self.force_fn
    
    def initialize_LR_stuff(self, gaussian_charges, species, ewald_input_params):
        # gaussian charges or point charges?
        gaussian_charges = gaussian_charges
        species_list = species
        if gaussian_charges[0] == 'covalent':
            radii = [element(x).covalent_radius * 0.01 * jnp.sqrt(2.0) for x in species_list]
        elif gaussian_charges[0] == 'ionic':
            radii = [element(x).ionic_radii[0].ionic_radius * 0.01 * jnp.sqrt(2.0) for x in species_list]
        elif gaussian_charges[0] in [b'False', 'False']:
            radii = [0. for x in species_list]
        else:
            radii = [float(x) for x in gaussian_charges]
        self.radii = jnp.array(radii) # (nspecies,)
        self.gaussian_charges = jnp.any(self.radii > 0)
        
        print('INFO - Gaussian charges is ', self.gaussian_charges)
        if self.gaussian_charges:
            print('\t Gauss radii', jnp.round(self.radii,3))
            
        self.acc_factor, self.Rc, self.Kc_sc, self.gaussian_smearing = ewald_input_params
        if self.ewald_acc_factor != 0: # in case you want to increase 
            self.acc_factor = self.ewald_acc_factor






# def run_MD():
    














def PANNAJAXMd(config, 
               box,
               species,
               capacity_multiplier=1.25,
               dr_threshold=0.0, 
               weights_file=-1):
    """
    Old function. Now replaced with class
    This works only for SR models. To enable LR models, I decided to build a class.

    """
    # Load configuration
    parameters = trainjax_parameter_parser(config, train=False)
    # Create descriptor
    preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
    preprocess.parse_parameters(parameters['descriptor_params'])
    # Make model and apply function
    raw_model = make_model(parameters, pre=preprocess)
    model = hk.without_apply_rng(hk.transform(raw_model))
    modapp = jax.jit(model.apply)
    # Load parameters
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
        model_state = pickle.load(f)
    # Create useful quantities
    nats = len(species)
    nsp = len(parameters['species'])
    mapping = dict(zip(parameters['species'], range(nsp)))
    internal_species = jnp.asarray([mapping[s] for s in species]+[0],dtype=jnp.int32)
    rcut = parameters['cutoff']
    # Create neighbor function, requesting sparse list
    # neighbor_fn = jax_md.partition.neighbor_list(displacement_fn, box, rcut, \
    #                                              capacity_multiplier=capacity_multiplier, \
    #                                              dr_threshold=dr_threshold, \
    #                                              format=jax_md.partition.Sparse)
    neighbor_fn = jax_md.custom_partition.neighbor_list_multi_image(
                                    None, 
                                    box, 
                                    rcut,
                                    capacity_multiplier=capacity_multiplier, 
                                    dr_threshold=dr_threshold,
                                    format=jax_md.partition.Sparse,
                                    fractional_coordinates=False)
    

    # The main function calling the model
    def energy_force_fn(coords, box, neighbors, rcut):
        # Creating the input dict
        # nn_vecs = jax.vmap(displacement_fn)(coords[neighbors.idx[1]],coords[neighbors.idx[0]])
        # nn_r = jnp.sqrt(jnp.sum(nn_vecs**2,axis=1))
        inda, indb = neighbors.idx
        # shifts.shifts is padded with atom-index N for invalid slots but shift vectors
        # themselves are still well-defined (usually 0) there; mask them out below.
        shifts_real = neighbors.shifts @ box.T     # same convention jaxMD uses internally
        
        nn_vecs = coords[indb] + shifts_real - coords[inda]   # r_j + shift - r_i
        nn_r = jnp.linalg.norm(nn_vecs, axis=1) # this has shape (7,)
        
        # # jaxMD considers "self" neighbours terms, i.e. where nn_r = 0
        # # plus, it pads
        # mask = (nn_r > 1e-09) # (nn_r < rcut) &  
        # # if not jnp.all(mask):
        # inda = jnp.where(mask, inda, 9999)
        # indb = jnp.where(mask, indb, 9999)
        # # indc = indc[mask]
        # nn_vecs = jnp.where(mask[:,None], nn_vecs, 9999)
        # nn_r = jnp.where(mask, nn_r, 9999)
        # # sp_a = specs[inda[mask]]
        # # sp_b = specs[indb[mask]]
        data = {
            'species': internal_species[:nats],
            'nats': jnp.asarray([nats]),
            'ntot': nats,
            'inda': inda,
            'sp_a': internal_species[inda],
            'indb': indb,
            'sp_b': internal_species[indb],
            'inde': jnp.zeros(nats,dtype=jnp.int32),
            'inds': jnp.zeros(len(inda),dtype=jnp.int32),
            'nn_vecs': nn_vecs,
            'nn_r': nn_r,
            'latvecs': box[None]}
        # Model call, out is [E, F]
        out = modapp(model_state, data)
        return out#, data

    # Compiled function to get only energy
    @jax.jit
    def energy_fn(coords, box, neighbors):
        return energy_force_fn(coords, box, neighbors, rcut)[0][0]

    # Compiled function to get only force
    @jax.jit
    def force_fn(coords, box, neighbors):
        return energy_force_fn(coords, box, neighbors, rcut)[1]

    return neighbor_fn, energy_fn, force_fn



# def PANNAJAX_neighbor_list_old_asitwas(displacement_fn, config, species, box,
#                            capacity_multiplier=1.25,
#                            dr_threshold=0.0, weights_file=-1):
#     # Load configuration
#     parameters = trainjax_parameter_parser(config)
#     # Create descriptor
#     preprocess = GvectLATTE(species=parameters['species_str'],compute_dgvect=True)
#     preprocess.parse_parameters(parameters['descriptor_params'])
#     # Make model and apply function
#     raw_model = make_model(parameters, pre=preprocess)
#     model = hk.without_apply_rng(hk.transform(raw_model))
#     modapp = jax.jit(model.apply)
#     # Load parameters
#     wfolder = os.path.dirname(config)+'/models/'
#     if weights_file == -1:
#         last_file = max(
#         (f for f in os.listdir(wfolder) if f.startswith("epoch_")),
#         key=lambda f: int(f.split("_")[1])
#         )
#         wfile = wfolder + last_file
#     else:
#         wfile = wfolder + weights_file
#     print('INFO - Getting model', wfile)
#     with open(wfile, 'rb') as f:
#         model_state = pickle.load(f)
#     # Create useful quantities
#     nats = len(species)
#     nsp = len(parameters['species'])
#     mapping = dict(zip(parameters['species'], range(nsp)))
#     internal_species = jnp.asarray([mapping[s] for s in species]+[0],dtype=jnp.int32)
#     rcut = parameters['cutoff']
#     # Create neighbor function, requesting sparse list
#     neighbor_fn = jax_md.partition.neighbor_list(displacement_fn, box, rcut, \
#                                                  capacity_multiplier=capacity_multiplier, \
#                                                  dr_threshold=dr_threshold, \
#                                                  format=jax_md.partition.Sparse)

#     # The main function calling the model
#     def energy_force_fn(coords, neighbors):
#         # Creating the input dict
#         displ = jax.vmap(displacement_fn)(coords[neighbors.idx[1]],coords[neighbors.idx[0]])
#         rads = jnp.sqrt(jnp.sum(displ**2,axis=1))
#         inda, indb = neighbors.idx
#         data = {
#             'species': internal_species[:nats],
#             'nats': jnp.asarray([nats]),
#             'ntot': nats,
#             'inda': inda,
#             'sp_a': internal_species[inda],
#             'indb': indb,
#             'sp_b': internal_species[indb],
#             'inde': jnp.zeros(nats,dtype=jnp.int32),
#             'nn_vecs': displ,
#             'nn_r': rads}
#         # Model call, out is [E, F]
#         out = modapp(model_state, data)
#         return out

#     # Compiled function to get only energy
#     @jax.jit
#     def energy_fn(coords, neighbors):
#         return energy_force_fn(coords, neighbors)[0][0]

#     # Compiled function to get only force
#     @jax.jit
#     def force_fn(coords, neighbors):
#         return energy_force_fn(coords, neighbors)[1]

#     return neighbor_fn, energy_fn, force_fn
