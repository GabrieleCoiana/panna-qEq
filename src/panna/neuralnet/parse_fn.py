"""Utilities to handling the input system
"""

import tensorflow as tf
import tensorflow.io as tfio
from panna.gvector.pbc import make_replicas
from panna.lib import ExampleJsonWrapper
import numpy as np
# import jax.numpy as jnp
import random
import os
from functools import partial
import jax
import time

# Point charges
from panna.electrostatics.Electrostatics_jax_qq import Ewald_summation, create_masks, set_ewald_parameters, set_ewald_parameters_for_batch, Electrostatics_finite_systems, recewald_qq_jax_vect_gaussians, Electrostatics_finite_systems_gaussians # for LR
# Gaussian charges
from panna.electrostatics.Electrostatics_jax_qq_gc import set_ewald_parameters_gaussians_for_batch, set_ewald_parameters_gaussians
from panna.electrostatics.Electrostatics_jax_qq_gc import Ewald_summation as ewald_gaussians

from mendeleev import element

try:
    import h5py
except ImportError:
    pass

class ParseFn():
    """Parse TFExample records and perform simple data augmentation.

    Parameters
    ----------
        g_size: int
            size of the g_vector
        zeros:
            array of zero's one value per specie.
        n_species: int
            number of species
        forces: boolean
            recover forces AND dg/dx
        energy_rescale: float
            scale the energy

    """
    def __init__(self,
                 g_size: int,
                 n_species: int,
                 forces: bool = False,
                 sparse_dgvect: bool = False,
                 energy_rescale: float = 1.0,
                 long_range_el: bool = False,
                 names: bool = False):
        self._g_size = g_size
        self._n_species = n_species
        self._forces = forces
        self._sparse_dgvect = sparse_dgvect
        self._energy_rescale = energy_rescale
        self._long_range_el = long_range_el
        self._names = names


    @property
    def feature_description(self):
        """Features of the example."""
        feat = {}
        feat["energy"] = tfio.FixedLenFeature([], dtype=tf.float32)
        feat["species"] = tfio.FixedLenSequenceFeature([],
            dtype=tf.int64,
            allow_missing=True,
            default_value=self._n_species)
        feat["gvects"] = tfio.FixedLenSequenceFeature(
            shape=[self._g_size],
            dtype=tf.float32,
            allow_missing=True)
        if self._names:
            feat["name"] = tfio.FixedLenFeature([], dtype=tf.dtypes.string,
                                                default_value=b'N.A.')
        if self._forces:
            if self._sparse_dgvect:
                feat["dgvect_size"] = tfio.FixedLenFeature([], dtype=tf.int64)
                feat["dgvect_values"] = tfio.FixedLenSequenceFeature([],
                    dtype=tf.float32,
                    allow_missing=True)
                feat["dgvect_indices1"] = tfio.FixedLenSequenceFeature([],
                    dtype=tf.float32,
                    allow_missing=True)
                feat["dgvect_indices2"] = tfio.FixedLenSequenceFeature([],
                    dtype=tf.float32,
                    allow_missing=True)
            else:
                feat["dgvects"] = tfio.FixedLenSequenceFeature([],dtype=tf.float32,
                                                               allow_missing=True)
            feat["forces"] = tfio.FixedLenSequenceFeature([],
                dtype=tf.float32,
                allow_missing=True)
        if self._long_range_el:
            feat['el_energy_kernel'] = tfio.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True, default_value=0.0)
            if self._forces:
                feat['el_force_kernel'] = tfio.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True, default_value=0.0)
            feat['atomic_charges'] = tfio.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True, default_value=0.0)
            feat['total_charge'] = tfio.FixedLenFeature([], dtype=tf.float32)



        return feat

    def _post_processing(self, example):
        example['energy'] = example['energy'] * self._energy_rescale
        return example

    def __call__(self, serialized):
        """Return a sample ready to be batched.

        Return
        ------
            species_tensor: Sparse Tensor, (n_atoms) value in range(n_species)
            g_vectors_tensor: Sparse Tensor, (n_atoms, g_size)
            energy: true energy value corrected with the zeros
        """
        examples = tfio.parse_example(serialized, features=self.feature_description)
        examples = self._post_processing(examples)
        return examples


def load_json(data_dir, mincut, maxcut, species_list, get_forces):
    all_examples = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == b'.example':
            all_examples.append(file)
    if len(all_examples) == 0:
        raise ValueError('No example file found.')
    # Species list decoded because it comes as b'' string
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    for ex in all_examples:
        example = ExampleJsonWrapper(os.path.join(data_dir,ex),species_list)
        nats = len(example.angstrom_positions)
        species = example.species_indexes
        positions, replicas = make_replicas(example.angstrom_positions, \
                                  example.angstrom_lattice_vectors, maxcut)
        nrep = len(replicas)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,nrep,3))
        rij_vec = posj+rep-posi
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))

        # Creating and filling all lists, padding where necessary (~ dims are padded)
        # We explicitly separate species because it simplifies the g construction
        # Mask with the first cutoff
        radial_mask1 = np.logical_and(rij < mincut, rij > 1e-8)
        # Mask with the second cutoff (but greater than first)
        radial_mask2 = np.logical_and(rij < maxcut, rij >= mincut)
        # Mask of nn species type
        species_mask = [np.reshape(species==s,(nats,1)) for s in range(nsp)]
        # Looping like this to have [n_atoms, n_species] order
        bool_mask1 = [[np.logical_and(r1,sm) for sm in species_mask] for r1 in radial_mask1]
        bool_mask2 = [[np.logical_and(r2,sm) for sm in species_mask] for r2 in radial_mask2]
        inds = [[np.where(m1)+np.where(m2) for m1, m2  in zip(sm1, sm2)] \
                    for sm1, sm2 in zip(bool_mask1,bool_mask2)]
        # Number of nn per atom and species, 2 cutoffs, shape [n_atoms,n_species,2]
        nn_num = [[[len(i[0]),len(i[0])+len(i[2])] for i in ii] for ii in inds]
        maxind = np.max(nn_num)
        # Indices of nn, shape [n_atoms,n_species,~nn]
        nn_inds = [[np.concatenate((i[0],i[2],np.zeros(maxind-n[1],dtype=np.int32))) \
                       for i,n in zip(ii,nn)] for ii,nn in zip(inds,nn_num)]
        # Vectors to nns, shape [n_atoms,n_species,~nn,3]
        nn_vecs = [[np.concatenate((r[i[0],i[1]],r[i[2],i[3]],np.zeros((maxind-n[1],3))), axis=0) \
                    for i,n in zip(ii,nn)] for r, ii, nn in zip(rij_vec,inds,nn_num)]
        # Distances of nns, shape [n_atoms,n_species,~nn]
        nn_r = [[np.concatenate((r[i[0],i[1]],r[i[2],i[3]],np.zeros(maxind-n[1]))) \
                    for i,n in zip(ii,nn)] for r, ii, nn in zip(rij,inds,nn_num)]
        # Masks for both radii, shape [n_atoms,n_species,~nn]
        mask1 = [[np.concatenate((np.ones(n[0]),np.zeros(maxind-n[0]))) for n in nn] for nn in nn_num]
        mask2 = [[np.concatenate((np.ones(n[1]),np.zeros(maxind-n[1]))) for n in nn] for nn in nn_num]

        data_dict = {'nats': nats,
                     'species': species,
                     'positions': positions,
                     'nn_inds': nn_inds,
                     'nn_num': nn_num,
                     'nn_vecs': nn_vecs,
                     'nn_r': nn_r,
                     'mask1': mask1,
                     'mask2': mask2,
                     'energy': example.ev_energy,
                     'name': ex[:-8]}
        if get_forces:
            data_dict['forces'] = example.forces
        yield data_dict

# Some consts
sym2num = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 
'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 
'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 
'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 
'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 
'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 
'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 
'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 
'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 
'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 
'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 
'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 
'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 
'Lv': 116, 'Ts': 117, 'Og': 118}
bohr2A = np.float32(0.52917721067)
Ha2eV = np.float32(2*13.6056980659)

# Function to pad a batch to a given number of atoms and pairs
def pad_batch(exout, at_mod, ntot, pair_mod, pair_max, ptot, nsp, batch_size, ewald_inputs=[]):
    # Extra atoms (could be 0), but we don't let the batch grow
    extra_at = (at_mod-(ntot%at_mod))%at_mod
    if extra_at>0:
        exout['species'].extend(nsp*np.ones(extra_at,dtype=np.int32))
        exout['inde'].extend(batch_size*np.ones(extra_at,dtype=np.int32))
    if 'forces' in exout.keys():
        exout['forces'].extend(np.zeros((extra_at,3)))
        # TODO: is it necessary to pad so much?!
    # if 'charges' in exout.keys():
    #     exout['charges'].extend(np.zeros((extra_at)))
    # if 'dipoles' in exout.keys():
    #     exout['dipoles'].extend(np.zeros((extra_at,3)))
    
    # print(pair_max, 'pmax')
    # Extra pair (could be 0)
    if pair_max == 0: # apply Franco's logics
        extra_pair = (pair_mod-(ptot%pair_mod))%pair_mod
    else:
        if ptot <= pair_max:
            extra_pair = pair_max - ptot
            # print(f'ptot {ptot} pair_max {pair_max} extra_pair {extra_pair}')
        else:
            # print(f'ptot {ptot} pair_max {pair_max} ')
            # round pair_tot to the next thousand, e.g. 15636 -> 16000
            ptot_r = np.ceil(ptot/1000).astype(int) * 1000
            extra_pair = ptot_r - ptot
            # print(f'ptot rounded {ptot_r} extra_pair {extra_pair}')
    
    if extra_pair>0:
        ones = np.ones(extra_pair,dtype=np.int32)
        exout['nn_r'].extend(np.ones(extra_pair))
        exout['nn_vecs'].extend(np.ones((extra_pair,3)))
        exout['sp_a'].extend(nsp*ones)
        exout['sp_b'].extend(nsp*ones)
        exout['inda'].extend(ntot*ones)
        exout['indb'].extend(ntot*ones)
        if 'stress' in exout.keys():
            exout['inds'].extend(ntot*ones)
        # exout['inde_lr'].extend(ntot*ones)
        exout['mask'].extend(np.zeros(extra_pair,dtype=np.int32))
    return exout


# Loading directly a batch of data as a list of pairs
def load_json_list_vmapped(data_dir, Rcut, species_list, forces, charges, dipoles, batch_size, one_shot, pair_mod, at_mod, pair_max, precompute_ewald_sums, ewald_inputs, pbc, Nats_max, long_range_pol, verbose, gaussian_charges, BORN, stress):
    all_examples = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == b'.example':
            all_examples.append(file)
    if len(all_examples) == 0:
        raise ValueError('No example file found.')
    # Species list decoded because it comes as b'' string
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    nex = len(all_examples)
    
    # gaussian charges or point charges?
    gaussian_charges = [g.decode('utf-8') for g in gaussian_charges]
    if gaussian_charges[0] == 'covalent':
        radii = [element(x).covalent_radius * 0.01 * np.sqrt(2.0) for x in species_list]
    elif gaussian_charges[0] == 'ionic':
        radii = [element(x).ionic_radii[0].ionic_radius * 0.01 * np.sqrt(2.0) for x in species_list]
    elif gaussian_charges[0] == 'False':
        radii = [0. for x in species_list]
    else:
        radii = [float(x) for x in gaussian_charges]
    radii = np.array(radii)



    for i in range(0,nex,batch_size):
        if verbose:
            print('Now processing batch {}'.format(i//batch_size))

        # Handle last bit: either make a small batch or skip
        if i+batch_size>nex:
            if one_shot:
                batch_size=nex-i
            else:
                break
        ntot = 0
        exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
                'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[],  'name':[],
                'indf':[], 'linsys_masks':[], 'gaussian_radii':[]}  # the last raw is added for LR}
        if not precompute_ewald_sums:
            exout['pos']  = []
            exout['latvecs'] = []
        if forces:
            exout['forces'] = []
        if charges:
            exout['charges'] = []
        if dipoles:
            exout['dipoles'] = []
        if long_range_pol:
            exout['indg'] = []
        if stress:
            exout['inds'] = []
            exout['stress'] = []
        
        Nats_list = np.zeros(batch_size, dtype=np.int32)   
        all_radiis, batch_pos, batch_latvecs = [], [], []
        for k in range(batch_size):
            example = ExampleJsonWrapper(os.path.join(data_dir,all_examples[i+k]),species_list)
            positions, replicas = make_replicas(example.angstrom_positions, \
                                      example.angstrom_lattice_vectors, Rcut)
                
            nats = len(example.angstrom_positions)
            exout['nats'].append(nats)
            posi = np.reshape(positions,(nats,1,1,3))
            posj = np.reshape(positions,(1,nats,1,3))
            rep = np.reshape(replicas,(1,1,-1,3))
            rij_vec = posj+rep-posi
            rij = np.sqrt(np.sum(rij_vec**2, axis=3))
            # Mask with the cutoff
            radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
            indall = np.where(radial_mask)
            inda = indall[0]
            indb = indall[1]
            exout['nn_r'].extend(rij[indall])
            exout['nn_vecs'].extend(rij_vec[indall])
            exout['species'].extend(example.species_indexes)
            exout['sp_a'].extend(example.species_indexes[inda])
            exout['sp_b'].extend(example.species_indexes[indb])
            # Indices get shifted with the atoms already in the batch
            exout['inda'].extend(inda+ntot)
            exout['indb'].extend(indb+ntot)
            exout['inde'].extend([k]*nats)
            exout['mask'].extend([1]*len(inda))
            ntot += nats
            all_radii = radii[example.species_indexes]
            exout['gaussian_radii'].append(np.pad(all_radii, (0,Nats_max+1-nats), constant_values=np.inf))
            
            exout['energy'].append(example.ev_energy)
            exout['name'].append(all_examples[i+k][:-8])
            Nats_list[k] = nats
            exout['indf'].extend([m for m in range(nats)]) # atom index within structure 
            
            batch_pos.append(np.pad(positions, ((0,Nats_max+1-nats),(0,0)), constant_values=np.inf))
            batch_latvecs.append(example.angstrom_lattice_vectors)
            
            try: # and see if there are forces in the dataset
                exout['forces'].extend(example.forces)
            except KeyError:
                pass
            try: # and see if there are charges in the dataset
                exout['charges'].extend(example.atomic_charges)
            except KeyError:
                pass 
            try: # and see if there are dipoles in the dataset
                exout['dipoles'].extend(example.atomic_dipoles)#np.pad(example.atomic_dipoles, ((0, Nats_max-nats),(0,0))))
            except KeyError:
                pass  
            try: # and see if there are dipoles in the dataset
                exout['stress'].append(example.stress)
            except KeyError:
                pass  
            if long_range_pol:
                # indexes for placing the E3 hardnesses
                exout['indg'].extend([m + Nats_max+1 for m in range(nats*3)]) # atom index within structure 
                # exout['inde_lr'].extend([k]*len(inda))
            if stress:
                exout['inds'].extend([k]*len(inda))
        exout['Efield_ext'] = []
        if not BORN == b'': # then add the dielectric info to the batch
            nats_BORN, species_BORN, pos_BORN, cell_BORN, eps_infty, born_charges = read_BORN_file(BORN)
            # exout['pos_BORN'] = pos_BORN[None]
            # exout['cell_BORN'] = cell_BORN[None]
            exout['eps_infty'] = eps_infty
            exout['born_charges'] = born_charges
        
            # I decided to use a different format (not .example) for the BORN file. 
            # I need to learn the mapping species_str -> species_idx (e.g. Ba Ti O -> 0 1 2)
            mapping = {}
            for s, i in zip(example.species_str, example.species_indexes):
                mapping.setdefault(s, i)
            indices_BORN = np.array([mapping[s] for s in species_BORN]) # Ba Ti O O O -> .example mapping (e.g. 0 1 2 2 2 )
            # exout['pos_BORN'] = np.pad(pos_BORN, ((0,Nats_max+1-nats_BORN),(0,0)), constant_values=np.inf)[None]
            # exout['cell_BORN'] = cell_BORN[None]
            born_radii = radii[indices_BORN]
            # exout['gaussian_radii_BORN'] = [np.pad(born_radii, (0,Nats_max+1-nats_BORN), constant_values=np.inf)]
            exout['Efield_ext'] = []
            
            
            batch_pos.append(np.pad(pos_BORN, ((0,Nats_max+1-nats_BORN),(0,0)), constant_values=np.inf))
            batch_latvecs.append(cell_BORN)
            exout['gaussian_radii'].append(np.pad(born_radii, (0,Nats_max+1-nats_BORN), constant_values=np.inf))
            Nats_list = np.concatenate((Nats_list,[nats_BORN]))
            exout['nats'].append(nats_BORN)
            exout['species'].extend(indices_BORN)
            exout['inde'].extend([k+1]*int(nats_BORN))
            exout['indf'].extend([m for m in range(nats_BORN)]) # atom index within structure 
            exout['indg'].extend([m + Nats_max+1 for m in range(nats_BORN*3)]) # atom index within structure 
            exout['forces'].extend(np.zeros((nats_BORN,3)))
            
            positions, replicas = make_replicas(pos_BORN, \
                                      cell_BORN, Rcut)
            posi = np.reshape(positions,(nats_BORN,1,1,3))
            posj = np.reshape(positions,(1,nats_BORN,1,3))
            rep = np.reshape(replicas,(1,1,-1,3))
            rij_vec = posj+rep-posi
            rij = np.sqrt(np.sum(rij_vec**2, axis=3))
            # Mask with the cutoff
            radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
            indall = np.where(radial_mask)
            inda = indall[0]
            indb = indall[1]
            exout['nn_r'].extend(rij[indall])
            exout['nn_vecs'].extend(rij_vec[indall])
            exout['sp_a'].extend(indices_BORN[inda])
            exout['sp_b'].extend(indices_BORN[indb])
            # Indices get shifted with the atoms already in the batch
            exout['inda'].extend(inda+ntot)
            exout['indb'].extend(indb+ntot)
            exout['mask'].extend([1]*len(inda))
            ntot += nats_BORN
            
            if stress:
                exout['inds'].extend([k+1]*len(inda))
            
            
            # linsys_masks = create_masks_pol([nats_BORN], Nats_max+1, n=4)
            # exout['linsys_masks_BORN'] = linsys_masks
          
            # 
            # # Ewald stuff
            # acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
            # if born_radii[0] == 0: # it's point charges
            #     ewald_params = set_ewald_parameters_for_batch(exout['pos_BORN'], exout['cell_BORN'], acc_factor, Rc=real_space_cutoff, gaussian_smearing=gaussian_smearing, Kc_sc=rec_space_cutoff, verbose=verbose)
            # else:
            #     ewald_params = set_ewald_parameters_gaussians_for_batch(exout['pos_BORN'], exout['cell_BORN'], exout['gaussian_radii_BORN'], acc_factor, Rc=real_space_cutoff, gaussian_smearing=gaussian_smearing, Kc_sc=rec_space_cutoff, verbose=verbose)
            # exout['ewald_input_rcuts'] = ewald_params[0]
            # exout['ewald_input_realspace'] = ewald_params[1]
            # exout['ewald_input_alphas'] = ewald_params[2]
            # exout['ewald_input_recspace'] = ewald_params[3]
                
        # at the end of iterating over the configurations...        
        if not long_range_pol: # then it's charge only
            linsys_masks = create_masks(Nats_list, Nats_max+1, n=1)
        else:  # it's the polarisable model
            linsys_masks = create_masks_pol(Nats_list, Nats_max+1, n=4)
        exout['linsys_masks'] = linsys_masks
        
        # precomputation of Ewald kernels DEPRECATED!
        # # precompute ewald sums or not 
        # if precompute_ewald_sums: 
        #     if verbose:
        #         print('Precomputing Ewald sums...')
        #     exout.update({'el_energy_kernels':[], 'el_force_kernels':[]})
        #     batch_pos = np.array(batch_pos)
        #     batch_latvecs = np.array(batch_latvecs)
        #     batch_radii = np.array(exout['gaussian_radii']) # this can be gaussian radii or zero if point charges
        #     if not long_range_pol: # then it's charges only
        #         e,f = compute_ewald_sums_charges(ewald_inputs, batch_pos, batch_latvecs, batch_radii, pbc, verbose=verbose)
        #     else: # it's the polarisable model
        #         e,f = compute_ewald_sums_pol(ewald_inputs, batch_pos, batch_latvecs, batch_radii, pbc, verbose=verbose)
                
        #     exout['el_energy_kernels'] = e
        #     exout['el_force_kernels'] = f
        #     # exout['linsys_masks'] = linsys_masks
        
        # ewald sums will be computed on the fly during training
            # if verbose:
            #     print('Ewald sums not precomputed')
    
        exout['pos'] = batch_pos
        exout['latvecs'] = batch_latvecs
        if pbc:
            acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
            # alpha = 1/(np.sqrt(2)*gaussian_smearing) if not gaussian_smearing == 0.0 else 0.0
            # WAS!!! ---
            if all_radii[0] == 0: # it's point charges
                ewald_params = set_ewald_parameters_for_batch(exout['nats'], exout['latvecs'], acc_factor, Rc=real_space_cutoff, gaussian_smearing=gaussian_smearing, Kc_sc=rec_space_cutoff, verbose=verbose)
            else:
                ewald_params = set_ewald_parameters_gaussians_for_batch(exout['nats'], exout['latvecs'], exout['gaussian_radii'], acc_factor, Rc=real_space_cutoff, gaussian_smearing=gaussian_smearing, Kc_sc=rec_space_cutoff, verbose=verbose)
            # end WAS! ---
            exout['ewald_input_Gcuts'] = ewald_params[0]
            exout['ewald_input_rcuts'] = ewald_params[1]
            exout['ewald_input_realspace'] = ewald_params[2]
            exout['ewald_input_alphas'] = ewald_params[3]
            exout['ewald_input_recspace'] = ewald_params[4]
                
            
        # This does not count the padding
        exout['ntot'] = ntot
        ptot = np.shape(exout['inda'])[0]
        bsize = batch_size if BORN == b'' else batch_size+1 
        # Padding
        exout = pad_batch(exout, at_mod, ntot, pair_mod, pair_max, ptot, nsp, bsize, ewald_inputs)
        
        yield exout



# import jax.numpy as jnp
def compute_ewald_sums_charges(ewald_inputs, all_positions, all_latvecs, all_radii, pbc, verbose=False):
    t0 = time.time()

    if pbc:
        # unpacking inputs for Ewald params setting
        acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
        
        if all_radii[0,0] == 0: # then the user wants point charges   
            print('Doing vmapped Ewald sum for point charges  on the batch')
            # --- Ewald summation ---              
            # inputs for Ewald sum
            input_realspace, alphas, input_recspace = set_ewald_parameters_for_batch(all_positions, all_latvecs, acc_factor, Rc=real_space_cutoff, gaussian_smearing=gaussian_smearing, Kc_sc=rec_space_cutoff, verbose=verbose)
            # jax.jit
            ew = (jax.vmap(partial(Ewald_summation, input_realspace=input_realspace, input_recspace=input_recspace)))
            out_ewald = ew(all_positions,
                all_latvecs,
                alphas,
                # input_realspace,
                # input_recspace,
                )
            e = out_ewald['el_energy_kernel']
            f = out_ewald['el_force_kernel']
            # b = out_ewald['linsys_bvector']
            # linsys_masks = out_ewald['linsys_masks']
            # --------------------
            t1 = time.time()
            print(t1-t0, ' sec Ewald sum')
        else: # we want gaussian charges
            print('Doing vmapped Ewald sum for Gaussian charges on the batch')
            input_realspace, alphas_screen, input_recspace = set_ewald_parameters_gaussians_for_batch(all_positions, all_latvecs, all_radii, acc_factor)
            # ew = (jax.vmap(partial(recewald_qq_jax_vect_gaussians, input_recspace=input_recspace )))
            ew = jax.vmap(partial(ewald_gaussians, input_realspace=input_realspace, input_recspace=input_recspace ))
            alphas = 1/(np.sqrt(2) * all_radii)
            out_ewald = ew(all_positions,
                      all_latvecs,
                      all_radii,
                      alphas_screen)
            e = out_ewald['el_energy_kernel']
            f = out_ewald['el_force_kernel']
            t1 = time.time()
            print(t1-t0, ' sec Ewald sum for Gaussians')
    else: # it's a finite system
        if all_radii[0,0] == 0: # then the user wants point charges   
            print('Computing electrostatics for a finite system with point charges')
            el = jax.vmap(Electrostatics_finite_systems)(all_positions)
            e = el['el_energy_kernel']
            f = el['el_force_kernel']
        else: # we want gaussian charges for a finite system
            print('Computing electrostatics for a finite system with Gaussian charges')
            alphas = 1/(np.sqrt(2) * all_radii)
            out = jax.vmap(Electrostatics_finite_systems_gaussians)(all_positions, alphas)
            e, f = out['el_energy_kernel'], out['el_force_kernel']
            
    # linsys_masks = create_masks(Nats_list, at_mod)
    return np.array(e),np.array(f)#, linsys_masks


def compute_ewald_sums_pol(ewald_inputs, all_positions, all_latvecs, all_radii, pbc, verbose=False):
    bsize, N, dofs = all_positions.shape
    t0 = time.time()
    
    if pbc:
        # unpacking inputs for Ewald params setting
        acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
        if all_radii[0,0] == 0: # then the user wants point charges   
            print('Doing vmapped Ewald sum pol on the batch')
            # --- Ewald summation ---

            rcuts, input_realspace, alphas, input_recspace = set_ewald_parameters_for_batch(all_positions, all_latvecs, acc_factor, Rc=real_space_cutoff, gaussian_smearing=gaussian_smearing, Kc_sc=rec_space_cutoff, verbose=verbose)
        
            # jax.jit
            ew = (jax.vmap(partial(ewald_pol, real_space_cutoff=rcuts, input_realspace=input_realspace, input_recspace=input_recspace)))
            out_ewald = ew(all_positions,
                all_latvecs,
                alphas,
                # input_realspace,
                # input_recspace,
                )
            e = out_ewald['el_energy_kernel']
            el_fqq_kernel = out_ewald['el_fqq_kernel']
            el_fqd_kernel = out_ewald['el_fqd_kernel']
            el_fdd_kernel = out_ewald['el_fdd_kernel']
    
            f = np.zeros((bsize, N*4, N*4, 3))
            f[:, 0:N, 0:N] = el_fqq_kernel
            f[:, 0:N,N:] = el_fqd_kernel
            f[:, N:,0:N] = np.swapaxes(el_fqd_kernel, 1,2)
            f[:, N:,N:] = el_fdd_kernel
    
            # --------------------
            t1 = time.time()
            print(t1-t0, ' sec Ewald sum')
        
        else: # the user wants Gaussian charges
            print('Doing vmapped Ewald sum Gaussian pol on the batch')
            # Getting Ewald params for the batch
            input_realspace, alphas_screen, input_recspace = set_ewald_parameters_gaussians_for_batch(all_positions, all_latvecs, all_radii, acc_factor)
            
            ew = jax.vmap(partial(ewald_gaussians_pol, input_realspace=input_realspace, input_recspace=input_recspace))
            out_ewald = ew(all_positions,
                           all_latvecs,
                           all_radii,
                           alphas_screen)
            e = out_ewald['el_energy_kernel']
            el_fqq_kernel = out_ewald['el_fqq_kernel']
            el_fqd_kernel = out_ewald['el_fqd_kernel']
            el_fdd_kernel = out_ewald['el_fdd_kernel']
            
            f = np.zeros((bsize, N*4, N*4, 3))
            f[:, 0:N, 0:N] = el_fqq_kernel
            f[:, 0:N,N:] = el_fqd_kernel
            f[:, N:,0:N] = np.swapaxes(el_fqd_kernel, 1,2)
            f[:, N:,N:] = el_fdd_kernel
            
        
    else: # it's a finite system
        if all_radii[0,0] == 0: # it's point charges   
            print('Computing electrostatics for a finite system')
            el = jax.vmap(Electrostatics_finite_systems)(all_positions)
            e = el['el_energy_kernel']
            f = el['el_force_kernel']
        else: # it's Gaussian charges
            raise NotImplementedError('Finite system Gaussian charges not implemented')
    
    
    
    # linsys_masks = create_masks(Nats_list, at_mod)

    return np.array(e), np.array(f)#, linsys_masks



# Loading directly a batch of data as a list of pairs
def load_all_json_list(data_dir, Rcut, species_list, get_forces, batch_size, one_shot, pair_mod, at_mod, ewald_inputs):
    all_examples = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == '.example':
            all_examples.append(file)
    if len(all_examples) == 0:
        raise ValueError('No example file found.')
    # Species list decoded because it comes as b'' string
    # species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    Nconf = len(all_examples)
    
    ntot = 0
    exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
            'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[], 'forces':[], 'name':[],
            'indf':[], 'charges':[], 'pos':[], 'latvecs':[] , 'el_energy_kernels':[], 'el_force_kernels':[], 'linsys_masks':[]} # the last raw is added for LR
    
    Nats_list = np.zeros(Nconf, dtype=np.int32)    
    for k in range(Nconf):
        print('Now processing {}th configuration'.format(k))


        example = ExampleJsonWrapper(os.path.join(data_dir,all_examples[k]),species_list)
        positions, replicas = make_replicas(example.angstrom_positions, \
                                  example.angstrom_lattice_vectors, Rcut)
            
        nats = len(example.angstrom_positions)
        exout['nats'].append(nats)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,-1,3))
        rij_vec = posj+rep-posi
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))
        # Mask with the cutoff
        radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
        indall = np.where(radial_mask)
        inda = indall[0]
        indb = indall[1]
        exout['nn_r'].extend(rij[indall])
        exout['nn_vecs'].extend(rij_vec[indall])
        exout['species'].extend(example.species_indexes)
        exout['sp_a'].extend(example.species_indexes[inda])
        exout['sp_b'].extend(example.species_indexes[indb])
        # Indices get shifted with the atoms already in the batch
        exout['inda'].extend(inda+ntot)
        exout['indb'].extend(indb+ntot)
        exout['inde'].extend([k]*nats)
        exout['mask'].extend([1]*len(inda))
        ntot += nats
        exout['energy'].append(example.ev_energy)
        exout['name'].append(all_examples[k][:-8])
        exout['forces'].extend(example.forces)
        

        # we pad now
        this_pos_padded = np.pad(positions, ( (0,at_mod-nats), (0,0) ), constant_values=np.inf)
        exout['pos'].append(this_pos_padded)
        exout['latvecs'].append(example.lattice_vectors)
        exout['indf'].extend([m for m in range(nats)]) # atom index within structure 
        exout['charges'].extend(example.atomic_charges)
        
        # all_positions[i] = this_pos_padded
        # all_latvecs[i] = example.angstrom_lattice_vectors
        Nats_list[k] = nats
        
    all_positions = np.array(exout['pos'], dtype=np.float32)
    all_latvecs = np.array(exout['latvecs'], dtype=np.float32)
    
    
    e,f,linsys_masks = compute_ewald_sums(ewald_inputs, all_positions, all_latvecs, Nats_list)
    

    exout['el_energy_kernels'].append(e)
    exout['el_force_kernels'].append(f)
    exout['linsys_masks'].append(linsys_masks)
  
    # This does not count the padding
    exout['ntot'] = ntot
    ptot = np.shape(exout['inda'])[0]
    # Padding
    exout = pad_batch(exout, at_mod, ntot, pair_mod, ptot, nsp, batch_size)
    
    return exout



# Loading directly a batch of data as a list of pairs
def load_json_list(data_dir, Rcut, species_list, get_forces, batch_size, one_shot, pair_mod, at_mod, stress):
    all_examples = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == b'.example':
            all_examples.append(file)
    if len(all_examples) == 0:
        raise ValueError('No example file found.')
    # Species list decoded because it comes as b'' string
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    nex = len(all_examples)
    for i in range(0,nex,batch_size):
        # print('Now processing {}th batch'.format(i))

        # Handle last bit: either make a small batch or skip
        if i+batch_size>nex:
            if one_shot:
                batch_size=nex-i
            else:
                break
        ntot = 0
        exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
                'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[], 'forces':[], 'name':[]} 
        if stress:
            exout['inds'] = []
            exout['stress'] = []
            exout['latvecs'] = []
            
        for k in range(batch_size):
            example = ExampleJsonWrapper(os.path.join(data_dir,all_examples[i+k]),species_list)
            positions, replicas = make_replicas(example.angstrom_positions, \
                                      example.angstrom_lattice_vectors, Rcut)
            nats = len(example.angstrom_positions)
            exout['nats'].append(nats)
            posi = np.reshape(positions,(nats,1,1,3))
            posj = np.reshape(positions,(1,nats,1,3))
            rep = np.reshape(replicas,(1,1,-1,3))
            rij_vec = posj+rep-posi
            rij = np.sqrt(np.sum(rij_vec**2, axis=3))
            # Mask with the cutoff
            radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
            indall = np.where(radial_mask)
            inda = indall[0]
            indb = indall[1]
            exout['nn_r'].extend(rij[indall])
            exout['nn_vecs'].extend(rij_vec[indall])
            exout['species'].extend(example.species_indexes)
            exout['sp_a'].extend(example.species_indexes[inda])
            exout['sp_b'].extend(example.species_indexes[indb])
            # Indices get shifted with the atoms already in the batch
            exout['inda'].extend(inda+ntot)
            exout['indb'].extend(indb+ntot)
            exout['inde'].extend([k]*nats)
            exout['mask'].extend([1]*len(inda))
            ntot += nats
            exout['energy'].append(example.ev_energy)
            exout['name'].append(all_examples[i+k][:-8])
            exout['forces'].extend(example.forces)
            if stress:
                exout['stress'].append(example.stress)
                exout['inds'].extend([k]*len(inda))
                exout['latvecs'].append(example.angstrom_lattice_vectors)
  
            # # added for LR
            # if len(ewald_inputs) > 1: # then we need to compute the Ewald sum

            #     # unpacking inputs for Ewald sum
            #     acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
                
            #     alpha = 1/(np.sqrt(2)*gaussian_smearing) if not np.isnan(gaussian_smearing) else np.nan
            #     # real_space_cutoff = real_space_cutoff if real_space_cutoff is not np.nan else None
            #     # rec_space_cutoff = rec_space_cutoff if rec_space_cutoff is not np.nan else None

                
            #     t0 = time.time()
            #     # --- Ewald summation ---
            #     linsys_masks = create_masks((nats,), at_mod)[0]
            #     input_realspace, alpha, input_recspace = set_ewald_parameters(positions, example.lattice_vectors, acc_factor, Rc=real_space_cutoff, alpha=alpha, Kc_sc=rec_space_cutoff, verbose=False)
                
            #     # extra_at = (at_mod-(ntot%at_mod))%at_mod
            #     # print(extra_at, 'EEEEE')
            #     positions_padded = np.pad(positions, ((0,at_mod-nats),(0,0)), constant_values=np.inf)
            #     # jax.jit
            #     ew = (partial(Ewald_summation, input_realspace=input_realspace, input_recspace=input_recspace))
            #     out_ewald = ew(positions_padded,
            #         example.lattice_vectors,
            #         alpha,
            #         # input_realspace,
            #         # input_recspace,
            #         )
            #     e = out_ewald['el_energy_kernel']
            #     f = out_ewald['el_force_kernel']
            #     # b = out_ewald['linsys_bvector']
            #     # linsys_masks = out_ewald['linsys_masks']
            #     # --------------------
            #     t1 = time.time()
            #     print(t1-t0, ' sec Ewald sum')
                
                
            #     # previous version
            #     # we pad now
            #     exout['pos'].append(np.pad(positions, ((0,at_mod-nats),(0,0)), constant_values=np.inf))
            #     exout['latvecs'].append(example.lattice_vectors)
            #     # ewald_params = set_ewald_parameters_for_batch(batch['pos'], batch['latvecs'], acc_factor, Rc=real_space_cutoff, alpha=alpha, Kc_sc=rec_space_cutoff, verbose=False, q=None)
            #     # parameters['ewald_params'] = ewald_params
                
                
                
                
            #     exout['indf'].extend([m for m in range(nats)]) # atom index within structure 
            #     exout['charges'].extend(example.atomic_charges)

            #     exout['el_energy_kernels'].append(e)
            #     exout['el_force_kernels'].append(f)
            #     exout['linsys_masks'].append(linsys_masks)
            #     # exout['linsys_bvectors'].append(b)
                

        # This does not count the padding
        exout['ntot'] = ntot
        ptot = np.shape(exout['inda'])[0]
        # Padding
        exout = pad_batch(exout, at_mod, ntot, pair_mod, 0, ptot, nsp, batch_size)
        
        yield exout

  



# Loading from spice hdf5 format
def load_spice_list(data_dir, Rcut, species_list, get_forces, batch_size, one_shot, pair_mod, at_mod):
    # Expecting data in Ha, bohr, with no PBC
    Ffact = -Ha2eV/bohr2A
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    # Creating conversion from atomic number to progressive index
    num2ord = {sym2num[s]:i for i,s in enumerate(species_list)}
    data = h5py.File(data_dir, 'r')
    molist = [(mo, i) for mo in data.keys() if len(data[mo])>0 \
                      for i in range(len(data[mo]['formation_energy']))]
    random.shuffle(molist)
    nex = len(molist)
    for i in range(0,nex,batch_size):
        # Handle last bit: either make a small batch or skip
        if i+batch_size>nex:
            if one_shot:
                batch_size=nex-i
            else:
                break
        ntot = 0
        exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
                'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[], 'forces':[], 'name':[]}
        for k in range(batch_size):
            mol, ind = molist[i+k]
            spnums = list(data[mol]['atomic_numbers'])
            nats = len(spnums)
            species = np.asarray([num2ord[n] for n in spnums])
            exout['nats'].append(nats)
            posi = np.reshape(data[mol]['conformations'][ind]*bohr2A,(nats,1,3))
            posj = np.reshape(data[mol]['conformations'][ind]*bohr2A,(1,nats,3))
            rij_vec = posj-posi
            rij = np.sqrt(np.sum(rij_vec**2, axis=2))
            # Mask with the cutoff
            radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
            indall = np.where(radial_mask)
            inda = indall[0]
            indb = indall[1]
            exout['nn_r'].extend(rij[indall])
            exout['nn_vecs'].extend(rij_vec[indall])
            exout['species'].extend(species)
            exout['sp_a'].extend(species[inda])
            exout['sp_b'].extend(species[indb])
            # Indices get shifted with the atoms already in the batch
            exout['inda'].extend(inda+ntot)
            exout['indb'].extend(indb+ntot)
            exout['inde'].extend([k]*nats)
            exout['mask'].extend([1]*len(inda))
            ntot += nats
            exout['energy'].append(data[mol]['formation_energy'][ind]*Ha2eV)
            exout['name'].append(mol+str(ind))
            exout['forces'].extend(data[mol]['dft_total_gradient'][ind] * Ffact)
        
        # This does not count the padding
        exout['ntot'] = ntot
        ptot = np.shape(exout['inda'])[0]
        # Padding
        exout = pad_batch(exout, at_mod, ntot, pair_mod, ptot, nsp, batch_size)
        yield exout

# Loading data in numpy format
def load_npy_list(data_dir, Rcut, species_list, get_forces, batch_size, one_shot, pair_mod, at_mod):
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    # Creating conversion from atomic number to progressive index
    num2ord = {sym2num[s]:i for i,s in enumerate(species_list)}
    all_files = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == b'.npy':
            all_files.append(file)    
    if len(all_files) == 0:
        raise ValueError('No data file found in {}.'.format(data_dir))
    # Shuffling files, loading one at a time
    random.shuffle(all_files)
    for f in all_files:
        data = np.load(data_dir+b'/'+f, allow_pickle=True)
        nex = len(data)-1
        indices = np.arange(1,nex+1)
        np.random.shuffle(indices)
        for i in range(0,nex,batch_size):
            this_batch_size = batch_size
            # Handle last bit: either make a small batch or skip
            if i+batch_size>nex:
                if one_shot:
                    this_batch_size=nex-i
                else:
                    break
            ntot = 0
            exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
                    'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[], 'forces':[], 'name':[]}
            for k in range(this_batch_size):
                conf = data[indices[i+k]]
                spnums = list(conf[2])
                nats = conf[1]
                species = np.asarray([num2ord[n] for n in spnums])
                exout['nats'].append(nats)
                positions, replicas = make_replicas(conf[5], conf[3], Rcut)
                posi = np.reshape(positions,(nats,1,1,3))
                posj = np.reshape(positions,(1,nats,1,3))
                rep = np.reshape(replicas,(1,1,-1,3))
                rij_vec = posj+rep-posi
                rij = np.sqrt(np.sum(rij_vec**2, axis=3))
                # Mask with the cutoff
                radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
                indall = np.where(radial_mask)
                inda = indall[0]
                indb = indall[1]
                exout['nn_r'].extend(rij[indall])
                exout['nn_vecs'].extend(rij_vec[indall])
                exout['species'].extend(species)
                exout['sp_a'].extend(species[inda])
                exout['sp_b'].extend(species[indb])
                # Indices get shifted with the atoms already in the batch
                exout['inda'].extend(inda+ntot)
                exout['indb'].extend(indb+ntot)
                exout['inde'].extend([k]*nats)
                exout['mask'].extend([1]*len(inda))
                ntot += nats
                exout['energy'].append(conf[6])
                name = conf[4]
                if len(conf)>8:
                    name = name+'_'+conf[8]
                exout['name'].append(name)
                exout['forces'].extend(conf[7])
            
            # This does not count the padding
            exout['ntot'] = ntot
            ptot = np.shape(exout['inda'])[0]
            # Padding
            exout = pad_batch(exout, at_mod, ntot, pair_mod, ptot, nsp, batch_size)
            yield exout
            
def read_BORN_file(filename):
    with open(filename, 'rb') as f:
        nats = np.genfromtxt(f, max_rows=1).astype(int)
        eps_infty = np.genfromtxt(f,  max_rows=1)
        born_charges = np.genfromtxt(f, max_rows=nats).ravel()#.reshape((nats,3,3))
        alat = np.genfromtxt(f, skip_header=1, max_rows=1)
        cell = np.genfromtxt(f, max_rows=3) * alat
        species = np.genfromtxt(f, max_rows=1, dtype=str)
        how_many = np.genfromtxt(f, max_rows=1)
        scaled_or_direct = np.genfromtxt(f, max_rows=1, dtype=str)
        scaled_pos = np.genfromtxt(f, max_rows=nats)
        pos = scaled_pos @ cell 
    return nats, species, pos, cell, eps_infty, born_charges          
          

# # Loading directly a batch of data as a list of pairs
# def load_json_list_ewald_sum_in_for_loop(data_dir, Rcut, species_list, get_forces, batch_size, one_shot, pair_mod, at_mod, ewald_inputs):
#     all_examples = []
#     for file in os.listdir(data_dir):
#         if os.path.splitext(file)[-1] == b'.example':
#             all_examples.append(file)
#     if len(all_examples) == 0:
#         raise ValueError('No example file found.')
#     # Species list decoded because it comes as b'' string
#     species_list = [s.decode('utf-8') for s in species_list]
#     nsp = len(species_list)
#     nex = len(all_examples)
#     for i in range(0,nex,batch_size):
#         print('Now processing {}th batch'.format(i))

#         # Handle last bit: either make a small batch or skip
#         if i+batch_size>nex:
#             if one_shot:
#                 batch_size=nex-i
#             else:
#                 break
#         ntot = 0
#         exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
#                 'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[], 'forces':[], 'name':[],
#                 'indf':[], 'charges':[], 'pos':[], 'latvecs':[] , 'el_energy_kernels':[], 'el_force_kernels':[], 'linsys_masks':[]} # the last raw is added for LR
#         for k in range(batch_size):
#             example = ExampleJsonWrapper(os.path.join(data_dir,all_examples[i+k]),species_list)
#             positions, replicas = make_replicas(example.angstrom_positions, \
#                                       example.angstrom_lattice_vectors, Rcut)
#             nats = len(example.angstrom_positions)
#             exout['nats'].append(nats)
#             posi = np.reshape(positions,(nats,1,1,3))
#             posj = np.reshape(positions,(1,nats,1,3))
#             rep = np.reshape(replicas,(1,1,-1,3))
#             rij_vec = posj+rep-posi
#             rij = np.sqrt(np.sum(rij_vec**2, axis=3))
#             # Mask with the cutoff
#             radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
#             indall = np.where(radial_mask)
#             inda = indall[0]
#             indb = indall[1]
#             exout['nn_r'].extend(rij[indall])
#             exout['nn_vecs'].extend(rij_vec[indall])
#             exout['species'].extend(example.species_indexes)
#             exout['sp_a'].extend(example.species_indexes[inda])
#             exout['sp_b'].extend(example.species_indexes[indb])
#             # Indices get shifted with the atoms already in the batch
#             exout['inda'].extend(inda+ntot)
#             exout['indb'].extend(indb+ntot)
#             exout['inde'].extend([k]*nats)
#             exout['mask'].extend([1]*len(inda))
#             ntot += nats
#             exout['energy'].append(example.ev_energy)
#             exout['name'].append(all_examples[i+k][:-8])
#             exout['forces'].extend(example.forces)
            
  
#             # added for LR
#             if len(ewald_inputs) > 1: # then we need to compute the Ewald sum

#                 # unpacking inputs for Ewald sum
#                 acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = ewald_inputs
                
#                 alpha = 1/(np.sqrt(2)*gaussian_smearing) if not np.isnan(gaussian_smearing) else np.nan
#                 # real_space_cutoff = real_space_cutoff if real_space_cutoff is not np.nan else None
#                 # rec_space_cutoff = rec_space_cutoff if rec_space_cutoff is not np.nan else None

                
#                 t0 = time.time()
#                 # --- Ewald summation ---
#                 linsys_masks = create_masks((nats,), at_mod)[0]
#                 input_realspace, alpha, input_recspace = set_ewald_parameters(positions, example.lattice_vectors, acc_factor, Rc=real_space_cutoff, alpha=alpha, Kc_sc=rec_space_cutoff, verbose=False)
                
#                 # extra_at = (at_mod-(ntot%at_mod))%at_mod
#                 # print(extra_at, 'EEEEE')
#                 positions_padded = np.pad(positions, ((0,at_mod-nats),(0,0)), constant_values=np.inf)
#                 # jax.jit
#                 ew = (partial(Ewald_summation, input_realspace=input_realspace, input_recspace=input_recspace))
#                 out_ewald = ew(positions_padded,
#                     example.lattice_vectors,
#                     alpha,
#                     # input_realspace,
#                     # input_recspace,
#                     )
#                 e = out_ewald['el_energy_kernel']
#                 f = out_ewald['el_force_kernel']
#                 # b = out_ewald['linsys_bvector']
#                 # linsys_masks = out_ewald['linsys_masks']
#                 # --------------------
#                 t1 = time.time()
#                 print(t1-t0, ' sec Ewald sum')
                
                
#                 # previous version
#                 # we pad now
#                 exout['pos'].append(np.pad(positions, ((0,at_mod-nats),(0,0)), constant_values=np.inf))
#                 exout['latvecs'].append(example.lattice_vectors)
#                 # ewald_params = set_ewald_parameters_for_batch(batch['pos'], batch['latvecs'], acc_factor, Rc=real_space_cutoff, alpha=alpha, Kc_sc=rec_space_cutoff, verbose=False, q=None)
#                 # parameters['ewald_params'] = ewald_params
                
                
                
                
#                 exout['indf'].extend([m for m in range(nats)]) # atom index within structure 
#                 exout['charges'].extend(example.atomic_charges)

#                 exout['el_energy_kernels'].append(e)
#                 exout['el_force_kernels'].append(f)
#                 exout['linsys_masks'].append(linsys_masks)
#                 # exout['linsys_bvectors'].append(b)
                

        
#         # This does not count the padding
#         exout['ntot'] = ntot
#         ptot = np.shape(exout['inda'])[0]
#         # Padding
#         exout = pad_batch(exout, at_mod, ntot, pair_mod, ptot, nsp, batch_size)
        
#         yield exout





# Can't remember what this does
# # Loading directly a batch of data as a list of pairs
# def load_json_list(data_dir, Rcut, species_list, get_forces, batch_size, one_shot, pair_mod, at_mod, precomputed_ewal_sums, ewald_params):
#     all_examples = []
#     for file in os.listdir(data_dir):
#         if os.path.splitext(file)[-1] == b'.example':
#             all_examples.append(file)
#     if len(all_examples) == 0:
#         raise ValueError('No example file found.')
#     # Species list decoded because it comes as b'' string
#     species_list = [s.decode('utf-8') for s in species_list]
#     nsp = len(species_list)
#     nex = len(all_examples)
#     for i in range(0,nex,batch_size):
#         print('Now processing {}th batch'.format(i))

#         # Handle last bit: either make a small batch or skip
#         if i+batch_size>nex:
#             if one_shot:
#                 batch_size=nex-i
#             else:
#                 break
#         ntot = 0
#         exout = {'nats':[], 'nn_r':[], 'nn_vecs':[], 'sp_a':[], 'sp_b':[], 'species':[],
#                 'inda':[], 'indb':[], 'inde':[], 'mask':[], 'energy':[], 'forces':[], 'name':[],
#                 'indf':[], 'charges':[], 'pos':[], 'latvecs':[] , 'el_energy_kernels':[], 'el_force_kernels':[], 'linsys_masks':[]} # the last raw is added for LR
#         for k in range(batch_size):
#             example = ExampleJsonWrapper(os.path.join(data_dir,all_examples[i+k]),species_list)
#             positions, replicas = make_replicas(example.angstrom_positions, \
#                                       example.angstrom_lattice_vectors, Rcut)
#             nats = len(example.angstrom_positions)
#             exout['nats'].append(nats)
#             posi = np.reshape(positions,(nats,1,1,3))
#             posj = np.reshape(positions,(1,nats,1,3))
#             rep = np.reshape(replicas,(1,1,-1,3))
#             rij_vec = posj+rep-posi
#             rij = np.sqrt(np.sum(rij_vec**2, axis=3))
#             # Mask with the cutoff
#             radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
#             indall = np.where(radial_mask)
#             inda = indall[0]
#             indb = indall[1]
#             exout['nn_r'].extend(rij[indall])
#             exout['nn_vecs'].extend(rij_vec[indall])
#             exout['species'].extend(example.species_indexes)
#             exout['sp_a'].extend(example.species_indexes[inda])
#             exout['sp_b'].extend(example.species_indexes[indb])
#             # Indices get shifted with the atoms already in the batch
#             exout['inda'].extend(inda+ntot)
#             exout['indb'].extend(indb+ntot)
#             exout['inde'].extend([k]*nats)
#             exout['mask'].extend([1]*len(inda))
#             ntot += nats
#             exout['energy'].append(example.ev_energy)
#             exout['name'].append(all_examples[i+k][:-8])
#             exout['forces'].extend(example.forces)
            
#             # else:
#             # previous version
#             # we pad now
#             exout['pos'].append(np.pad(positions, ((0,at_mod-nats),(0,0)), constant_values=np.inf))
#             exout['latvecs'].append(example.lattice_vectors)
            
#             # added for LR
#             exout['indf'].extend([m for m in range(nats)]) # atom index within structure 
#             exout['charges'].extend(example.atomic_charges)
        
        
#         # added for LR
#         if precomputed_ewal_sums: # then we just take them 
#             e,f,linsys_masks = ewald_params

#             exout['el_energy_kernels'].append(e)
#             exout['el_force_kernels'].append(f)
#             exout['linsys_masks'].append(linsys_masks)


#         # This does not count the padding
#         exout['ntot'] = ntot
#         ptot = np.shape(exout['inda'])[0]
#         # Padding
#         exout = pad_batch(exout, at_mod, ntot, pair_mod, ptot, nsp, batch_size)
        
#         yield exout
