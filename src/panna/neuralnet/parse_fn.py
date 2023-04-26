"""Utilities to handling the input system
"""

import tensorflow as tf
import tensorflow.io as tfio
from panna.gvector.pbc import make_replicas
from panna.lib import ExampleJsonWrapper
import numpy as np
import os

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
