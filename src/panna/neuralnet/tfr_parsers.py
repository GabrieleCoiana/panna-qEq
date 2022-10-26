"""Utilities to handling the input system
"""

import tensorflow as tf


def parse_fn(example,
             g_size,
             zeros,
             n_species,
             *,
             forces=False,
             per_atom_quantity=False,
             sparse_dgvect=False,
             energy_rescale=1.0):
    """Parse TFExample records and perform simple data augmentation.

    Parameters
    ----------
        example: a batch of example obj
        g_size: size of the g_vector
        zeros: array of zero's one value per specie.
        n_species: number of species
        forces: recover forces and dg/dx
        per_atom_quantity: recover a per_atom_quantity
        energy_rescale: scale the energy

    Return
    ------
        species_tensor: Sparse Tensor, (n_atoms) value in range(n_species)
        g_vectors_tensor: Sparse Tensor, (n_atoms, g_size)
        energy: true energy value corrected with the zeros
    """
    # species is a vector of length ?number of atoms? padded to a biggest number
    # with n_species as value
    feat = {
        "energy":
        tf.FixedLenFeature([], dtype=tf.float32),
        "species":
        tf.FixedLenSequenceFeature([],
                                   dtype=tf.int64,
                                   allow_missing=True,
                                   default_value=n_species),
        "gvects":
        tf.FixedLenSequenceFeature([g_size],
                                   dtype=tf.float32,
                                   allow_missing=True)
    }
    if forces:
        if not sparse_dgvect:
            feat["dgvects"] = tf.FixedLenSequenceFeature([],
                                                         dtype=tf.float32,
                                                         allow_missing=True)
        else:
            feat["dgvect_size"] = tf.FixedLenFeature([], dtype=tf.int64)
            feat["dgvect_values"] = tf.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True)
            feat["dgvect_indices1"] = tf.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True)
            feat["dgvect_indices2"] = tf.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True)
        feat["forces"] = tf.FixedLenSequenceFeature([],
                                                    dtype=tf.float32,
                                                    allow_missing=True)
    if per_atom_quantity:
        feat['per_atom_quantity'] = tf.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True, default_value=0.0)

    parsed = tf.parse_example(example, features=feat)
    # remove the zero bias
    biases = tf.gather(tf.concat([zeros, [0.0]], axis=0), parsed['species'])
    energy = parsed['energy'] - tf.reduce_sum(biases, axis=1)
    parsed['energy'] = energy * energy_rescale

    return parsed
