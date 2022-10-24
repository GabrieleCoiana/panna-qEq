###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os

import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def tfr_writer(filename, data, path='.'):
    """ Tfrecord writer

    Parameters
    ----------
    filename: string
    data: iterable of Tensorflow examples
    path: string
    """
    with tf.io.TFRecordWriter(
            os.path.join(path, '{}.tfrecord'.format(filename)))\
            as record_writer:
        for entry in data:
            record_writer.write(entry.SerializeToString())


def example_tf_packer(example,
                      forces=False,
                      sparse_dgvect=False,
                      per_atom_quantity=False):
    """Create an example with only the g-vectors without padding

    Parameters
    ----------
    example: instance of Example
    forces: Boolean, optional
        flag to add derivative and forces
        default False
    sparse_dgvect: Boolean, optional
        flag to save sparse derivatives
        default False
    per_atom_quantity: Boolean
        flag to store a per atom quantity
        default False

    Returns
    -------
       tf.train.Example with the following keys:
         + gvects: (number of atoms * g-vector size) vector of float
         + species: (number of atoms) vector of integer
                    in range 0, number of species - 1
         + energy: (1) vector of float
         --- if forces
         + dgvects : (number of atoms * g-vecotr size * 3 * number of atom )
         + forces : (3 * number of atoms )
         --- if per atom quantity
         + per atom quantity (number_of_atom)
    """
    feature = {
        'gvects': _floats_feature(example.gvects.flatten()),
        'species': _int64_feature(example.species_vector),
        'energy': _floats_feature([example.true_energy]),
        'name': _bytes_feature([example.name.encode()])
    }

    if forces:
        if not sparse_dgvect:
            feature['dgvects'] = _floats_feature(example.dgvects.flatten())
        else:
            feature['dgvect_size'] = _int64_feature(
                [len(example.dgvect_values.flatten())])
            feature['dgvect_values'] = _floats_feature(
                example.dgvect_values.flatten())
            feature['dgvect_indices1'] = _floats_feature(
                example.dgvect_indices1.flatten())
            feature['dgvect_indices2'] = _floats_feature(
                example.dgvect_indices2.flatten())
        feature['forces'] = _floats_feature(example.forces.flatten())

    if per_atom_quantity:
        feature['per_atom_quantity'] = _floats_feature(
            example.per_atom_quantity)

    return tf.train.Example(features=tf.train.Features(feature=feature))
