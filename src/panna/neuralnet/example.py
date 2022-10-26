###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import tensorflow as tf

if __package__ != 'panna.neuralnet':
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from lib.example_bin import Example
else:
    from ..lib.example_bin import Example


def iterator_over_tfdata(g_size, *args, **kwargs):
    """ TFdata unpacker

    Args:
        g_size: size of the G's
        args: all the tfdata files that one wants to parse
    kwargs:
        derivatives: bool, whether forces should be calculated
        sparse_derivatives: bool, whether derivatives are stored in sparse form
    Return:
        iterator over the record in the files
    """
    for file in args:
        record_iterator = tf.python_io.tf_record_iterator(path=file)
        for record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(record)
            exact_energy = example.features.feature['energy'].float_list.value[
                0]
            species_vector = example.features.feature[
                'species'].int64_list.value
            g_vectors = np.reshape(
                example.features.feature['gvects'].float_list.value,
                (len(species_vector), g_size))
            if 'name' in example.features.feature:
                name = example.features.feature['name'].bytes_list.value[
                    0].decode()
            else:
                name = ''
            if 'key' in example.features.feature:
                key = example.features.feature['key'].bytes_list.value[
                    0].decode()
            else:
                key = ''
            if ('derivatives' in kwargs) and kwargs['derivatives']:
                if ('sparse_derivatives' in kwargs
                    ) and kwargs['sparse_derivatives']:
                    d_g_vector_values = example.features.feature[
                        'dgvect_values'].float_list.value
                    d_g_vector_indices1 = example.features.feature[
                        'dgvect_indices1'].float_list.value
                    d_g_vector_indices2 = example.features.feature[
                        'dgvect_indices2'].float_list.value
                    if not len(d_g_vector_values) or \
                       not len(d_g_vector_indices1) or \
                       not len(d_g_vector_indices2):
                        raise ValueError(
                            'Sparse derivatives requested but not '
                            'present in the tfr file')
                    d_g_vectors = []

                else:
                    d_g_vectors_flat = example.features.feature[
                        'dgvects'].float_list.value
                    if not len(d_g_vectors_flat):
                        raise ValueError('Derivatives requested but not '
                                         'present in the tfr file')
                    d_g_vectors = np.reshape(
                        d_g_vectors_flat,
                        (len(species_vector), g_size, len(species_vector) * 3))
                    d_g_vector_values = []
                    d_g_vector_indices1 = []
                    d_g_vector_indices2 = []

                if len(example.features.feature['forces'].float_list.value):
                    #there are reference forces
                    ref_forces = example.features.feature[
                        'forces'].float_list.value
                else:
                    ref_forces = []
            else:
                d_g_vectors = []
                d_g_vector_values = []
                d_g_vector_indices1 = []
                d_g_vector_indices2 = []
                ref_forces = []
                # we dont care if you have reference forces
                # if you are not predicting forces

            yield Example(g_vectors=g_vectors,
                          species_vector=species_vector,
                          true_energy=exact_energy,
                          d_g_available=('derivatives' in kwargs)
                          and kwargs['derivatives'],
                          d_g_vectors=d_g_vectors,
                          d_g_vector_values=d_g_vector_values,
                          d_g_vector_indices1=d_g_vector_indices1,
                          d_g_vector_indices2=d_g_vector_indices2,
                          name=name,
                          forces=ref_forces)
