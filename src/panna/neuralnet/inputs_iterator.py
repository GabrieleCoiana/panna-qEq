###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""Utilities to handling the input system
"""
import os
from typing import Callable
import tensorflow as tf
import numpy as np

from panna.neuralnet.parse_fn import load_json
from panna.neuralnet.parse_fn import load_json_list, load_json_list_vmapped, load_spice_list, load_npy_list, load_all_json_list

tf.autograph.set_verbosity(5)

def input_pipeline(data_dir: str,
                   batch_size: int,
                   parse_fn: Callable,
                   stress: bool,
                   name: str = 'input_pipeline',
                   prefetch_buffer_size_multiplier: int = 1,
                   num_parallel_readers: int = 4,
                   num_parallel_calls: int = 4,
                   cache: bool = False,
                   oneshot: bool = False,
                   shuffle: bool = True,
                   input_format: str = 'tfr',
                   extra_data: dict = {},
                   pbc = True,
                   **kwargs):
    """Construct input iterator.

    Parameters
    ----------

    data_dir: directory for data, must contain a "train_tf" subfolder
    batch_size: integer
    parse_fn: function to parse the data from tfrecord file
    name: name scope

    *_buffer_size_multiplier:
      batchsize times this number

    num_parallel_readers:
      process that are doing Input form devices, reading is not deterministic,
      if you need determinism change to the code-base must be applied

    num_parallel_calls:
      call of the parse function

    cache: bool
      if the data can be memorized in memory this will avoid rereading them each
      time.  *huge performance boost*

    oneshot:
      experimental, do not set

    Returns
    -------
        initializable_iterator, recover input data to feed the model

    Note
    ----
        * shuffling batch and buffer size multiplier default are
          randomly chosen by me

        * initializable iterator can be changed to one shot iterator
          in future version to better comply with documentation

        * a maximum number of epoch should also be added to this routine.
    """
    # We distinguish between models that require a padded list of atoms with neighbours
    # and those that require a list of pairs
    if ('nn_mode' in extra_data) and extra_data['nn_mode']=='list':
        mode = 'list'
    else:
        mode = 'pad'
    # Dictionaries needed for example parsing
    if input_format=='example' or input_format=='spice' or input_format=='npy':
        if mode == 'pad':
            output_signature = {'nats': tf.TensorSpec(shape=(), dtype=tf.int32),
                'species': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'positions': tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                'nn_inds': tf.TensorSpec(shape=(None,None,None), dtype=tf.int32),
                'nn_num': tf.TensorSpec(shape=(None,None,2), dtype=tf.int32),
                'nn_vecs': tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32),
                'nn_r': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                'mask1': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                'mask2': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                'energy': tf.TensorSpec(shape=(), dtype=tf.float32),
                'name': tf.TensorSpec(shape=(), dtype=tf.string)}
            padded_shapes = {'nats': [],
                'species': [None],
                'positions': [None, 3],
                'nn_inds': [None, None, None],
                'nn_num': [None, None, 2],
                'nn_vecs': [None, None, None, 3],
                'nn_r': [None, None, None],
                'mask1': [None, None, None],
                'mask2': [None, None, None],
                'energy': [],
                'name': []}
            padding_values={'nats': 0,
                'species': len(extra_data['species']),
                'positions': 0.0,
                'nn_inds': 0,
                'nn_num': 0,
                'nn_vecs': 0.0,
                'nn_r': 0.0,
                'mask1': 0.0,
                'mask2': 0.0,
                'energy': 0.0,
                'name': None}
            if parse_fn._forces:
                output_signature['forces'] = tf.TensorSpec(shape=(None,3), dtype=tf.float32)
                padded_shapes['forces'] = [None,3]
                padding_values['forces'] = 0.0
            if stress:
                output_signature['forces'] = tf.TensorSpec(shape=(None,3,3), dtype=tf.float32)
        elif mode == 'list':
            output_signature = {'nats': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'ntot': tf.TensorSpec(shape=(), dtype=tf.int32),
                'inda': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'indb': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'inde': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'nn_vecs': tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                'nn_r': tf.TensorSpec(shape=(None), dtype=tf.float32),
                'species': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'sp_a': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'sp_b': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'mask': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'energy': tf.TensorSpec(shape=(None), dtype=tf.float32),
                'name': tf.TensorSpec(shape=(None), dtype=tf.string),
                'forces': tf.TensorSpec(shape=(None,3), dtype=tf.float32) }
            if stress:
                output_signature['stress'] = tf.TensorSpec(shape=(None,3,3), dtype=tf.float32)
                output_signature['inds'] = tf.TensorSpec(shape=(None), dtype=tf.int32)
                output_signature['latvecs'] = tf.TensorSpec(shape=(None,3,3), dtype=tf.float32)



    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    # interleave section is very useful
    with tf.name_scope(name):
        # create a dataset of files
        if input_format=='tfr':
            data = tf.data.Dataset.list_files(os.path.join(data_dir, "*.tfrecord"))
            # in parallel read the files.
            data = data.interleave(tf.data.TFRecordDataset,
                                   cycle_length=num_parallel_readers,
                                   deterministic=None)
        elif input_format=='example':
            if mode == 'pad':
                data = tf.data.Dataset.from_generator(load_json, \
                    args=[data_dir,extra_data['mincut'],extra_data['maxcut'],
                          extra_data['species'],parse_fn._forces], \
                    output_signature=output_signature)
            elif mode == 'list':
                data = tf.data.Dataset.from_generator(load_json_list, \
                    args=[data_dir,extra_data['Rcut'],
                          extra_data['species'],parse_fn._forces,
                          batch_size,oneshot,extra_data['pair_mod'],
                          extra_data['at_mod'],
                          stress], \
                    output_signature=output_signature) 

        elif input_format=='spice':
            if mode == 'list':
                data = tf.data.Dataset.from_generator(load_spice_list, \
                    args=[data_dir,extra_data['Rcut'],
                          extra_data['species'],parse_fn._forces,
                          batch_size,oneshot,extra_data['pair_mod'],
                          extra_data['at_mod']], \
                    output_signature=output_signature)
        elif input_format=='npy':
            if mode == 'list':
                data = tf.data.Dataset.from_generator(load_npy_list, \
                    args=[data_dir,extra_data['Rcut'],
                          extra_data['species'],parse_fn._forces,
                          batch_size,oneshot,extra_data['pair_mod'],
                          extra_data['at_mod']], \
                    output_signature=output_signature)

        if cache:
            # cache the data in memory avoiding reading them again from storage
            # devices
            data = data.cache()

        # this will simply repeat the dataset.
        # repeat is done before shuffling,
        # this solves the problem of not complete batches but
        # blurries the idea of epoch because they are mixed together.
        # for a good explanation:
        # https://www.tensorflow.org/guide/data#processing_multiple_epochs
        if not oneshot:
            data = data.repeat()

        if shuffle:
            shuffle_buffer_size_multiplier = kwargs.get(
                'shuffle_buffer_size_multiplier', 10)
            # shuffle_buffer_size_multiplier = 1000
            # perform the shuffling, this can be performend only when
            # each iteration we will go through a different shuffling
            # eg from TF doc:
            # dataset = tf.data.Dataset.range(3)
            # dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            # dataset = dataset.repeat(2)
            # [1, 0, 2, 1, 2, 0]
            if mode == 'pad':
                data = data.shuffle(buffer_size=shuffle_buffer_size_multiplier,
                                    reshuffle_each_iteration=True)
            elif mode == 'list':
                # print('AAAAAAAAA', kwargs.get('shuffle_buffer_size_multiplier', 10))
                # data = data.batch(batch_size=2)
                data = data.shuffle(buffer_size=batch_size * shuffle_buffer_size_multiplier,
                                    reshuffle_each_iteration=True)
                

        if input_format=='tfr':
            # perform batching
            data = data.batch(batch_size=batch_size)
            # Unpack the data and perform the processing
            data = data.map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
        elif input_format=='example' and mode == 'pad':
            # batching and padding the example dicts if we didn't load a list
            data = data.padded_batch(batch_size=batch_size,
                        padded_shapes=padded_shapes,
                        padding_values=padding_values)

        # prefetch a number of batches
        data = data.prefetch(buffer_size=prefetch_buffer_size_multiplier)
    # print('fffffff')
    return data





def input_pipeline_lr(data_dir: str,
                   batch_size: int,
                   # parse_fn: Callable,
                   forces : bool,
                   charges : bool,
                   dipoles : bool,
                   stress : bool,
                   name: str = 'input_pipeline',
                   prefetch_buffer_size_multiplier: int = 1,
                   num_parallel_readers: int = 4,
                   num_parallel_calls: int = 4,
                   cache: bool = False,
                   oneshot: bool = False,
                   shuffle: bool = True,
                   input_format: str = 'tfr',
                   long_range_pol: bool = False,
                   gaussian_charges: bool = False,
                   precompute_ewald_sums: bool = False,
                   extra_data: dict = {},
                   pbc = True,
                   **kwargs):
    """Construct input iterator.

    Parameters
    ----------

    data_dir: directory for data, must contain a "train_tf" subfolder
    batch_size: integer
    parse_fn: function to parse the data from tfrecord file
    name: name scope

    *_buffer_size_multiplier:
      batchsize times this number

    num_parallel_readers:
      process that are doing Input form devices, reading is not deterministic,
      if you need determinism change to the code-base must be applied

    num_parallel_calls:
      call of the parse function

    cache: bool
      if the data can be memorized in memory this will avoid rereading them each
      time.  *huge performance boost*

    oneshot:
      experimental, do not set

    Returns
    -------
        initializable_iterator, recover input data to feed the model

    Note
    ----
        * shuffling batch and buffer size multiplier default are
          randomly chosen by me

        * initializable iterator can be changed to one shot iterator
          in future version to better comply with documentation

        * a maximum number of epoch should also be added to this routine.
    """
    # We distinguish between models that require a padded list of atoms with neighbours
    # and those that require a list of pairs
    if ('nn_mode' in extra_data) and extra_data['nn_mode']=='list':
        mode = 'list'
    else:
        mode = 'pad'
    # Dictionaries needed for example parsing
    if input_format=='example' or input_format=='spice' or input_format=='npy':
        if mode == 'pad':
            output_signature = {'nats': tf.TensorSpec(shape=(), dtype=tf.int32),
                'species': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'positions': tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                'nn_inds': tf.TensorSpec(shape=(None,None,None), dtype=tf.int32),
                'nn_num': tf.TensorSpec(shape=(None,None,2), dtype=tf.int32),
                'nn_vecs': tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32),
                'nn_r': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                'mask1': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                'mask2': tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                'energy': tf.TensorSpec(shape=(), dtype=tf.float32),
                'name': tf.TensorSpec(shape=(), dtype=tf.string)}
            padded_shapes = {'nats': [],
                'species': [None],
                'positions': [None, 3],
                'nn_inds': [None, None, None],
                'nn_num': [None, None, 2],
                'nn_vecs': [None, None, None, 3],
                'nn_r': [None, None, None],
                'mask1': [None, None, None],
                'mask2': [None, None, None],
                'energy': [],
                'name': []}
            padding_values={'nats': 0,
                'species': len(extra_data['species']),
                'positions': 0.0,
                'nn_inds': 0,
                'nn_num': 0,
                'nn_vecs': 0.0,
                'nn_r': 0.0,
                'mask1': 0.0,
                'mask2': 0.0,
                'energy': 0.0,
                'name': None}
            if forces:#parse_fn._forces:
                output_signature['forces'] = tf.TensorSpec(shape=(None,3), dtype=tf.float32)
                padded_shapes['forces'] = [None,3]
                padding_values['forces'] = 0.0
            if charges:
                output_signature['charges'] = tf.TensorSpec(shape=(None), dtype=tf.float32)
            if dipoles:
                output_signature['dipoles'] = tf.TensorSpec(shape=(None,3), dtype=tf.float32)
        elif mode == 'list':
            # common output signature
            output_signature = {'nats': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'ntot': tf.TensorSpec(shape=(), dtype=tf.int32),
                'inda': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'indb': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'inde': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'nn_vecs': tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                'nn_r': tf.TensorSpec(shape=(None), dtype=tf.float32),
                'species': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'sp_a': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'sp_b': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'mask': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'energy': tf.TensorSpec(shape=(None), dtype=tf.float32),
                'name': tf.TensorSpec(shape=(None), dtype=tf.string),
                'indf': tf.TensorSpec(shape=(None), dtype=tf.int32),
                'linsys_masks':tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                'gaussian_radii':tf.TensorSpec(shape=(None,None), dtype=tf.float32)}                
            if forces:#parse_fn._forces:
                output_signature['forces'] = tf.TensorSpec(shape=(None,3), dtype=tf.float32)
            if stress:#parse_fn._forces:
                output_signature['stress'] = tf.TensorSpec(shape=(None,3,3), dtype=tf.float32)
                output_signature['inds'] = tf.TensorSpec(shape=(None), dtype=tf.int32)
            if charges:
                output_signature['charges'] = tf.TensorSpec(shape=(None), dtype=tf.float32)
            if dipoles:
                output_signature['dipoles'] = tf.TensorSpec(shape=(None,3), dtype=tf.float32)
            if long_range_pol:
                output_signature['indg'] = tf.TensorSpec(shape=(None), dtype=tf.int32)
                
            # TODO: eliminate need for Efield_ext!
            output_signature['Efield_ext'] = tf.TensorSpec(shape=(None), dtype=tf.float32) 
            if extra_data['BORN'] != '':
                # output_signature['pos_BORN'] = tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32)
                # output_signature['cell_BORN'] = tf.TensorSpec(shape=(1, 3, 3), dtype=tf.float32)
                # output_signature['gaussian_radii_BORN'] = tf.TensorSpec(shape=(None), dtype=tf.float32)
                output_signature['eps_infty'] = tf.TensorSpec(shape=(None), dtype=tf.float32)
                output_signature['born_charges'] = tf.TensorSpec(shape=(None), dtype=tf.float32)
                output_signature['Efield_ext'] = tf.TensorSpec(shape=(None), dtype=tf.float32)
                # output_signature['linsys_masks_BORN'] = tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)
                                           
            if precompute_ewald_sums:
                output_signature.update(
                {'el_energy_kernels':tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                 'el_force_kernels': tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32)})
            else: # you compute things on the fly
                # in any case you need positons and lattice vectors
                output_signature.update({'pos':     tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                                         'latvecs': tf.TensorSpec(shape=(None, 3,3), dtype=tf.float32)})
                if pbc: # you do ewald sum
                    output_signature.update({'ewald_input_Gcuts':   tf.TensorSpec(shape=(None), dtype=tf.float32),
                                             'ewald_input_rcuts':   tf.TensorSpec(shape=(None), dtype=tf.float32),
                                             'ewald_input_realspace':tf.TensorSpec(shape=(None), dtype=tf.float32),
                                             'ewald_input_alphas':   tf.TensorSpec(shape=(None), dtype=tf.float32),
                                             'ewald_input_recspace': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)})
                    # # delete these two when you're finished with aaa_LR.py
                    # output_signature.update({'ewald_input_realspace_pc':tf.TensorSpec(shape=(None), dtype=tf.float32),
                    #                          'ewald_input_alphas_pc':   tf.TensorSpec(shape=(None), dtype=tf.float32),
                    #                          'ewald_input_recspace_pc': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)})
                    # output_signature.update({'ewald_input_realspace_gc':tf.TensorSpec(shape=(None), dtype=tf.float32),
                    #                          'ewald_input_alphas_gc':   tf.TensorSpec(shape=(None), dtype=tf.float32),
                    #                          'ewald_input_recspace_gc': tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)})
                
    # https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    # interleave section is very useful
    with tf.name_scope(name):
        # create a dataset of files
        if input_format=='tfr':
            data = tf.data.Dataset.list_files(os.path.join(data_dir, "*.tfrecord"))
            # in parallel read the files.
            data = data.interleave(tf.data.TFRecordDataset,
                                   cycle_length=num_parallel_readers,
                                   deterministic=None)
        elif input_format=='example':
            if mode == 'pad': # NOT IMPLEMENTED
                data = tf.data.Dataset.from_generator(load_json, \
                    args=[data_dir,extra_data['mincut'],extra_data['maxcut'],
                          extra_data['species'],forces],#parse_fn._forces], \
                    output_signature=output_signature)
            elif mode == 'list':
                # if extra_data['precomputed_ewald_sums'] == False: # old version
                data = tf.data.Dataset.from_generator(load_json_list_vmapped, \
                    args=[data_dir,extra_data['Rcut'],
                          extra_data['species'],forces,charges,dipoles,#parse_fn._forces,
                          batch_size,oneshot,
                          extra_data['pair_mod'],
                          extra_data['at_mod'], 
                          extra_data['pair_max'],
                          precompute_ewald_sums,
                          extra_data['ewald_input_params'], pbc, extra_data['Nats_max'],
                          long_range_pol, extra_data['verbose'], extra_data['gaussian_charges'],
                          extra_data['BORN'], 
                          stress], \
                    output_signature=output_signature) #extra_data['ewald_inputs']

                # else:
                #     print('AAAA')
                #     examples_lists = load_all_json_list(data_dir, extra_data['Rcut'], extra_data['species'], parse_fn._forces, batch_size, oneshot, extra_data['pair_mod'], extra_data['at_mod'], extra_data['ewald_input_params'])
                #     return examples_lists
                #     print('BBB')
                #     data = tf.data.Dataset.from_tensor_slices(examples_lists)
                #     print('CCC')
                    
        elif input_format=='spice':
            if mode == 'list':
                data = tf.data.Dataset.from_generator(load_spice_list, \
                    args=[data_dir,extra_data['Rcut'],
                          extra_data['species'],parse_fn._forces,
                          batch_size,oneshot,extra_data['pair_mod'],
                          extra_data['at_mod']], \
                    output_signature=output_signature)
        elif input_format=='npy':
            if mode == 'list':
                data = tf.data.Dataset.from_generator(load_npy_list, \
                    args=[data_dir,extra_data['Rcut'],
                          extra_data['species'],parse_fn._forces,
                          batch_size,oneshot,extra_data['pair_mod'],
                          extra_data['at_mod']], \
                    output_signature=output_signature)

        if cache:
            # cache the data in memory avoiding reading them again from storage
            # devices
            data = data.cache()

        # this will simply repeat the dataset.
        # repeat is done before shuffling,
        # this solves the problem of not complete batches but
        # blurries the idea of epoch because they are mixed together.
        # for a good explanation:
        # https://www.tensorflow.org/guide/data#processing_multiple_epochs
        if not oneshot:
            data = data.repeat()

        if shuffle:
            shuffle_buffer_size_multiplier = kwargs.get(
                'shuffle_buffer_size_multiplier', 10)
            # shuffle_buffer_size_multiplier = 1000
            # perform the shuffling, this can be performend only when
            # each iteration we will go through a different shuffling
            # eg from TF doc:
            # dataset = tf.data.Dataset.range(3)
            # dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            # dataset = dataset.repeat(2)
            # [1, 0, 2, 1, 2, 0]
            if mode == 'pad':
                data = data.shuffle(buffer_size=shuffle_buffer_size_multiplier,
                                    reshuffle_each_iteration=True)
            elif mode == 'list':
                # print('AAAAAAAAA', kwargs.get('shuffle_buffer_size_multiplier', 10))
                # data = data.batch(batch_size=2)
                data = data.shuffle(buffer_size=batch_size * shuffle_buffer_size_multiplier,
                                    reshuffle_each_iteration=True)
                

        if input_format=='tfr':
            # perform batching
            data = data.batch(batch_size=batch_size)
            # Unpack the data and perform the processing
            data = data.map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
        elif input_format=='example' and mode == 'pad':
            # batching and padding the example dicts if we didn't load a list
            data = data.padded_batch(batch_size=batch_size,
                        padded_shapes=padded_shapes,
                        padding_values=padding_values)

        # prefetch a number of batches
        data = data.prefetch(buffer_size=prefetch_buffer_size_multiplier)
    # print('fffffff')
    return data
