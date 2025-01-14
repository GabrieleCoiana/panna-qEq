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

from functools import partial
from panna.neuralnet.parse_fn import load_json, load_spice, load_npy, load_xyz

# Function go from RaggedTensor batch to concatenated tensors padded to the nearest multiple
def pad_batch(at_mod, pair_mod, nsp, batch_size, x):
    ntot = tf.reduce_sum(x['nats'])
    x['ntot'] = ntot
    # Hack to build indices
    x['inde'] = tf.map_fn(lambda x:x['n']*tf.ones(x['l'],dtype=tf.int32), {'l':x['nats'],'n':tf.range(batch_size)},
                          fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32)).values
    x['inda'] = tf.map_fn(lambda x:x['i']+x['s'], {'i':x['inda'],'s':tf.cast(x['species'].row_splits[:-1], tf.int32)},
                          fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32)).values
    x['indb'] = tf.map_fn(lambda x:x['i']+x['s'], {'i':x['indb'],'s':tf.cast(x['species'].row_splits[:-1], tf.int32)},
                          fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32)).values
    # Flattened versions of the ragged
    for k in ['nn_vecs', 'nn_r', 'species', 'sp_a', 'sp_b', 'forces', 'targets']:
        if k in x:
            x[k] = x[k].values
    ptot = tf.shape(x['inda'])[0]
    # Padding
    extra_at = (at_mod-(ntot%at_mod))%at_mod
    # if extra_at>0:
    x['species'] = tf.concat([x['species'], nsp*tf.ones(extra_at,dtype=tf.int32)], axis=0)
    x['inde'] = tf.concat([x['inde'], batch_size*tf.ones(extra_at,dtype=tf.int32)], axis=0)
    if 'forces' in x:
        x['forces'] = tf.concat([x['forces'], tf.zeros((extra_at,3),dtype=tf.float32)], axis=0)
    if 'targets' in x:
        x['targets'] = tf.concat([x['targets'], tf.zeros(extra_at,dtype=tf.int32)], axis=0)
    extra_pair = (pair_mod-(ptot%pair_mod))%pair_mod
    x['mask'] = tf.ones(ptot, dtype=tf.int32)
    # if extra_pair>0:
    ones = tf.ones(extra_pair, dtype=tf.int32)
    x['nn_r'] = tf.concat([x['nn_r'], tf.ones(extra_pair,dtype=tf.float32)], axis=0)
    x['nn_vecs'] = tf.concat([x['nn_vecs'], tf.ones((extra_pair,3),dtype=tf.float32)], axis=0)
    x['sp_a'] = tf.concat([x['sp_a'], nsp*ones], axis=0)
    x['sp_b'] = tf.concat([x['sp_b'], nsp*ones], axis=0)
    x['inda'] = tf.concat([x['inda'], ntot*ones], axis=0)
    x['indb'] = tf.concat([x['indb'], ntot*ones], axis=0)
    x['mask'] = tf.concat([x['mask'], tf.zeros(extra_pair,dtype=tf.int32)], axis=0)
    return x

def input_pipeline(data_dir: str,
                   batch_size: int,
                   parse_fn: Callable,
                   name: str = 'input_pipeline',
                   prefetch_buffer_size_multiplier: int = 1,
                   num_parallel_readers: int = 4,
                   num_parallel_calls: int = 4,
                   cache: bool = False,
                   oneshot: bool = False,
                   shuffle: bool = True,
                   input_format: str = 'tfr',
                   extra_data: dict = {},
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
    # Dictionaries needed for example parsing
    if input_format in ['example', 'spice', 'npy', 'xyz']:
        output_signature = {'nats': tf.TensorSpec(shape=[], dtype=tf.int32),
            'inda': tf.TensorSpec(shape=[None], dtype=tf.int32),
            'indb': tf.TensorSpec(shape=[None], dtype=tf.int32),
            'nn_vecs': tf.TensorSpec(shape=[None,3], dtype=tf.float32),
            'nn_r': tf.TensorSpec(shape=[None], dtype=tf.float32),
            'species': tf.TensorSpec(shape=[None], dtype=tf.int32),
            'sp_a': tf.TensorSpec(shape=[None], dtype=tf.int32),
            'sp_b': tf.TensorSpec(shape=[None], dtype=tf.int32),
            'energy': tf.TensorSpec(shape=[], dtype=tf.float32),
            'name': tf.TensorSpec(shape=[], dtype=tf.string),
            'forces': tf.TensorSpec(shape=[None,3], dtype=tf.float32)}
        if input_format=='example':
            output_signature['targets'] = tf.TensorSpec(shape=[None], dtype=tf.int32)
            output_signature['conf_targets'] = tf.TensorSpec(shape=[], dtype=tf.int32)


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
        else:
            if input_format=='example':
                loader_func = load_json
            elif input_format=='spice':
                loader_func = load_spice
            elif input_format=='npy':
                loader_func = load_npy
            elif input_format=='xyz':
                loader_func = load_xyz
            data = tf.data.Dataset.from_generator(loader_func, \
                args=[data_dir,extra_data['Rcut'],
                      extra_data['species'],parse_fn._forces,
                      shuffle,], \
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
            # perform the shuffling, this can be performend only when
            # each iteration we will go through a different shuffling
            # eg from TF doc:
            # dataset = tf.data.Dataset.range(3)
            # dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
            # dataset = dataset.repeat(2)
            # [1, 0, 2, 1, 2, 0]
            data = data.shuffle(buffer_size=batch_size * shuffle_buffer_size_multiplier,
                                reshuffle_each_iteration=True)

        if input_format=='tfr':
            # perform batching
            data = data.batch(batch_size=batch_size)
            # Unpack the data and perform the processing
            data = data.map(map_func=parse_fn, num_parallel_calls=num_parallel_calls)
        else:
            # Ragged batch, then making a list and padding
            data = data.ragged_batch(batch_size=batch_size, drop_remainder=not oneshot)
            data = data.map(partial(pad_batch, extra_data['at_mod'], extra_data['pair_mod'], 
                                    len(extra_data['species']), batch_size))

        # prefetch a number of batches
        data = data.prefetch(buffer_size=prefetch_buffer_size_multiplier)
    return data
