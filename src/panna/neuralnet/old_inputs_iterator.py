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

from panna.neuralnet.parse_fn import load_json

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
    # Dictionaries needed for example parsing
    if input_format=='example':
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
            data = tf.data.Dataset.from_generator(load_json, \
                args=[data_dir,extra_data['mincut'],extra_data['maxcut'],
                      extra_data['species'],parse_fn._forces], \
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
        elif input_format=='example':
            # batching and padding the example dicts
            data = data.padded_batch(batch_size=batch_size,
                        padded_shapes=padded_shapes,
                        padding_values=padding_values)

        # prefetch a number of batches
        data = data.prefetch(buffer_size=prefetch_buffer_size_multiplier)
    return data
