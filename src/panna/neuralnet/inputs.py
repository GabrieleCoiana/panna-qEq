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
import tensorflow as tf


def input_iterator(data_dir,
                   batch_size,
                   parse_fn,
                   name,
                   shuffle_buffer_size_multiplier=10,
                   prefetch_buffer_size_multiplier=20,
                   num_parallel_readers=8,
                   num_parallel_calls=8,
                   cache=False,
                   *args,
                   oneshot=None):
    """Construct input iterator.

    Parameters
    ----------
        data_dir: directory for data, must contain a
                  "train_tf subfolder"
        batch_size: batch size
        parse_fn: function to parse the data from tfrecord file
        name: name scope
        *_buffer_size_multiplier: batchsize times this number
        num_parallel_readers: process that are doing Input form drive
        num_parallel_calls: call of the parse function

        oneshot: experimental, do not set

        TODO: construct a double system to handle in_place
              evaluation of accuracy

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
    with tf.name_scope(name):
        with tf.device('/cpu:0'):
            # create a dataset of strings with filename
            # order is not deterministic, can chenge in future version
            data_files = tf.data.Dataset.list_files(
                os.path.join(data_dir, "*.tfrecord"))
            # apply dataset transformation and return
            # a  ParallelInterleaveDataset
            # that is a dataset, the original one with a function.
            dataset = data_files.apply(
                # emine: this is being deprecated soon..
                tf.contrib.data.parallel_interleave(
                    # The function embedded in the
                    # ParallelInterleaveDataset, read the file
                    lambda filename: tf.data.TFRecordDataset(filename),
                    cycle_length=num_parallel_readers))
            if cache:
                dataset = dataset.cache()
            dataset = dataset.shuffle(
                buffer_size=batch_size * shuffle_buffer_size_multiplier)
            if not oneshot:
                dataset = dataset.repeat()
            #define how to prefetch data
            dataset = dataset.prefetch(
                buffer_size=batch_size * prefetch_buffer_size_multiplier)
            #define batchsize of the dataset
            dataset = dataset.batch(batch_size=batch_size)
            #define how to parse the elements in the batch
            dataset = dataset.map(
                map_func=parse_fn, num_parallel_calls=num_parallel_calls)

    return dataset.make_initializable_iterator()
