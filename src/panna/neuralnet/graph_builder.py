import os

import tensorflow as tf

from .inputs import input_iterator
from .regularizations import l1l2_regularizations
from .train_ops import train_neural_network


def graph_builder(config, system_scaffold):
    """
    Parameters
    ----------
      config: config file
      system_scaffold: a valid scaffold
    """

    # aux computational variables
    global_step = tf.train.get_or_create_global_step()

    io_info = config['IO_INFORMATION']
    lr_parms = config['TRAINING_PARAMETERS']
    # TODO if empty should work, not crash
    parallel = config['PARALLELIZATION']

    if io_info.get('input_format', 'TFR') == "TFR":
        _parse_fn = system_scaffold.tfr_parse_function
    else:
        raise ValueError('input_format not available')

    batch_size = lr_parms.getint('batch_size')
    if not os.path.isdir(io_info.get('data_dir')):
        raise FileNotFoundError(io_info.get('data_dir'))
    t_iterator = input_iterator(
        io_info.get('data_dir'),
        batch_size,
        _parse_fn,
        'train',
        parallel.getint('shuffle_buffer_size_multiplier', 10),
        parallel.getint('prefetch_buffer_size_multiplier', 10),
        parallel.getint('num_parallel_readers', 1),
        parallel.getint('num_parallel_calls', 1),
        cache=parallel.getboolean('dataset_cache', False))

    # this call recover all the batches needed for a step
    batches = t_iterator.get_next()

    # creation of the lr function
    learning_rate = lr_parms.getfloat('learning_rate')

    if not lr_parms.getboolean('learning_rate_constant', True):
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step,
            lr_parms.getfloat('learning_rate_decay_step'),
            lr_parms.getfloat('learning_rate_decay_factor'),
            staircase=False)

    loss, predictions = system_scaffold.tf_network(batch_size, batches)

    tf.summary.scalar('learning_rate', learning_rate)

    # creation of the loss quantities

    w_l_norm_sum, b_l_norm_sum = l1l2_regularizations(
        lr_parms.getfloat('wscale_l1',
                          0.0), lr_parms.getfloat('wscale_l2', 0.0),
        lr_parms.getfloat('bscale_l1', 0.0),
        lr_parms.getfloat('bscale_l2', 0.0))

    minimizable_quantity = tf.add_n([loss, w_l_norm_sum, b_l_norm_sum],
                                    name='Total_Loss')

    tf.add_to_collection('losses', minimizable_quantity)
    train_op = train_neural_network(
        minimizable_quantity,
        global_step,
        learning_rate,
        lr_parms.getfloat('beta1', 0.9),
        lr_parms.getfloat('beta2', 0.999),
        lr_parms.getfloat('adam_eps', 1e-8),
        clip_value=lr_parms.getfloat('clip_value', 0.0))

    # add all the loss quantities to the graph
    losses = tf.get_collection('losses')

    for loss in losses:
        name = loss.op.name.split("/")[-1]
        tf.summary.scalar("1.Losses/" + name, loss)

    # add all the stat quantities to the graph
    losses = tf.get_collection('training_stats')

    for loss in losses:
        name = loss.op.name.split("/")[-1]
        tf.summary.scalar("2.training_stats/" + name, loss)

    return train_op, t_iterator, predictions, global_step, loss
