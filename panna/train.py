###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import control_flow_ops, resources, variables

from lib.log import emit_splash_screen, splash_screen_record
from neuralnet import parameter_file_parser
from neuralnet.graph_builder import graph_builder
from neuralnet.scaffold_selector import scaffold_getter

logger = logging.getLogger('panna')


def setup_environment(flags):  # pylint: disable=too-many-branches
    """ Setup the environment

    - resolve parallelization over nodes
    - open loggers
    - log splash screen
    """
    # extrapolate info for parallel running
    if flags.list_of_nodes != '':
        list_of_nodes = flags.list_of_nodes.split(',')
    else:
        list_of_nodes = ['localhost']

    if flags.task_index_variable != '':
        task_index = int(os.environ[flags.task_index_variable])
    else:
        task_index = 0

    if len(list_of_nodes) > 1 and flags.task_index_variable != '':
        # parallel run
        parallel_environment = True
    else:
        if not flags.debug_parallel:
            # serial run
            parallel_environment = False
        else:
            # fake parallel for debug on a single machine
            parallel_environment = True
            list_of_nodes = ['one', 'two']
            task_index = flags.debug_parallel_index

    # set up file loggers
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fsh = logging.FileHandler('{}_{}.log'.format(list_of_nodes[task_index],
                                                 time.strftime('%H_%M_%S')))
    fsh.setFormatter(formatter)
    # splash screen also here
    fsh.emit(splash_screen_record)

    logger.addHandler(fsh)
    # now that loggers are complete we can f
    # do the splash screen in every file logger

    # creation of the cluster
    if not flags.debug_parallel:
        if flags.parameter_servers:
            logger.info('parameter server (ps) is active: '
                        'please optimize the number of ps'
                        'wrt your model, usually 1 should fit')

            if flags.parameter_servers > len(list_of_nodes) - 1:
                logger.info('No nodes available as workers')
                sys.exit(1)

            list_of_worker_nodes = [
                '{}:{}'.format(x, flags.communication_port)
                for x in list_of_nodes[:-flags.parameter_servers]
            ]
            list_of_parameters_nodes = [
                '{}:{}'.format(x, flags.communication_port)
                for x in list_of_nodes[-flags.parameter_servers:]
            ]

            logger.info('ps/nodes: {}/{}'.format(len(list_of_parameters_nodes),
                                                 len(list_of_nodes)))
            logger.info('workers/nodes: {}/{}'.format(len(list_of_worker_nodes),
                                                      len(list_of_nodes)))
        else:
            list_of_worker_nodes = [
                '{}:{}'.format(x, flags.communication_port) for x in list_of_nodes
            ]

            list_of_parameters_nodes = []
    else:
        # fake cluster for debug
        list_of_worker_nodes = ['localhost:22222']
        list_of_parameters_nodes = ['localhost:22223']

    logger.debug('list of ps: {}'.format(','.join(list_of_parameters_nodes)))
    logger.debug('list of wk: {}'.format(','.join(list_of_worker_nodes)))
    if parallel_environment:
        cluster = tf.train.ClusterSpec({
            'worker': list_of_worker_nodes,
            'ps': list_of_parameters_nodes
        })
    else:
        cluster = tf.train.ClusterSpec({
            'worker': list_of_worker_nodes,
        })
    # extended splash screen
    if parallel_environment:
        logger.info('running in parallel environment')
        logger.info('node name: {}'.format(list_of_nodes[task_index]))
        logger.info('node: {}/{}'.format(task_index + 1, len(list_of_nodes)))
        logger.info('is chief: {}'.format((task_index == 0)))
    else:
        logger.info('running in serial environment')

    return list_of_nodes, cluster, task_index


def setup_folders(train_dir, scaffold):
    if tf.gfile.Exists(train_dir):
        for fname in tf.gfile.ListDirectory(train_dir):
            if (fname == 'checkpoint' or fname == 'graph.pbtxt'
                    or fname.split(".")[0] == 'event'
                    or fname.split(".")[0] == 'model'):
                # here we could just make a new directory
                # inside and continue rather than exit
                logger.info('saved checkpoint present, '
                            'networks will be restarted form here')
                break
    else:
        tf.gfile.MakeDirs(train_dir)
    if not tf.gfile.Exists(os.path.join(train_dir, 'networks_metadata.json')):
        scaffold.dump_scaffold_metadata(train_dir)


def train(flags, config, system_scaffold, cluster, task_index):
    """Main train operation
    """
    # split parameters in subinstance
    # io_parameters, parallelization_parameters,\
    # train_parameters, system_scaffold = parameters

    io_info = config['IO_INFORMATION']
    log_frequency = io_info.getint('log_frequency', 1)
    tensorboard_log_frequency = io_info.getint('tensorboard_log_frequency',
                                               log_frequency)

    parallel = config['PARALLELIZATION']

    lr_parms = config['TRAINING_PARAMETERS']
    batch_size = lr_parms.getint('batch_size')
    avg_speed_steps = io_info.getint('avg_speed_steps', log_frequency)

    with tf.Graph().as_default():
        # set local machine as default device for operation
        # this construction require a ps, fix later to support no ps
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            # build the graph
            train_op, t_iterator, predictions, global_step, loss =\
                graph_builder(config, system_scaffold)

            # config the  Proto
            config_proto = tf.ConfigProto(
                log_device_placement=io_info.get('log_device_placement', False),
                inter_op_parallelism_threads=parallel.getint(
                    'inter_op_parallelism_threads', 0),
                intra_op_parallelism_threads=parallel.getint(
                    'intra_op_parallelism_threads', 0))

            # building of the ckpt
            saver = tf.train.Saver(
                max_to_keep=io_info.getint('max_ckpt_to_keep', 100000),
                sharded=True)

            init_op = control_flow_ops.group(
                variables.global_variables_initializer(),
                resources.initialize_resources(resources.shared_resources()),
            )
            # init op are run only by chief
            # local init op are run by every worker
            scaffold = tf.train.Scaffold(init_op=init_op,
                                         saver=saver,
                                         local_init_op=[t_iterator.initializer])

            # HOOKS
            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime.
                """
                def __init__(self, *args, starting_step=0, **kwargs):
                    self._starting_step = starting_step
                    self._start_time = time.time()
                    self._step = 0
                    self._writer = None
                    self._writer2 = None

                def begin(self):
                    self._step = self._starting_step
                    self._start_time = time.time()
                    self._writer = open('delta_e.dat', 'w')
                    self._writer2 = open('delta_e_std.dat', 'w')

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs([predictions, global_step])

                def after_run(self, run_context, run_values):
                    if self._step <= 0:
                        return
                    if self._step % log_frequency == 0:

                        delta_e = run_values.results[0]

                        self._writer.write(
                            ' '.join(['%.8f' % (x) for x in delta_e]) + '\n')
                        self._writer2.write(
                            '%.8f %.8f %.8f' % (np.mean(delta_e),
                                                np.mean(np.abs(delta_e)),
                                                np.std(delta_e)) + '\n')
                        self._writer.flush()
                        self._writer2.flush()
                        time_str = time.strftime("%m-%d %H:%M:%S", time.gmtime())
                        logger.info('{}, global step: {}'.format(
                            time_str, run_values.results[1]))

                    if self._step % avg_speed_steps == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        if duration > 0.0:
                            self._start_time = current_time
                            examples_per_sec = avg_speed_steps * batch_size / duration
                            sec_per_batch = duration / avg_speed_steps
                            format_str = '{:4.2f} examples/sec, {:4.2f} sec/batch'
                            logger.info(format_str.format(examples_per_sec,
                                                          sec_per_batch))

            hooks = [
                tf.train.StopAtStepHook(
                    last_step=config['TRAINING_PARAMETERS'].getint('max_steps'))
            ]

            chief_hooks = [tf.train.NanTensorHook(loss), _LoggerHook()]
            # end of hooks

            server = tf.train.Server(cluster, job_name="worker", task_index=task_index)

            with tf.train.MonitoredTrainingSession(
                    master=server.target,
                    is_chief=(task_index == 0),
                    checkpoint_dir=io_info.get('train_dir'),
                    scaffold=scaffold,
                    save_summaries_steps=tensorboard_log_frequency,
                    log_step_count_steps=tensorboard_log_frequency * 100,
                    # note: every op trigger the hook!!
                    hooks=hooks,
                    chief_only_hooks=chief_hooks,
                    save_checkpoint_steps=io_info.getint('save_checkpoint_steps', 1),
                    config=config_proto) as mon_sess:
                if flags.debug:
                    mon_sess = tf_debug.LocalCLIDebugWrapperSession(mon_sess)
                    logger.debug('---variable of the session---')
                    logger.debug('\n'.join(
                        [' ' + var.name for var in tf.global_variables()]))
                    logger.debug('-----')
                # infinite loop, only hooks act here
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default='',
                        help="config file",
                        required=True)
    parser.add_argument('--list_of_nodes',
                        type=str,
                        default='',
                        help='comma-separated list of nodes')
    parser.add_argument("--task_index_variable",
                        type=str,
                        default='',
                        help="task index variable name, depends on workload manager")
    parser.add_argument("--debug",
                        action='store_true',
                        default=False,
                        help='enable debug mode, partial support only')
    parser.add_argument(
        "--debug_parallel",
        action='store_true',
        help='debug parallel flag, use to have 2 server in local config')
    parser.add_argument("--parameter_servers",
                        type=int,
                        default=0,
                        help='number of parameter server, default 0')
    parser.add_argument("--debug_parallel_index", type=int, help='index 0 or 1')
    parser.add_argument("--communication_port",
                        type=int,
                        default=22222,
                        help='communication port for parallel implementation')
    flags, _unparsed = parser.parse_known_args()

    
    if flags.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    # set up console logger only for the chief
    if flags.task_index_variable != '':
        task_index = int(os.environ[flags.task_index_variable])
    else:
        task_index = 0

    if task_index == 0:
        emit_splash_screen(logger)
    parameters = parameter_file_parser(flags.config)

    list_of_nodes, cluster, task_index = setup_environment(flags)
    scaffold = scaffold_getter(parameters)
    setup_folders(parameters['IO_INFORMATION'].get('train_dir'), scaffold)

    if task_index >= (len(list_of_nodes) - flags.parameter_servers):
        logger.info('Parameter server.')
        # realign task_index at zero
        task_index = task_index - len(list_of_nodes) + flags.parameter_servers
        server = tf.train.Server(cluster, job_name="ps", task_index=task_index)
        server.join()
    else:
        train(flags, parameters, scaffold, cluster, task_index)


if __name__ == '__main__':
    main()
