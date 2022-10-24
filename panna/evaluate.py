###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
""" Code use to evaluate a dataset on a checkpoint/s
note:
    if batchsize == -1 then use the whole dataset
"""
import argparse
import configparser
from functools import partial
import logging
import multiprocessing as mp
import os
import random as rnd
import sys
import time

from tensorflow.errors import NotFoundError

import neuralnet as net
from lib.example_bin import load_example
from lib.log import emit_splash_screen
from lib.parser_callable import converters
from neuralnet.scaffold_selector import scaffold_selector

logger = logging.getLogger('panna')


def _eval_function(example, scaffold, add_offset):
    """Helper for parallelization

    Args:
        example: a example obj
        scaffold: a scaffold obj
    Return:
        a string: filename, n_natoms, dft, prediction

    """
    en_prediction, *_ = scaffold.evaluate(example, add_offset=add_offset)
    en_dft = example.true_energy
    string = '{} {} {} {}'.format(example.name, example.n_atoms, en_dft, en_prediction)
    return string


def _eval_function_forces(example, scaffold, add_offset):
    """Helper for parallelization when computing forces

    Parameters
    ----------
    example:
        the example to be evaluate

    scaffold:
        the scaffold with the networks that will
        be used in the evaluation

    add_offset: bool
        boolean to enable/disable the offset

    Returns
    -------
    string
        'n_atoms reference energy predicted energy'
    string
        'example_id atom_id fx fy fz'
    """
    en_prediction, forces_prediction, *_ = scaffold.evaluate(example=example,
                                                             forces_flag=True,
                                                             add_offset=add_offset)

    n_atoms = example.n_atoms
    en_ref = example.true_energy
    forces_ref = example.forces

    string = '{} {} {} {}'.format(example.name, example.n_atoms, en_ref, en_prediction)

    forces_ref = forces_ref.reshape(n_atoms, 3)
    forces_prediction = forces_prediction.reshape(n_atoms, 3)
    string2_list = [
        example.name + ' {} '.format(idx)
        + ' '.join([str(f) for f in forces_prediction[idx]]) + ' '
        + ' '.join([str(f) for f in forces_ref[idx]]) for idx in range(n_atoms)
    ]
    string2 = '\n'.join(string2_list) + '\n'
    return string, string2


def _ckpts_selector(train_dir, validation_parameters):
    ck_steps = net.Checkpoint.checkpoint_step_list(train_dir)
    single_step = validation_parameters.getboolean('single_step', False)
    step_number = validation_parameters.getint('step_number', None)
    subsampling = validation_parameters.getint('subsampling', None)
    start = validation_parameters.getint('start', 0)
    end = validation_parameters.getint('end', len(ck_steps))

    if (single_step and not step_number):
        logger.info('evaluating last checkpoint')
        ck_steps = [ck_steps[-1]]
    elif (single_step and step_number):
        logger.info('evaluating step number %d', step_number)
        logger.info('not available in this version -- see steps finder')
        raise NotImplementedError()
    elif (not single_step and subsampling):
        logger.info('evaluation of all the checkpoints at steps of %d ', subsampling)
        ck_steps = ck_steps[start:end:subsampling]
    else:

        logger.info(f'evaluation of all the checkpoints in range {start} {end}')
        logger.info('range given in file index and not training step')
        ck_steps = ck_steps[start:end]
    return ck_steps


def collect_data_files(data_dir):
    dataset_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    logger.info('files in the dataset folder: %d ', len(dataset_files))
    return dataset_files


def load_networks(nets_dir, nets_format, format_options=None):
    if nets_format == 'tf_checkpoint':
        ck_steps = _ckpts_selector(nets_dir, format_options)

        # load ckpts
        ckpts = [net.Checkpoint(nets_dir, ck_step) for ck_step in ck_steps]
        #logger.debug('EK checkpoints available: %d' ,len(ckpts))
        # extract scaffols
        scaffolds = []
        for ckpt in ckpts:
            try:
                scaf = ckpt.get_scaffold
                scaffolds.append(scaf)
            except NotFoundError:
                logger.warning('%s not found because of TF bug', ckpt.filename)

    elif nets_format == 'PANNA':
        scaffold = scaffold_selector('PANNA').load_dump(nets_dir)
        scaffolds = [scaffold]

    else:
        logger.info('Unknown network format.')
        sys.exit(0)

    return scaffolds


def open_writers(scaffolds, out_dir, compute_forces):
    scaffolds_non_computed = []
    energy_writers = []
    if compute_forces:
        force_writers = []

    # filter already computed scaffolds
    for scaffold in scaffolds:
        name = scaffold.name
        file_name = os.path.join(out_dir, '{}.dat'.format(name))

        if os.path.isfile(file_name) and os.path.getsize(file_name) > 100:
            logger.info('%s already computed', file_name)
            continue

        scaffolds_non_computed.append(scaffold)

        energy_writers.append(open(os.path.join(out_dir, '{}.dat'.format(name)), 'w'))
        if compute_forces:
            force_writers.append(
                open(os.path.join(out_dir, '{}_forces.dat'.format(name)), 'w'))
    writers = []
    for e_writer in energy_writers:
        writers.append([e_writer])
        e_writer.write('#filename n_atoms e_ref e_nn\n')

    if compute_forces:
        for f_writer, writer in zip(force_writers, writers):
            writer.append(f_writer)
            f_writer.write('#filename atom_id '
                           'fx_nn fy_nn fz_nn '
                           'fx_ref fy_ref fz_ref\n')
    return scaffolds_non_computed, writers


def _load_example(file_name):
    tmp = load_example(file_name)
    # dummy call trigger desparsification
    # tmp.dgvects
    return tmp


def load_single_file(pool, n_loaded_examples, dataset_files, batch_size):
    examples = []
    # when the examples are all separated in different files
    if batch_size == -1 and n_loaded_examples == 0:
        # we want to load all the example in memory
        # this is done only once
        example_files = dataset_files
        tt = time.time()
        logger.info('loading all the example in the folder, '
                    'this operation may take a while')
        examples_parameters = [(x, ) for x in example_files]
        examples = pool.starmap(_load_example, examples_parameters)
        logger.info('loading all the example in the folder,' 'done')
        tt = (time.time() - tt) * 1000
        logger.info(f'time = {tt/1000:2.2f} s, <{tt/len(examples):2.2f}> ms')

    elif batch_size > 0:
        # if the validation dataset is too big can be useful
        # to only randomly sample it
        rnd.shuffle(dataset_files)
        example_files = dataset_files[:batch_size]
        logger.info('loading a subset of all the available examples in the folder')
        examples_parameters = [(x, ) for x in example_files]
        examples = pool.starmap(load_example, examples_parameters)
    return examples


def load_tfr(_pool, n_loaded_examples, dataset_files, g_size, compute_forces,
             sparse_derivatives):
    if n_loaded_examples == 0:
        examples = []
        logger.info('loading TFR, this may take a while')
        for example in net.iterator_over_tfdata(g_size,
                                                *dataset_files,
                                                derivatives=compute_forces,
                                                sparse_derivatives=sparse_derivatives):
            examples.append(example)
        return examples
    return []


def run_eval(scaffolds, writers, number_of_process, load_function, compute_forces,
             add_offset):
    logger.info('----start evaluation----')
    pool = mp.Pool(processes=number_of_process)
    n_scaffolds = len(scaffolds)

    examples = []
    for idx, (scaffold, writer) in enumerate(zip(scaffolds, writers)):
        logger.info('validating network: %d/%d', idx + 1, n_scaffolds)
        examples_new = load_function(pool, len(examples))
        if examples_new:
            examples = examples_new
        if not examples:
            logger.info('no example found')
            break

        parallel_feed = [(example, scaffold, add_offset) for example in examples]

        if compute_forces:
            tt = time.time()
            energy_forces = pool.starmap(_eval_function_forces, parallel_feed)
            string = '\n'.join([x[0] for x in energy_forces])
            string2 = ''.join([x[1] for x in energy_forces])
            tt = (time.time() - tt) * 1000
            logger.info(f'time = {tt/1000:2.2f} s, <{tt/len(parallel_feed):2.2f}> ms')
        else:
            string = '\n'.join(pool.starmap(_eval_function, parallel_feed))

        writer[0].write(string)
        writer[0].flush()
        if compute_forces:
            writer[1].write(string2)
            writer[1].flush()

    for writer in writers:
        writer[0].close()
        if compute_forces:
            writer[1].close()


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='configuration filename',
                        required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug flag, not working for now',
                        required=False)
    args = parser.parse_args()

    emit_splash_screen(logger)

    config = configparser.ConfigParser(converters=converters)
    config.read(args.config)

    io_info = config['IO_INFORMATION']
    validation_options = config['VALIDATION_OPTIONS']
    dataset_files = collect_data_files(io_info.get('data_dir'))
    scaffolds = load_networks(io_info.get('train_dir'),
                              io_info.get('networks_format', 'tf_checkpoint'),
                              validation_options)

    out_dir = io_info.get('eval_dir')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if config.has_section('PARALLELIZATION'):
        number_of_process = config['PARALLELIZATION'].getint('number_of_process', 4)
    else:
        number_of_process = 4

    compute_forces = validation_options.getboolean('compute_forces', False)

    scaffolds, writers = open_writers(scaffolds, out_dir, compute_forces)

    example_format = io_info.get('example_format', 'TFR')
    add_offset = validation_options.getboolean('add_offset', True)

    if example_format == 'single_files':
        batch_size = validation_options.getint('batch_size', -1)
        load = partial(load_single_file,
                       dataset_files=dataset_files,
                       batch_size=batch_size)
    elif example_format == 'TFR':
        g_size = config['TFR_STRUCTURE'].getint('g_size')
        sparse_derivatives = config['TFR_STRUCTURE'].getboolean(
            'sparse_derivatives', False)
        load = partial(load_tfr,
                       dataset_files=dataset_files,
                       g_size=g_size,
                       compute_forces=compute_forces,
                       sparse_derivatives=sparse_derivatives)
    else:
        raise ValueError('Example format not supported')
    run_eval(scaffolds, writers, number_of_process, load, compute_forces, add_offset)


if __name__ == '__main__':
    main()
