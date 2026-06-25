###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import configparser
import logging
import multiprocessing as mp
import os
import sys

import numpy as np

from panna import gvector
from panna.lib import ExampleJsonWrapper
from panna.lib.log import emit_splash_screen, init_logging
from panna.lib.parser_callable import converters
from panna.gvector.LR_electrostatic_term import Ewald_summation

np.set_printoptions(16)

logger = logging.getLogger('panna')

def _parse_file(conf_file):
    """ Parser helper
    """
    config = configparser.ConfigParser(converters=converters)
    config.read(conf_file)
    return config


def _symmetry_function(symmetry_function: configparser.ConfigParser,
                       pbc_directions: list = None):
    """

    Parameters
    ----------
    symmetry_function : config.ConfigParser
        'SYMMETRY_FUNCTION' section of the configparser
    pbc_directions: list of 3 Boolean values
        pbc in first, second, third lattice vector directions (often, this will be  x, y, z). 
        This directive override any other option,
        if left blank pbc are inferred from example files

    Returns
    -------
    gvector function
        a function ready to be used to compute gvectors. The function
        depends on the input.
    """
    compute_dgvect = symmetry_function.getboolean('include_derivatives', False)
    if compute_dgvect:
        sparse_dgvect = symmetry_function.getboolean('sparse_derivatives', False)
    else:
        sparse_dgvect = False

    species = symmetry_function.get('species', None)

    g_type = symmetry_function.get('type')

    # New g vector types should be added below:
    if g_type == 'mBP':
        gvect_func = gvector.GvectmBP(compute_dgvect=compute_dgvect,
                                      species=species,
                                      sparse_dgvect=sparse_dgvect,
                                      pbc_directions=pbc_directions)

    elif g_type == 'BP':
        gvect_func = gvector.GvectBP(compute_dgvect=compute_dgvect,
                                     species=species,
                                     sparse_dgvect=sparse_dgvect,
                                     pbc_directions=pbc_directions)

    elif g_type == 'SB':
        gvect_func = gvector.GvectSB(compute_dgvect=compute_dgvect,
                                     species=species,
                                     sparse_dgvect=sparse_dgvect,
                                     pbc_directions=pbc_directions)

    elif g_type == 'B2':
        gvect_func = gvector.GvectB2(compute_dgvect=compute_dgvect,
                                     species=species,
                                     sparse_dgvect=sparse_dgvect,
                                     pbc_directions=pbc_directions)
    elif g_type == 'WmBP':
        gvect_func = gvector.GvectWmBP(compute_dgvect=compute_dgvect,
                                       species=species,
                                       sparse_dgvect=sparse_dgvect,
                                       pbc_directions=pbc_directions)
    else:
        raise ValueError('Not recognized symmetry function type')

    return gvect_func


def extract_gvect_func(config):
    """ Find the gvector function in the configuration file,
    we need to know the PBC as well because it changes how the 
    gvector functions are called.

    Parameters
    ----------
    config : config.ConfigParser
        a config file for the gvector calculator

    Return
    ------
    gvector function
        a function ready to be used to compute gvectors. The function
        depends on the input.
    """

    # PBC related prameters
    if config.has_section('PBC'):
        logger.warning('PBC will be read from configuration file, json must be consistent '
                       'otherwise the code will fail')
        if config.has_option('PBC','use_pbc'):              
            pbc_directions = np.asarray(
                config['PBC'].get_comma_list_booleans('use_pbc'))
        else:
            # for back compatibility only, don't carry to newer versions of the code:
            pbc_directions = np.asarray(
                [config['PBC'].getboolean('pbc{}'.format(x), False) for x in range(1, 4)])
    else:
        pbc_directions = None
        logger.info('PBC will be determined by lattice parameters in the json file for each example')

    gvect_func = _symmetry_function(config['SYMMETRY_FUNCTION'], pbc_directions)

    # Every gvect type may have different param set, needs its own parser to parse the config
    gvect_func.parse_parameters(config['GVECT_PARAMETERS'])

    logger.info('g type: %s', gvect_func.name)
    logger.info('g DOI: %s', gvect_func.doi)
    logger.info('g size: %d', gvect_func.gsize)
    return gvect_func


def _remove_already_computed_keys(all_example_keys, log_dir):
    """
    Parameters
    ----------
    all_example_keys : list of string
    logdir: string
    Returns
    -------
    list of string
        non computed keys
    """
    try:
        with open(os.path.join(log_dir, 'gvect_already_computed.dat')) as file_stream:
            logger.info('the computation has been restarted (a file named '
                        'gvect_already_computed.dat has been found).'
                        'Already computed files will not be recomputed.')
            keys_already_computed = file_stream.read().split(',')
    except FileNotFoundError:
        keys_already_computed = []
    keys_already_computed = set(keys_already_computed)
    all_example_keys = set(all_example_keys)

    logger.info('computed keys %d/%d',
                max(len(keys_already_computed), 1) - 1, len(all_example_keys))

    return list(all_example_keys - keys_already_computed)


def compute(config, gvect_func, *, cli_folder_info=None):
    """Compute main method

    Parameters
    ----------
    config : configparser.ConfigParser
    gvect_func : gvector function

    cli_folder_info : str
        folder where to execute the compute function
        this folder override the one in the config file
        The structure must be the following:

        folder_info
        + examples

    Outputs
    -------
    A file in the log dir with all the computed keys
    """

    # IO INFO
    folder_info = config['IO_INFORMATION']
    input_json_dir = folder_info.get('input_json_dir', None)
    binary_out_dir = folder_info.get('output_gvect_dir', './bin')
    log_dir = folder_info.get('log_dir', '.')

    if cli_folder_info:
        logger.info('overriding folder parameters with cli values')
        input_json_dir = os.path.join(folder_info, 'examples')
        binary_out_dir = os.path.join(folder_info, 'bin')
        log_dir = folder_info

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(binary_out_dir):
        os.makedirs(binary_out_dir)

    # miscellaneous parameters. if you add a new experimental parameter, add it here:
    if 'MISC' in config:
        misc = config['MISC']
        per_atom_energy = misc.getboolean('per_atom_energy', False)
    else:
        per_atom_energy = False

    if 'LONG_RANGE' in config:
        LR = config['LONG_RANGE']
        acc_factor = LR.getfloat('acc_factor', 1e-6)
        kmax = LR.getfloat('kmax', None)
        #rs is the screening length
        rs = LR.getfloat('r_screen', None)
        long_range_el = LR.getboolean('long_range_el', False)
        PBC = LR.getboolean('PBC', True)
    else:
        long_range_el = False
        acc_factor = 1e-6

    # Gvectors will be calculated only for files that end with .example extension 
    all_example_keys = []
    for file in os.listdir(input_json_dir):
        name, ext = os.path.splitext(file)
        if ext == '.example':
            all_example_keys.append(name)
    if len(all_example_keys) == 0:
        logger.info('No example found. Stopping')
        sys.exit(1)

    example_keys = _remove_already_computed_keys(all_example_keys, log_dir)

    # parallelization related parameters
    number_of_process = config['PARALLELIZATION'].getint('number_of_process', 1)
    pool = mp.Pool(number_of_process)

    log_for_recover = open(os.path.join(log_dir, 'gvect_already_computed.dat'), 'a')

    logger.info('--start--')
    while True:
        logger.info('----run----')
        parallel_batch = []
        elements_in_buffer = 0
        while elements_in_buffer < number_of_process:
            try:
                key = example_keys.pop()
            except IndexError:
                # exit conditions form process population if
                # no more key to compute
                break

            logger.debug("loading %s:%s", elements_in_buffer, key)

            # === THE CODE WORKS IN ANGSTROM AND EV ===
            example = ExampleJsonWrapper(
                os.path.join(input_json_dir, '{}.example'.format(key)),
                gvect_func.species_idx_2str)

            # load common quantities
            example_dict = {
                'key': example.key,
                'lattice_vectors': example.angstrom_lattice_vectors,
                'species': example.species_indexes,
                'positions': example.angstrom_positions,
                'E': example.ev_energy
            }
            if long_range_el:
                example_dict['total_charge'] = example.total_charge
                example_dict['atomic_charges'] = example.atomic_charges
                example_dict['species_sequence'] = example.species_str

            # load specific quantities
            if per_atom_energy:
                example_dict['per_atom_quantity'] = example.per_atom_ev_energy

            if gvect_func.compute_dgvect:
                example_dict['forces'] = example.forces

            logger.debug(example_dict)
            parallel_batch.append(example_dict)
            elements_in_buffer += 1

        if not parallel_batch:
            # exit condition if the computation has ended
            break

        logger.info('start parallel computation, %d to go', len(example_keys))

        # COMPUTE gvect and other quantities
        logger.debug('compute gvectors')
        feed = []
        if long_range_el:
            feed_LR = []
        for example_dict in parallel_batch:
            feed.append(
                (example_dict['key'], example_dict['positions'],
                 example_dict['species'], example_dict['lattice_vectors']))
            if long_range_el:
                feed_LR.append(
                    (example_dict['key'], example_dict['positions'],
                     example_dict['lattice_vectors'], acc_factor,
                     example_dict['species_sequence'], PBC,
                     kmax, rs, gvect_func.compute_dgvect))
        if long_range_el:
            feed_LR = pool.starmap(Ewald_summation, feed_LR)
        feed = pool.starmap(gvect_func, feed)

        # reorganize the results
        for example_dict in parallel_batch:
            for element in feed:
                if example_dict['key'] == element['key']:
                    example_dict.update(element)
                    logger.debug('assigned')
            if long_range_el:
                for element in feed_LR:
                    if example_dict['key'] == element['key']:
                        example_dict.update(element)
                        logger.debug('assigned')



        # SAVE computed result
        logger.debug('save gvectors')
        feed = []
        for example_dict in parallel_batch:
            feed.append((example_dict, binary_out_dir))

        pool.starmap(gvector.binary_encoder, feed)

        # final log for recovery
        log_for_recover.write(','.join(
            [example_dict['key'] for example_dict in parallel_batch]))
        log_for_recover.write(',')
        log_for_recover.flush()
    log_for_recover.close()


def main():
    parser = argparse.ArgumentParser(description='Gvectors calculator')

    parser.add_argument('-c', '--config', type=str, help='config file', required=True)
    parser.add_argument('-f',
                        '--folder_info',
                        type=str,
                        help='folder_info, if supplied override config',
                        required=False)
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug flag',
                        required=False)

    args = parser.parse_args()
    init_logging()
    emit_splash_screen(logger)

    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # parse gvector input file
    config = _parse_file(args.config)

    # extract which gvector type and therefore compute function to use
    # this function also sets all the config parameters needed for that gvect type
    gvect_func = extract_gvect_func(config)

    # compute gvector
    compute(config, gvect_func, cli_folder_info=args.folder_info)


if __name__ == '__main__':
    main()
