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
import os

import numpy as np

import gvector
import neuralnet
from lib import init_logging

# logger
logger = logging.getLogger('panna')


def parser_file(conf_file):
    """parser file

    in_path: path to the directory to be compressed
    out_path: path to the save directory
    elements_per_file: number of element for each tfrecord file
    n_atoms: simulation info: number of atoms in each file
    num_species: simulation info: number of species
    g_size: simulation info: size of g-vector
    prefix: an extra prefix, optional, it is added in front of the file_name
              eg: train for the training set

    IMPORTANT:
      In the in_path there must be ONLY files that are binary representation
      of already calculated g-vector

    """
    config = configparser.ConfigParser()
    config.read(conf_file)

    class Parameters():
        pass

    parameters = Parameters()

    io_param = config['IO_INFORMATION']
    parameters.in_path = io_param.get('input_dir', None)
    parameters.out_path = io_param.get('output_dir', None)
    parameters.elements_per_file = io_param.getint('elements_per_file', 1000)
    parameters.prefix = io_param.get('prefix', '')

    gv_param = config['CONTENT_INFORMATION']
    parameters.num_species = gv_param.getint('n_species', None)
    parameters.derivatives = gv_param.getboolean('include_derivatives', False)
    parameters.sparse_derivatives = gv_param.getboolean(
        'sparse_derivatives', False)
    parameters.per_atom_quantity = gv_param.getboolean(
        'include_per_atom_quantity', False)

    return parameters


def main(parameters):
    """Package all the gvector in a folder to tfrecord files
    """

    all_example_names = []
    for file in os.listdir(parameters.in_path):
        name, ext = os.path.splitext(file)
        if ext=='.bin':
            all_example_names.append(file)
    if len(all_example_names)==0:
        logger.info('No example found. Stopping')
        exit(1)
    files = [
        os.path.join(parameters.in_path, x)
        for x in all_example_names
    ]
    n_files = int(np.ceil(len(files) / parameters.elements_per_file))

    # divided files in subsets, each set is a new tfr file
    files = [
        files[i * parameters.elements_per_file:(i + 1) *
              parameters.elements_per_file] for i in range(n_files)
    ]

    if not os.path.exists(parameters.out_path):
        os.makedirs(parameters.out_path)

    if parameters.prefix != '':
        file_pattern = parameters.prefix + '-{}-{}'
    else:
        file_pattern = '{}-{}'

    n_tfrs = len(files)

    for idx, tfr_elements in enumerate(files):
        logger.info('file %d/%d', idx + 1, n_tfrs)

        filename = file_pattern.format(idx + 1, n_tfrs)
        target_file = os.path.join(parameters.out_path,
                                   '{}.tfrecord'.format(filename))

        if os.path.isfile(target_file) and (os.path.getsize(target_file) > 0):
            logger.info('file already computed')
            continue

        payload = [
            gvector.example_tf_packer(neuralnet.load_example(x),
                                      parameters.derivatives,
                                      parameters.sparse_derivatives,
                                      parameters.per_atom_quantity)
            for x in tfr_elements
        ]
        gvector.tfr_writer(filename=filename,
                       path=parameters.out_path,
                       data=payload)


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser(description='TFR packer')
    PARSER.add_argument('-c',
                        '--config',
                        type=str,
                        help='config file',
                        required=True)
    ARGS = PARSER.parse_args()
    PARAMETERS = parser_file(ARGS.config)
    main(PARAMETERS)
