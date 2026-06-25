###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import configparser
from functools import partial
import logging
import os
import sys

import numpy as np

from panna import gvector
from panna.lib.tfr_data_structure import tfr_writer, example_tf_packer
from panna.lib import init_logging
from panna.lib.example_bin import load_example

# logger
logger = logging.getLogger('panna')


def recover_bin_files(in_path):
    """ find bin file in a given path
    """
    all_example_files = []
    for file in os.listdir(in_path):
        _name, ext = os.path.splitext(file)
        if ext == '.bin':
            file_name = os.path.join(in_path, file)
            all_example_files.append(file_name)
    return all_example_files


def packer_func(gv_param):
    """Return the packer function with all parameters set."""
    derivatives = gv_param.getboolean('include_derivatives', False)
    sparse_derivatives = gv_param.getboolean('sparse_derivatives', False)
    per_atom_quantity = gv_param.getboolean('include_per_atom_quantity', False)
    long_range_el = gv_param.getboolean('long_range_el', False)
    return partial(example_tf_packer,
                   forces=derivatives,
                   sparse_dgvect=sparse_derivatives,
                   per_atom_quantity=per_atom_quantity,
                   long_range_el=long_range_el)


def partition_files(files, partition_size):
    """ partition the files in sub-lists of size partition_size
    if element are not enough last list will be of different size
    """
    n_parts = int(np.ceil(len(files) / partition_size))

    # divided files in subsets, each set is a new tfr file
    files = [
        files[i * partition_size:(i + 1) * partition_size]
        for i in range(n_parts)
    ]
    return files


def tfr_packer_writer(files, packer, out_path, prefix=None):
    """Save the data in tfrs in the out_path.

    Parameters
    ----------
    files: list of list of path to bin files
    packer: func
        a packer function
    out_path: path/stirng
    prefix: string, optional
         a prefix for the tfrs names

    Returns
    -------
    None

    Store len(files) tfrs files in out path.
    The name of a tfrs is: prefix-{file number}-{of files}.tfrecord
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if prefix:
        file_pattern = prefix + '-{}-{}'
    else:
        file_pattern = '{}-{}'

    n_tfrs = len(files)

    for idx, tfr_elements in enumerate(files):
        logger.info('file %d/%d', idx + 1, n_tfrs)

        filename = file_pattern.format(idx + 1, n_tfrs)

        # check if file already exists
        target_file = os.path.join(out_path, '{}.tfrecord'.format(filename))
        if os.path.isfile(target_file) and (os.path.getsize(target_file) > 0):
            logger.info('file already computed')
            continue

        payload = [packer(load_example(x)) for x in tfr_elements]

        tfr_writer(filename=filename, path=out_path, data=payload)


def main():
    parser = argparse.ArgumentParser(description='TFR packer')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='config file',
                        required=True)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    init_logging()

    io_param = config['IO_INFORMATION']
    in_path = io_param.get('input_dir', None)
    out_path = io_param.get('output_dir', None)
    examples_files = recover_bin_files(in_path)
    if len(examples_files) == 0:
        logger.info('No example found. Stopping')
        sys.exit(1)

    partition_size = io_param.getint('elements_per_file', 1000)
    examples_files = partition_files(examples_files, partition_size)

    gv_param = config['CONTENT_INFORMATION']
    my_example_tf_packer = packer_func(gv_param)
    prefix = io_param.get('prefix', None)

    tfr_packer_writer(examples_files, my_example_tf_packer, out_path, prefix)


if __name__ == '__main__':
    main()
