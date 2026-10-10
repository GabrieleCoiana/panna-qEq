###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""This tool can be used to update old binary files to the new format.
Attention! this tool have some hard coded global variables that need to be
changed

supported conversion:
  - no_version to v0
  - no_version to v0_sparse
  - v0 to v0_sparse

CONVERSION_PARAMETERS is a dict with all additional parameters that can be
needed by the converter.
"""
import argparse
import logging
import multiprocessing as mp
import os
import sys
from itertools import repeat
import numpy as np

try:
  from panna.gvector.binary_encoder import binary_encoder
  from panna.lib.log import emit_splash_screen
  from panna.lib.example_bin import Example, load_example
except:
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from gvector.binary_encoder import binary_encoder
  from lib.log import emit_splash_screen
  from lib.example_bin import Example, load_example

logger = logging.getLogger(__name__)

STARTING_VERSION = 'v0'
END_VERSION = 'v0_sparse'
CONVERSION_PARAMETERS = {'derivatives': True, 'per_atom_quantity': False}


def _load_example_preversion(filename, derivatives=False):
    """ Load binary example
    """
    key = filename.split('/')[-1]
    data = np.fromfile(filename, np.float32)

    n_atoms = int(data[0])
    g_size = int(data[1])
    en = data[2]
    spec_tensor_bytes = n_atoms
    gvect_tensor_bytes = n_atoms * g_size
    prev_bytes = 3

    spec_tensor = data[prev_bytes:prev_bytes + spec_tensor_bytes]
    prev_bytes += spec_tensor_bytes
    gvect_tensor = data[prev_bytes:prev_bytes + gvect_tensor_bytes]
    prev_bytes += gvect_tensor_bytes

    if derivatives:
        if data.size == prev_bytes:
            raise ValueError('Derivatives requested but not '
                             'present in the file')
        dgvec_tensor_bytes = n_atoms**2 * g_size * 3
        dgvect_tensor = data[prev_bytes:prev_bytes + dgvec_tensor_bytes]
        prev_bytes += dgvec_tensor_bytes

        if data.size > prev_bytes:
            # If there is more, then forces are stored
            forces_bytes = n_atoms * 3
            forces = data[prev_bytes:prev_bytes + forces_bytes]
        else:
            forces = []
    else:
        dgvect_tensor = []
        forces = []

    # building the data
    en = np.reshape(en, [1])[0]
    spec_tensor = np.int64(np.reshape(spec_tensor, [n_atoms]))
    gvect_tensor = np.reshape(gvect_tensor, [n_atoms, g_size])
    if (derivatives):
        dgvect_tensor = np.reshape(dgvect_tensor,
                                   [n_atoms, g_size, n_atoms * 3])

    return Example(g_vectors=gvect_tensor,
                   species_vector=spec_tensor,
                   true_energy=en,
                   d_g_available=derivatives,
                   d_g_vectors=dgvect_tensor,
                   forces=forces,
                   name=key)


def _extension_chooser(version):
    if version == 'no_version':
        logger.info('this version had no extension, '
                    'please check that folder contains only files '
                    'that are binary examples')
        extension = ''
    else:
        logger.info('file with bin extension will be loaded')
        extension = '.bin'
    return extension


def _from_no_to_v0(in_file, out_path, **kwargs):
    derivatives = kwargs.get('derivatives', False)
    example = _load_example_preversion(in_file, derivatives)
    # convert example to dict
    example_dict = {}
    example_dict['key'] = example.name
    example_dict['E'] = example.true_energy
    example_dict['species'] = example.species_vector
    example_dict['Gvect'] = example.gvects

    if derivatives:
        example_dict['dGvect'] = example.dgvects
        example_dict['forces'] = example.forces
    binary_encoder(example_dict, out_path)


def _remove_already_converted_files(in_file, out_file):
    already_converted = set(out_file)
    not_converted = set(in_file)
    logger.info('files to convert %d', len(not_converted))
    logger.info('already computed files %d', len(already_converted))
    return list(not_converted - already_converted)


def _search_file_w_extension(folder, extension):
    files_name = []
    for file_name in os.listdir(folder):
        if extension:
            files_name.append(file_name[:-len(extension)])
        else:
            files_name.append(file_name)
    return files_name


def _from_no_to_v0_sparse(in_file, out_path, **kwargs):
    derivatives = True
    example = _load_example_preversion(in_file, derivatives)
    # convert example to dict
    example_dict = {}
    example_dict['key'] = example.name
    example_dict['E'] = example.true_energy
    example_dict['species'] = example.species_vector
    example_dict['Gvect'] = example.gvects

    dg_vector = example.dgvects
    example_dict['forces'] = example.forces

    n_atoms = example.n_atoms
    g_size = example.g_size
    dg_vector = dg_vector.reshape([n_atoms, g_size, n_atoms, 3])
    idxes = np.where(dg_vector != 0.0)
    values = dg_vector[idxes]
    i, j, k, direction = idxes
    idx_1 = i * g_size + j
    idx_2 = 3 * k + direction
    dgdx_indices = np.vstack([idx_1, idx_2]).T
    example_dict['dGvect_val'] = values
    example_dict['dGvect_ind'] = dgdx_indices
    binary_encoder(example_dict, out_path)


def _from_v0_to_v0_sparse(in_file, out_path, **kwargs):
    example = load_example(in_file)
    # convert example to dict
    example_dict = {}
    example_dict['key'] = example.name
    example_dict['E'] = example.true_energy
    example_dict['species'] = example.species_vector
    example_dict['Gvect'] = example.gvects

    dg_vector = example.dgvects
    example_dict['forces'] = example.forces
    if 'per_atom_quantity' in kwargs:
        if kwargs['per_atom_quantity']:
            example_dict['per_atom_quantity'] = example.per_atom_quantity

    n_atoms = example.n_atoms
    g_size = example.g_size
    dg_vector = dg_vector.reshape([n_atoms, g_size, n_atoms, 3])
    idxes = np.where(dg_vector != 0.0)
    values = dg_vector[idxes]
    i, j, k, direction = idxes
    idx_1 = i * g_size + j
    idx_2 = 3 * k + direction
    dgdx_indices = np.vstack([idx_1, idx_2]).T
    example_dict['dGvect_val'] = values
    example_dict['dGvect_ind'] = dgdx_indices
    binary_encoder(example_dict, out_path)


def starmap_with_kwargs(pool, fn, args_iter, kwargs):
    args_for_starmap = zip(repeat(fn), args_iter, repeat(kwargs))
    return pool.starmap(_apply_args_and_kwargs, args_for_starmap)


def _apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def main(indir, outdir, number_of_processes=2):
    # create outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # load old files
    logger.info('searching old files')
    extension = _extension_chooser(STARTING_VERSION)
    input_files_name = _search_file_w_extension(indir, extension)

    logger.info('searching already computed files')
    final_extension = _extension_chooser(END_VERSION)
    out_files_name = _search_file_w_extension(outdir, final_extension)

    input_files_name = _remove_already_converted_files(input_files_name,
                                                       out_files_name)
    logger.info('remaining files to convert %d', len(input_files_name))

    if STARTING_VERSION == 'no_version' and END_VERSION == 'v0':
        converting_function = _from_no_to_v0
    elif STARTING_VERSION == 'no_version' and END_VERSION == 'v0_sparse':
        converting_function = _from_no_to_v0_sparse
    elif STARTING_VERSION == 'v0' and END_VERSION == 'v0_sparse':
        converting_function = _from_v0_to_v0_sparse
    else:
        logger.warning('conversion not implemented')
        exit(1)

    pool = mp.Pool(number_of_processes)
    logger.info('staring conversion')
    loc = 0

    while True:
        parallel_batch = [
            (os.path.join(indir, file_name) + extension, outdir)
            for file_name in input_files_name[loc:loc + number_of_processes]
        ]
        if not parallel_batch:
            break
        logger.info('computing elements %d - %d', loc,
                    loc + number_of_processes)
        starmap_with_kwargs(pool, converting_function, parallel_batch,
                            CONVERSION_PARAMETERS)
        loc += number_of_processes

    logger.info('end conversion')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='update bin example files')
    PARSER.add_argument('-i',
                        '--indir',
                        type=str,
                        help='in path',
                        required=True)
    PARSER.add_argument('-o',
                        '--outdir',
                        type=str,
                        help='out path',
                        required=True)
    PARSER.add_argument('-p',
                        '--parallel',
                        type=int,
                        help='number of processes',
                        default=2,
                        required=False)

    ARGS = PARSER.parse_args()
    if ARGS.indir and ARGS.outdir:
        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s - %(message)s')
        emit_splash_screen(logger)
        main(indir=ARGS.indir,
             outdir=ARGS.outdir,
             number_of_processes=ARGS.parallel)
