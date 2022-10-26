###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
## CALCULATE FINGERPRINTS
## A LA USPEX
## -i -o --nproc --dimension

import argparse
import itertools
import json
import multiprocessing as mp
import os
import logging

import random
from functools import partial

import numpy as np
from scipy.special import erf

# PANNA interfaces
# add PANNA folder to PYTHONPATH variable to enable these
# SEE README.md
from gvector.pbc import replicas_max_idx
from lib.log import init_logging

logger = logging.getLogger('panna.tools')


def fingprint(Rmax, delta, sigma, outdir, d, filename):
    '''This module computes a fingerprint array for a given configuration in panna json
       e.g: for a system with H and C: {"HH":[....],"wHH": real-number, "HC":[....].. etc}

       Rmax: cutoff
       delta: descritization step / bin interval
       sigma: gaussian width
       outdir: output folder where fingerprint is written
       d: dimensions (3, regular 2 surface material)
       filename is the input json (currectly in .example extension)
    '''
    with open(filename) as file_stream:
        example = json.load(file_stream)

    
    unit_of_length = example.get('unit_of_length', '')
    if unit_of_length in ['bohr', 'au', 'Bohr']:
        A2unit = 1.0 / 0.529177
    elif unit_of_length in ['A', 'Ang', 'ang', 'angstrom', 'Angstrom']:
        A2unit = 1.0
    else:
        logger.info('unit of length not recognized, assumed Angstrom')
        A2unit = 1.0

    Rmax = Rmax * A2unit
    delta = delta * A2unit
    sigma = sigma * A2unit

    energy = float(example['energy'][0])
    if example['energy'][1] in ['eV', 'ev', 'EV']:
        unit2eV = 1.0
    elif example['energy'][1] in ['Ry', 'rydberg', 'Rydberg', 'Ryd']:
        unit2eV = 13.605698066

    outfile = filename.split('/')[-1].split('.example')[0] + ".fprint"

    lattice_vectors = np.asarray(example['lattice_vectors']).astype(float)
    volume = np.abs(
        np.dot(lattice_vectors[0],
               np.cross(lattice_vectors[1], lattice_vectors[2])))

    if d == 2:
        volume = 1.0  # vol is not important for 2d structures, dont use it in the definition of FPs

    atomic_position_unit = example['atomic_position_unit']

    pos_vector = []
    type_vector = []

    # recover atomic info from file
    for atom in example['atoms']:
        try:
            idx, symbol, position, *dummy = atom
        except ValueError:
            idx, symbol, position = atom

        if atomic_position_unit == "crystal":
            pos_vector.append(
                np.dot(
                    np.transpose(lattice_vectors),
                    np.asarray(position).astype(float)))
        else:
            pos_vector.append(np.asarray(position).astype(float))

        type_vector.append(symbol)

    pos_vector = np.asarray(pos_vector)
    print(type_vector)
    all_symbols = list(set(type_vector))
    _symbol_map = {char: idx for idx, char in enumerate(all_symbols)}

    # convert type vector to int vector with internal mapping
    type_vector = np.asarray([_symbol_map[x] for x in type_vector],
                             dtype=np.int)

    n_atoms = len(type_vector)
    n_species = len(all_symbols)

    # compute total weight
    weight_ab = 0
    unique, counts = np.unique(type_vector, return_counts=True)
    unique_counts = dict(zip(unique, counts))

    for a in range(n_species):
        n_atoms_type_a = unique_counts[a]
        for b in range(a, n_species):
            n_atoms_type_b = unique_counts[b]
            weight_ab += n_atoms_type_a * n_atoms_type_b
    #################################

    radial_sampling_vector_bottom = np.arange(0, Rmax, delta)
    radial_sampling_vector_top = np.arange(0, Rmax, delta) + delta

    # creating all the displacement vectors
    max_indices = replicas_max_idx(lattice_vectors, Rmax)
    l_max, m_max, n_max = max_indices
    if d == 2:
        n_max = 0  # uspex generated 2d files have always z for their aperiodic axis.
        # and the structure always fits in single cell
    l_list = range(-l_max, l_max + 1)
    m_list = range(-m_max, m_max + 1)
    n_list = range(-n_max, n_max + 1)

    # number of replicas, 3
    replicas_idxs = np.asarray(list(itertools.product(l_list, m_list, n_list)))
    n_replicas = len(replicas_idxs)
    # the lattice matrix should be [a1, a2, a3]
    # nubmer of replicas, 3
    # replicas_displeacemnt_vecotrs = np.einsum('ij,jk->ik', replicas_idxs,
    #                                           lattice_vectors)
    replicas_displeacemnt_vecotrs = replicas_idxs @ lattice_vectors
    # number of replicas * number of atoms, 3
    extended_pos_vector = (
        pos_vector[:, np.newaxis, :] + replicas_displeacemnt_vecotrs).reshape(
            n_atoms * n_replicas, 3)
    extended_type_vector = np.tile(type_vector[:, np.newaxis],
                                   [1, n_replicas]).flatten()

    # distance matrix

    extended_deltas = pos_vector[:, np.newaxis, :] - extended_pos_vector
    extended_distances = np.linalg.norm(extended_deltas, axis=-1)

    fingerprints_dict = {}

    for specie_a in range(n_species):
        row_index = np.where(type_vector == specie_a)[0]
        n_atoms_a = len(row_index)
        for specie_b in range(specie_a, n_species):
            column_index = np.where(extended_type_vector == specie_b)[0]
            submatrix = extended_distances[row_index[:, None], column_index]
            radius_mask = np.logical_and(submatrix < Rmax, submatrix > 1e-5)
            submatrix = submatrix[radius_mask]

            n_atoms_b = len(np.where(type_vector == specie_b)[0])

            submatrix_bottom_sampling = radial_sampling_vector_bottom -\
                submatrix[:, np.newaxis]
            submatrix_top_sampling = radial_sampling_vector_top -\
                submatrix[:, np.newaxis]
            erf_delta = erf(
                submatrix_top_sampling / (np.sqrt(2) * sigma)) - erf(
                    submatrix_bottom_sampling / (np.sqrt(2) * sigma))

            dbl_sum = np.sum(erf_delta / submatrix[:, np.newaxis]**2, axis=0)

            normalization = .5 * volume / (
                4 * np.pi * n_atoms_a * n_atoms_b * delta)
            fingerprint_ab = -1 + normalization * dbl_sum

            name_string = all_symbols[specie_a] + all_symbols[specie_b]
            fingerprints_dict[name_string] = fingerprint_ab.tolist()
            fingerprints_dict['w' +
                              name_string] = n_atoms_a * n_atoms_b / weight_ab
    fingerprints_dict['energy'] = energy * unit2eV
    fingerprints_dict['vol'] = volume * (1.0 / A2unit)**3
    fingerprints_dict['number_atoms'] = n_atoms

    with open(os.path.join(outdir, outfile), 'w') as fingerprint_outstream:
        json.dump(fingerprints_dict, fingerprint_outstream)
    return


def main(indir, outdir, nproc, d):
    #compute many configuration at a time
    logger.info('Input dir %s', indir)
    logger.info('Output dir requested %s', outdir)
    logger.info('Number of parallel processes %d', nproc)
    logger.info('Structure is assumed %d dimensional', d)
    p = mp.Pool(nproc)
    if os.path.isdir(outdir):
        outdir = os.path.abspath(outdir)
        logger.info('Output dir exists: %s', outdir)
    else:
        os.makedirs(outdir)
        outdir = os.path.abspath(outdir)
        logger.info('Output dir created: %s', outdir)

    fp_done_key = []

    #COMMENT OUT THIS PART IF YOU DONT WANT TO CHECK FOR THE EXAMPLES ALREADY PROCESSED
    for rt, dirs, files in os.walk(outdir):
        for f in files:
            if f.endswith('.fprint') and os.stat(os.path.join(rt,
                                                              f)).st_size != 0:
                fp_done_key.append(f.split('.fprint')[0])
    ###########
    #
    print('the #  json files already done', flush=True)
    print(len(fp_done_key), flush=True)
    #print(fp_done_key[0], flush=True)
    jsonfiles = []
    for rt, dirs, files in os.walk(indir):
        for f in files:
            #print(f)
            if f.endswith('.example') and f.split(
                    '.example')[0] not in fp_done_key:
                #print(f)
                jsonfiles.append(os.path.join(rt, f))

    print('the #  json files to be done {}'.format(len(jsonfiles)), flush=True)
    #print(len(jsonfiles), flush=True)
    if len(jsonfiles) > 0:
        random.shuffle(jsonfiles)
        print('the first one in line {}'.format(jsonfiles[0]), flush=True)
    #jsonfiles=[os.path.join(indir, 'T1200_step1414.example')]
    Rmax = 10.0  #Ang
    delta = 0.08  #Ang
    sigma = 0.03  #Ang
    #sigma = sigma/np.sqrt(2 * np.log(2))
    ###############################################################
    #ff=open(os.path.join(outdir , 'fp_done.dat'), 'a')
    Fingprint = partial(fingprint, Rmax, delta, sigma, outdir, d)
    i = 0
    print(
        'Jsons to be done per proc = {}'.format(int(len(jsonfiles) / nproc)),
        flush=True)
    while i <= int(len(jsonfiles) / nproc):
        ll = i * nproc
        lu = (i + 1) * nproc
        try:
            fi = jsonfiles[ll:lu]
        except IndexError:
            fi = jsonfiles[ll, len(files)]
        data = p.map(Fingprint, fi)
        #ff.write(','.join(map(str,fi)))
        i += 1


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser(description='makes FingerPrints')
    PARSER.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)
    PARSER.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
    PARSER.add_argument(
        '--nproc', type=int, help='omp_num_threads', required=False, default=1)
    PARSER.add_argument(
        '--dimension',
        type=int,
        help='number of dimensions',
        required=False,
        default=3)

    ARGS = PARSER.parse_args()
    logger.info('BEGIN FP CALCULATION')
    main(
        indir=ARGS.indir,
        outdir=ARGS.outdir,
        nproc=ARGS.nproc,
        d=ARGS.dimension)
