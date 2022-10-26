###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
""" Distance calculator between fingerprints
"""
import argparse

import json
import os
import itertools
import logging

# import multiprocessing as mp

import numpy as np

from lib.log import init_logging
logger = logging.getLogger('panna.tools')


class PartialFingerprintNotFound(ValueError):
    pass


class PartialWeightNotFound(ValueError):
    pass


def cosine_weighted_distance(fingerprint_1, fingerprint_2, weights):
    """ distance between fingerprints

    Fingerprint distances are computed as defined in 10.1063/1.3079326

    Parameters
    ----------
    fingerprint_1: numpy_array (fingerprint_size)
    fingerprint_2: numpy_array (fingerprint_size)
                   This routine can also handle (?, fingerprint_size)
                   where ? is the number of fingerprint we want to
                   compare fingerprint 1 with.
    weights: numpy_array (n_species, n_species)

    Returns
    -------
    numpy_array (1) or (?)
          Where ? is the number of fingerprints passed in fingerprint_2
          if only one fingerprint is passed ()

    """
    den_1 = np.sqrt(np.sum(np.sum(fingerprint_1**2, axis=-1) * weights))
    den_2 = np.sqrt(
        np.sum(np.sum(fingerprint_2**2, axis=-1) * weights, axis=-1))

    numerator = np.sum(
        np.sum(fingerprint_1 * fingerprint_2, axis=-1) * weights, axis=-1)
    distances = .5 * (1 - numerator / (den_1 * den_2))
    return distances.reshape(-1)


def euclid_weighted_distance(fingerprint_1, fingerprint_2, weights):
    """ distance between fingerprints

    Fingerprint distances are computed as defined in 10.1063/1.3079326

    Parameters
    ----------
    fingerprint_1: numpy_array (fingerprint_size)
    fingerprint_2: numpy_array (fingerprint_size)
                   This routine can also handle (?, fingerprint_size)
                   where ? is the number of fingerprint we want to
                   compare fingerprint 1 with.
    weights: numpy_array (n_species, n_species)

    Returns
    -------
    numpy_array (1) or (?)
          Where ? is the number of fingerprints passed in fingerprint_2
          if only one fingerprint is passed ()
    """
    den_1 = np.sqrt(np.sum(np.sum(fingerprint_1**2, axis=-1) * weights))
    den_2 = np.sqrt(
        np.sum(np.sum(fingerprint_2**2, axis=-1) * weights, axis=-1))

    numerator = np.sum(
        np.sum((fingerprint_1 - fingerprint_2)**2, axis=-1) * weights, axis=-1)

    if (numerator < 0).any():
        tmp = numerator[numerator < 0]
        logger.info('%f %f %f', tmp, den_1, den_2[numerator < 0])

    logger.debug('distance call')

    distances = numerator / (den_1**2 + den_2**2 + 2 * den_1 * den_2)
    # the distance is of shape (?, 1) or (1) so we need to reshape it
    return distances.reshape(-1)


def symmetric_solver(fingerprint, key):
    """Dictionary symmetric extractor

    Solve the problem of symmetric fingerprints, F_ab = F_ba
    Same with the weights.

    Parameters
    ----------
    fingerprint: fingerprint dictionary as in PANNA
    key: tuple of 2 string
         the two string must be two atomic species

    Returns
    -------
    numpy_array (fingerprint_size), numpy_array(1)
          First is the partial fingerprint of the two atomic species
          Second is the partial weights of the two atomic species

    Raise
    -----
    PartialFingerprintNotFound:
        if the partial fingerprint has not been found
    PartialWeightNotFound:
        if the partial weight has not been found
    """
    part_fingerprint = fingerprint.get(''.join(key),
                                       fingerprint.get(''.join(key[::-1])))
    part_weights = fingerprint.get('w' + ''.join(key),
                                   fingerprint.get('w' + ''.join(key[::-1])))
    if not part_fingerprint:
        raise PartialFingerprintNotFound('{} not found'.format(key))
    if not part_weights:
        raise PartialWeightNotFound('{} not found'.format(key))
    return part_fingerprint, part_weights


def main(indir, outdir, nproc, species_sequence, euclidean):
    """
    """

    logger.info('Input dir %s', indir)
    logger.info('Output dir requested %s', outdir)
    logger.warning('NOT IMPLEMENTED, Number of parallel processes %s', nproc)

    # processors_pool = mp.Pool(nproc)

    if os.path.isdir(outdir):
        outdir = os.path.abspath(outdir)
        logger.info('Output dir exists: %s', outdir)
    else:
        os.makedirs(outdir)
        outdir = os.path.abspath(outdir)
        logger.info('Output dir created: %s', outdir)

    examples = [x for x in os.listdir(indir) if x[-6:] == 'fprint']

    species_product = list(
        itertools.product(species_sequence.split(','), repeat=2))
    number_of_species = len(species_sequence.split(','))

    fingerprints = []
    name_energy_volume = []
    weights = np.asarray([False])

    for file in examples:
        with open(os.path.join(indir, file)) as file_stream:
            data = json.load(file_stream)

        n_atoms = int(data['number_atoms'])
        e_per_atom = float(data['energy']) / n_atoms
        v_per_atom = float(data['vol']) / n_atoms
        name_energy_volume.append('{} {} {} {} {}'.format(
            file, data['energy'], data['vol'], e_per_atom, v_per_atom))
        fingerprint = []
        tmp_weights = []
        for key in species_product:
            part_fingerprint, part_weights = symmetric_solver(data, key)
            fingerprint.append(part_fingerprint)
            tmp_weights.append(part_weights)
        fingerprint_len = len(part_fingerprint)
        fingerprint = np.asarray(fingerprint).reshape(
            number_of_species, number_of_species, fingerprint_len)
        fingerprints.append(fingerprint)
        tmp_weights = np.asarray(tmp_weights).reshape(number_of_species,
                                                      number_of_species)
        if weights.all():
            if not np.equal(weights, tmp_weights).all():
                logger.error('error in weights, not comparable')
                break
        else:
            weights = tmp_weights
    fingerprints = np.asarray(fingerprints)

    with open(
            os.path.join(
                outdir, 'name_energy_volume_'
                'energyperatom_volumeperatom.dat'), 'w') as file_stream:
        file_stream.write('\n'.join(name_energy_volume))

    number_of_elemts_off_diagonals = int(
        len(fingerprints) * (len(fingerprints) - 1) / 2)
    distances = np.zeros(number_of_elemts_off_diagonals)

    start_idx = 0
    end_idx = len(fingerprints) - 1

    if euclidean:
        distance = euclid_weighted_distance
    else:
        distance = cosine_weighted_distance

    for idx in range(len(fingerprints) - 1):
        distances[start_idx:end_idx] = distance(
            fingerprints[idx], fingerprints[idx + 1:], weights)
        start_idx = end_idx
        end_idx = start_idx + len(fingerprints) - idx - 2

    np.save(os.path.join(outdir, 'distance_matrix.npy'), distances)


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser(
        description='calculate distances between Fingerprints')
    PARSER.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)
    PARSER.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
    PARSER.add_argument(
        '-s',
        '--species_sequence',
        type=str,
        help='comma separated species',
        required=True)
    PARSER.add_argument(
        '--nproc', type=int, help='omp_num_threads', required=False, default=1)
    PARSER.add_argument(
        '--euclidean',
        help='produce per atom name_energy_volume',
        action='store_true')

    CLI_ARGS = PARSER.parse_args()

    main(
        indir=CLI_ARGS.indir,
        outdir=CLI_ARGS.outdir,
        nproc=CLI_ARGS.nproc,
        species_sequence=CLI_ARGS.species_sequence,
        euclidean=CLI_ARGS.euclidean)
