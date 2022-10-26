###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import os
import logging
import numpy as np
import multiprocessing as mp
from datetime import datetime

# logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(levelname)s - %(message)s')

# console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def cluster_distance_matrix(distance_matrix, cutoff):
    """ Cluster a triu matrix.

    Parameters
    ----------
    distance_matrix: list of numpy_array
        given a square matrix each element of the list is the triangular
        upper part of the given row, diagonal elements are excluded.
        example of a 5 x 5 matrix:
           [x(1 2 3 4 ),
            x x(5 6 7 ),
            x x x(8 9 ),
            x x x x(10),
            x x x x x]
    cutoff: float
        cutoff of the computation.

    Returns
    -------
    numpy_array (matrix side):
        for each element return it's cluster with the given cutoff,
        cluster indexes are randomly sorted
    """

    # cluster every element with itself
    cluster = np.arange(0, len(distance_matrix) + 1, dtype=np.int32)

    for idx, element in enumerate(distance_matrix):
        # find all the elements within the cutoff
        local_cluster = element < cutoff
        # find all the labels within cutoff
        labels_within_local_cluster = cluster[idx + 1:][local_cluster]
        # extend the labels adding the central element label (it is missing due
        # due to the absence of diagonal matrix elements)
        # NOTE: the symmetry of the matrix is exploited here because points
        # that the current central point does not see, because of triangular
        # matrix, have already cycled on and already grouped you with them
        labels_within_local_cluster = np.append(labels_within_local_cluster,
                                                cluster[idx])
        # define a label for the local environment,
        # by convention the min of the founded labels
        new_label = labels_within_local_cluster.min()
        # relabel all the cluster that extends within the local cluster with
        # the new label
        for label in labels_within_local_cluster:
            cluster[cluster == label] = new_label
    # the last element can either be isolated, so it has a cluster for
    # itself, than no operation is needed or be within someone else cluster
    # in that case it's index has already been updated
    return cluster


def cluster_different_cutoffs(distance_matrix,
                              discrete_interval,
                              pool_size=1,
                              stream_1=None,
                              stream_2=None,
                              output=True,
                              interval_minimum=None,
                              interval_maximum=None):
    """ Cluster a triu matrix with different cutoffs

    Parameters
    ----------
    distance_matrix: list of numpy_array
        given a square matrix each element of the list is the triangular
        upper part of the given row, diagonal elements are excluded.
        example of a 5 x 5 matrix:
           [x(1 2 3 4 ),
            x x(5 6 7 ),
            x x x(8 9 ),
            x x x x(10),
            x x x x x]
    discrete_interval: float
        interval used to discretize the connectivity distance (cutoff),
        the probed connectivity distance will go from min(distance_matrix)
        to max(distance_matrix) at step of discrete_interval
    pool_size: integer, optional
        number of process for the parallelization (each process compute
        one interval)
    stream_1: file stream, optional
        a file where to write the cluster result, one line will have an
        integer for each element we are clusterizing, same integer means same
        cluster. Each line will correspond to a different interval
        If not provided data are not dumped
    stream_2: file stream, optional
        a file where to write the clustering stats, one line will have
        the probed interval and the number of clusters.
        If not provided data are not dumped
    output: Boolean, optional
        if False the routine dose not produce any output, useful for very
        big data set where the machine does not have enough ram to store it
    interval_minimum: float
        if provide fix the minimum value that will be used as cutoff
        in clustering, default: minimum distance between fingerprints
    interval_maximum: float
        if provide fix the maximum value that will be used as cutoff
        in clustering, default: maximum distance between fingerprints
    Returns
    -------
    list of numpy_arrays: (matrix side)
        for each connectivity distance value and for each element return
        it's cluster
    numpy_array: (number of interval)
        probed intervals
    None:
        if output is false
    """

    if interval_minimum is None:
        dm_min = np.min([row.min() for row in distance_matrix])
    else:
        dm_min = interval_minimum
    if interval_maximum is None:
        dm_max = np.max([row.max() for row in distance_matrix])
    else:
        dm_max = interval_maximum

    probing_range = np.arange(dm_min, dm_max, discrete_interval)\
        + .5 * discrete_interval
    logger.info('probed space stats: Min: %3.3e, Max: %3.3e, samples: %d',
                dm_min, dm_max, len(probing_range))
    clusters = []
    pool = mp.Pool(pool_size)
    number_of_batches = int(np.ceil(len(probing_range) / pool_size))
    for batch_idx in range(number_of_batches):
        deltas = probing_range[pool_size * batch_idx:pool_size *
                               (batch_idx + 1)]
        logger.info('%s start processing batch %d of %d',
                    datetime.now().strftime('%H:%M:%S'), batch_idx + 1,
                    number_of_batches)
        tmp_clusters = pool.starmap(cluster_distance_matrix,
                                    [(distance_matrix, delta)
                                     for delta in deltas])
        for delta, cluster in zip(deltas, tmp_clusters):
            if stream_1:
                stream_1.write('{:6.3e} {}\n'.format(delta,
                                                     len(np.unique(cluster))))
            if stream_2:
                stream_2.write(' '.join(['{:d}'.format(x)
                                         for x in cluster]) + '\n')
            if output:
                clusters.append(cluster)
            else:
                del cluster

        if stream_1:
            stream_1.flush()
        if stream_2:
            stream_2.flush()

        del tmp_clusters
    if output:
        return clusters, probing_range
    return None


def read_name_energy_volume(name_energy_volume_file, per_atom=False):
    """ read the name energy volume file,
    a simple csv space separated with 3 columns, name, energy, volume

    Parameters
    ----------
    name_energy_volume_file: string
        filename
    per_atom: flag to load per atom quantities

    Return
    ------
    3 numpy array, (number of finger prints)
        name, energy, volume
    """

    names = []
    energies = []
    volumes = []
    with open(name_energy_volume_file, 'r') as file_stream:
        for line in file_stream:
            name, energy, volume, energy_per_atom, \
                volume_per_atom = line.split(' ')
            names.append(name)
            energies.append(energy_per_atom if per_atom else energy)
            volumes.append(volume_per_atom if per_atom else volume)

    return np.asarray(names), np.asarray(
        energies, dtype=np.float), np.asarray(
            volumes, dtype=np.float)


def read_distance_matrix(distance_file,
                         number_of_fingerprint,
                         energies_correction=None,
                         alpha=0):
    """ read a distance matrix, if requested correct it with the energy

    This call DOESN'T return a matrix. If alpha and energy_file are passed
    the distances are corrected with the energy.

    Parameters
    ----------
    distance_file: string
        file name with the distances, binary, one row for each matrix row
        only the triangular upper part of the matrix is be stored. Diagonal
        elements are assumed to be 0.
    number_of_fingerprint: integer
        number of fingerprint, this is needed to reconstruct the matrix
    energies_correction: numpy_array (number of finger prints), optional
        vector with energies
    alpha: float, optional
        energy multiplier to correct distance with energy.

    Return
    ------
    list of numpy_array
        each element of the list is the triangular upper part of the
        given row, diagonal elements are excluded. Example of a
        5 x 5 matrix:
           [x(1 2 3 4 ),
            x x(5 6 7 ),
            x x x(8 9 ),
            x x x x(10),
            x x x x x]
    """
    distance_matrix = []
    distance_vector = np.load(distance_file)

    start_idx = 0
    end_idx = number_of_fingerprint - 1

    for idx in range(number_of_fingerprint - 1):
        row = distance_vector[start_idx:end_idx]
        if alpha > 0:
            d_2 = row**2
            e_2 = (energies_correction[idx] -
                   energies_correction[idx + 1:])**2 * alpha
            row = np.sqrt(d_2 + e_2)

        distance_matrix.append(row)
        start_idx = end_idx
        end_idx = start_idx + number_of_fingerprint - idx - 2
    return distance_matrix


def main(input_directory,
         output_directory,
         discrete_interval,
         alpha,
         pool_size=1,
         interval_minimum=None,
         interval_maximum=None,
         per_atom=False):
    """Cluster a distance matrix.

    Load distance matrix (binary format, only triangular upper)
    Clusetrize it
    Save resulting vector to file

    Parameters
    ----------
    input_directory: str
        input directory
    output_direcotry: str
        output directory
    discrete_interval: float
        spacing between different range of clustering (??)
    alpha: float
        energy multiplier when computing the augmented distance
    pool_size: integer
        number of process
    interval_minimum: float
        if provide fix the minimum value that will be used as cutoff
        in clustering
    interval_maximum: float
        if provide fix the maximum value that will be used as cutoff
        in clustering
    per_atom: bool
        load per atom energy and volume file

    Returns
    -------
    None
       Save a file in output directory
    """
    if os.path.isdir(output_directory):
        output_directory = os.path.abspath(output_directory)
        logger.info('Output dir exists: %s', output_directory)
    else:
        os.makedirs(output_directory)
        output_directory = os.path.abspath(output_directory)
        logger.info('Output dir created: %s', output_directory)

    distance_matrix_file = os.path.join(input_directory, 'distance_matrix.npy')
    name_energy_volume_file = os.path.join(
        input_directory, 'name_energy_volume_'
        'energyperatom_volumeperatom.dat')
    names, energies, volumes = read_name_energy_volume(name_energy_volume_file,
                                                       per_atom)

    number_of_fingerprints = len(names)
    logger.info('Number of fingerprints: %d', number_of_fingerprints)

    distance_matrix = read_distance_matrix(
        distance_matrix_file, number_of_fingerprints, energies, alpha)

    if interval_minimum is None:
        interval_minimum = np.min([row.min() for row in distance_matrix])
    if interval_maximum is None:
        interval_maximum = np.max([row.max() for row in distance_matrix])

    file_stream_1 = open(
        os.path.join(
            output_directory,
            'probed_interval_{:2.2e}_{:2.2e}_dint_{}_alpha_{}.dat'.format(
                interval_minimum, interval_maximum, discrete_interval, alpha)),
        'w')
    file_stream_2 = open(
        os.path.join(
            output_directory,
            'clusters_interval_{:2.2e}_{:2.2e}_dint_{}_alpha_{}.dat'.format(
                interval_minimum, interval_maximum, discrete_interval, alpha)),
        'w')

    file_stream_1.write('#interval unique_clusters\n')

    cluster_different_cutoffs(distance_matrix, discrete_interval, pool_size,
                              file_stream_1, file_stream_2, False,
                              interval_minimum, interval_maximum)

    file_stream_1.close()
    file_stream_2.close()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='calc distances')
    PARSER.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)

    PARSER.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)

    PARSER.add_argument(
        '-dint',
        '--dinterval',
        type=float,
        help='distance threshold interval',
        required=True)

    PARSER.add_argument(
        '-min', '--dmin', type=float, help='distance minimum', required=False)

    PARSER.add_argument(
        '-max', '--dmax', type=float, help='distance maximum', required=False)

    PARSER.add_argument(
        '--nproc',
        type=int,
        help='omp number of threads',
        required=False,
        default=1)

    PARSER.add_argument(
        '-alp',
        '--alpha',
        type=float,
        help='energy multiplier',
        required=False,
        default=0)

    PARSER.add_argument(
        '--per_atom',
        help='load per_atom energy and volume',
        action='store_true')

    ARGS = PARSER.parse_args()
    main(
        input_directory=ARGS.indir,
        output_directory=ARGS.outdir,
        discrete_interval=ARGS.dinterval,
        alpha=ARGS.alpha,
        pool_size=ARGS.nproc,
        interval_minimum=ARGS.dmin,
        interval_maximum=ARGS.dmax,
        per_atom=ARGS.per_atom)
