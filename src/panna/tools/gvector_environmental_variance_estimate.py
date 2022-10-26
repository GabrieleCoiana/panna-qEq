""" module to compute environmental variances of
examples
"""
import argparse
import logging
import os
import numpy as np

from neuralnet.example import load_example
from lib.distances import cos_distance

from lib import init_logging

logger = logging.getLogger('panna.tools')


def main(indir, outdir, number_of_species, per_atom):
    """
    """

    logger.info('Input dir %s', indir)
    logger.info('Output dir requested %s', outdir)
    logger.info('Number of species %d', number_of_species)

    if os.path.isdir(outdir):
        outdir = os.path.abspath(outdir)
        logger.info('Output dir exists: %s', outdir)
    else:
        os.makedirs(outdir)
        outdir = os.path.abspath(outdir)
        logger.info('Output dir created: %s', outdir)

    examples = [x for x in os.listdir(indir)]

    name_energy_variance = []
    for file in examples:
        path = os.path.join(indir, file)
        example = load_example(path, number_of_species)

        energy = example.true_energy
        if per_atom:
            n_atoms = example.n_atoms
            energy = energy / n_atoms
        gvectors = example.gvects
        number_of_elemts_off_diagonals = int(
            len(gvectors) * (len(gvectors) - 1) / 2)
        distances = np.zeros(number_of_elemts_off_diagonals)
        start_idx = 0
        end_idx = len(gvectors) - 1

        for idx in range(len(gvectors) - 1):
            distances[start_idx:end_idx] = cos_distance(
                gvectors[idx], gvectors[idx + 1:])
            start_idx = end_idx
            end_idx = start_idx + len(gvectors) - idx - 2
        variance = distances.var()
        name_energy_variance.append('{},{},{}'.format(file, energy, variance))

    with open(
            os.path.join(
                outdir, 'name_energy_variance.dat' if not per_atom else
                'name_energy_variance_per_atom.dat'), 'w') as file_stream:
        file_stream.write('name,energy,variance\n')
        file_stream.write('\n'.join(name_energy_variance))


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser(
        description='calculate environmental variances')
    PARSER.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)
    PARSER.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
    PARSER.add_argument(
        '-ns',
        '--number_of_species',
        type=int,
        help='number of species',
        required=True)
    PARSER.add_argument(
        '--per_atom',
        help='produce per atom name_energy_volume',
        action='store_true')

    CLI_ARGS = PARSER.parse_args()

    main(
        indir=CLI_ARGS.indir,
        outdir=CLI_ARGS.outdir,
        number_of_species=CLI_ARGS.number_of_species,
        per_atom=CLI_ARGS.per_atom)
