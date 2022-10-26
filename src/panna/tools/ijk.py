###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import argparse
import itertools
import logging
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

# PANNA interfaces
# add PANNA folder to PYTHONPATH variable to enable these
from gvector.pbc import replicas_max_idx

from lib import ExampleJsonWrapper
from lib import init_logging

# logger
logger = logging.getLogger('panna.tools')



def main(args):
    examples = os.listdir(args.source)
    atomic_sequence = args.atomic_sequence.split(',')
    species_str_2_idx = dict(zip(atomic_sequence, range(len(atomic_sequence))))
    n_species = len(atomic_sequence)

    if args.number_of_elements > 0:
        examples = examples[:args.number_of_elements]

    logger.info('for now PBC are guessed based on the example file only')

    n_graphs = int(n_species * (n_species + 1) / 2)
    containers = [[np.empty(0), np.empty(0)] for x in range(n_graphs)]

    for eg in examples:
        example = ExampleJsonWrapper(
            os.path.join(args.source, eg), species_str_2_idx)
        positions = example.positions
        n_atoms = example.number_of_atoms
        box = example.lattice_vectors
        species = example.species_indexes
        max_indices = replicas_max_idx(box, args.r_cut)
        l_max, m_max, n_max = max_indices
        l_list = range(-l_max, l_max + 1)
        m_list = range(-m_max, m_max + 1)
        n_list = range(-n_max, n_max + 1)
        replicas = np.asarray(list(itertools.product(l_list, m_list, n_list)))
        replicas = replicas @ box
        n_replicas = len(replicas)
        # creating a tensor with the coordinate of all the needed atoms
        # and reshape it as positions tensor
        # shape: n_atoms, n_replicas, 3
        positions_extended = positions[:, np.newaxis, :] + replicas
        positions_extended = positions_extended.reshape(
            n_atoms * n_replicas, 3)
        # creating the equivalent species tensor
        # the order is [...all atom 1 replicas.....,... all atom2 replicas....,]
        species = np.tile(species[:, np.newaxis],
                          (1, n_replicas)).reshape(n_atoms * n_replicas)
        # computing x_i - x_j between all the atom in the unit cell and all
        # the atom in the cell + replica tensor
        # shape: n_atoms, n_replicas, 3
        deltas = positions[:, np.newaxis, :] - positions_extended
        # using the deltas to compute rij for each atom in the unit cell and
        # all the atom in the cell + replica tensor
        # shape: n_atoms, n_replicas
        rij = np.linalg.norm(deltas, axis=-1)
        # mask to remove replicas outside required cutoff
        # shape: n_atoms, n_replicas
        angular_mask = np.logical_and(rij < args.r_cut, rij > 1e-8)

        for idx in range(n_atoms):
            # shape: atoms_inside_cutoff, 3
            cutoff_deltas = deltas[idx, angular_mask[idx]]
            # shape: atoms_inside_cutoff
            cutoff_species = species[angular_mask[idx]]
            cutoff_rij = rij[idx, angular_mask[idx]]

            container_idx = 0
            for atom_kind_1 in range(n_species):
                for atom_kind_2 in range(atom_kind_1, n_species):
                    # for each atom_kind we take the idxs of all atoms of that
                    # kind inside the cutoff, we do this for both the species
                    species_idxs_1 = np.where(cutoff_species == atom_kind_1)
                    species_idxs_2 = np.where(cutoff_species == atom_kind_2)
                    # in the same way we extract all the quanities needed to
                    # compute the angular contribution
                    cutoff_deltas_ij = cutoff_deltas[species_idxs_1]
                    cutoff_deltas_ik = cutoff_deltas[species_idxs_2]
                    cutoff_r_ij = cutoff_rij[species_idxs_1]
                    cutoff_r_ik = cutoff_rij[species_idxs_2]
                    # = computation of the angle between ikj triplet =
                    # numerator: ij dot ik
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    a = np.sum(
                        cutoff_deltas_ij[:, np.newaxis, :] * cutoff_deltas_ik,
                        2)
                    # denominator: ij * ik
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    b = cutoff_r_ij[:, np.newaxis] * cutoff_r_ik
                    # element by element ratio
                    cos_theta_ijk = a / b
                    # correct numerical error
                    cos_theta_ijk[cos_theta_ijk >= 1.0] = 1.0
                    cos_theta_ijk[cos_theta_ijk <= -1.0] = -1.0
                    # compute the angle
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    theta_ijk = np.arccos(cos_theta_ijk)
                    # compute the avg distance
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    r_ijk = (cutoff_r_ij[:, np.newaxis] +
                             cutoff_r_ik[np.newaxis, :]) / 2
                    # creation of a mask to esclude counting of j==k
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    f = np.logical_or(
                        np.abs(cutoff_r_ij[:, np.newaxis] - cutoff_r_ik) >
                        1e-5, cos_theta_ijk < .99999)
                    # mask application
                    # shape: n_of_triplet 1x2,
                    #        radial_ang_sampling, ang_sampling
                    theta_ijk = theta_ijk[f]
                    r_ijk = r_ijk[f]
                    containers[container_idx][0] = np.append(
                        containers[container_idx][0], r_ijk)
                    containers[container_idx][1] = np.append(
                        containers[container_idx][1], theta_ijk)
                    container_idx += 1
    if args.dump:
        with open('ijk.dump', 'w') as f:
            idx = 0
            for atom_kind_1 in range(n_species):
                for atom_kind_2 in range(atom_kind_1, n_species):
                    f.write('# {}x{} points\n'.format(
                        atomic_sequence[atom_kind_1],
                        atomic_sequence[atom_kind_2]))
                    for r, t in zip(containers[idx][0], containers[idx][1]):
                        f.write('{} {}\n'.format(r, t))
                    idx += 1

    # Plot
    fig, axes = plt.subplots(n_graphs, sharex=False, sharey=False)

    if n_graphs == 1:
        axes = [axes]

    for idx, container in enumerate(containers):
        logger.info('container {} number of elements {}'.format(
            idx, len(container[0])))

    plt_idx = 0
    for atom_kind_1 in range(n_species):
        for atom_kind_2 in range(atom_kind_1, n_species):
            axes[plt_idx].set_title('{}x{} distribution'.format(
                atomic_sequence[atom_kind_1], atomic_sequence[atom_kind_2]))
            r, theta = containers[plt_idx]
            im = axes[plt_idx].hexbin(
                r,
                theta,
                gridsize=args.bins,
                cmap=cm.jet,
                bins='log' if args.log else None)
            axes[plt_idx].set_ylim(0, 3.14)
            axes[plt_idx].set_xlabel('(rij +  rik)/2')
            axes[plt_idx].set_ylabel('theta')
            fig.colorbar(im, ax=axes[plt_idx])
            plt_idx += 1

    plt.show()


if __name__ == '__main__':
    init_logging()
    parser = argparse.ArgumentParser(description='ijk distribution plotter')
    parser.add_argument(
        '-s',
        '--source',
        type=str,
        help='source folder where example files are located',
        required=True)
    parser.add_argument(
        '-r',
        '--r_cut',
        type=float,
        help='radial cutoff in Angstrom',
        required=True)
    parser.add_argument(
        '-a',
        '--atomic_sequence',
        type=str,
        help='atomic sequence, comma separated',
        required=True)
    parser.add_argument(
        '-b',
        '--bins',
        type=int,
        default=100,
        help='number of bins on radial axis, '
        'angular axis will be generated '
        'accordingly. optional, default: 100',
        required=False)
    parser.add_argument(
        '-n',
        '--number_of_elements',
        type=int,
        default=-1,
        help='number of elements to use to create the graph '
        'useful if the data set is too big. '
        'Optional, default: all the elements',
        required=False)
    parser.add_argument(
        "--log",
        action='store_true',
        help='put the color scale of the heat map in log scale')
    parser.add_argument(
        "--dump",
        action='store_true',
        help='dump the data used to generate the heat '
        'map in to a readable format')
    args = parser.parse_args()
    main(args)
