###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import itertools
import logging

import numpy as np

from .helper_functions import G_angular_BP, G_radial
from .gvect_base import GvectBase, requires_parameters
from .pbc import replicas_max_idx

logger = logging.getLogger(__name__)


class GvectBP(GvectBase):
    """Original G2-G4 Behler Parinello"""

    name = 'BP'
    doi = '10.1103/PhysRevLetter.98.146401'

    _gvect_parameters = {
        # RADIAL_COMPONENTS
        'Rc_rad': lambda self, x: np.array(x) * self.unit2A,
        'Rs_rad': lambda self, x: np.array(x) * self.unit2A,
        'eta_rad': lambda self, x: np.array(x) / (self.unit2A * self.unit2A),
        # ANGULAR_COMPONENTS
        'Rc_ang': lambda self, x: np.array(x) * self.unit2A,
        'eta_ang': lambda self, x: np.array(x) / (self.unit2A * self.unit2A),
        'zeta': lambda self, x: np.array(x),
    }

    def parse_parameters(self, gv_param):
        self.units = gv_param.get('gvect_parameters_unit', 'angstrom')
        # pylint: disable=invalid-name

        # RADIAL_COMPONENTS
        Rc_rad = gv_param.getfloat('Rc_rad')
        self.update_parameter('Rc_rad', Rc_rad)

        Rs0_rad = gv_param.getfloat('Rs0_rad', 0.0)
        RsN_rad = gv_param.getint('RsN_rad', None)
        Rs_rad = gv_param.get_comma_list_floats('Rs_rad_list', [])

        if len(Rs_rad) > 1:
            logger.info('Radial Gaussian centers are set by Rs_rad_list')
            Rs_rad = np.asarray(Rs_rad)
        else:
            logger.info('Radial Gaussian centers are set by Rs0_rad, Rc_rad, RsN_rad')
            Rsst_rad = (Rc_rad - Rs0_rad) / RsN_rad
            Rs_rad = np.arange(Rs0_rad, Rc_rad, Rsst_rad)

        self.update_parameter('Rs_rad', Rs_rad)
        eta_rad = gv_param.get_comma_list_floats('eta_rad')
        self.update_parameter('eta_rad', eta_rad)

        # ANGULAR_COMPONENTS
        self.update_parameter('Rc_ang', gv_param.getfloat('Rc_ang', Rc_rad))

        eta_ang = gv_param.get_comma_list_floats('eta_ang', eta_rad)
        self.update_parameter('eta_ang', eta_ang)

        self.update_parameter('zeta', gv_param.get_comma_list_floats('zeta'))

    @property
    @requires_parameters
    def gsize(self):
        return int(
            len(self.Rs_rad) * len(self.eta_rad) * self.number_of_species
            + len(self.eta_ang) * len(self.zeta) * 2 * self.number_of_species)

    @property
    @requires_parameters
    def gvect(self):
        gvect = {
            # RADIAL_COMPONENTS
            'Rc_rad': self.Rc_rad,
            'Rs_rad': self.Rs_rad,
            'eta_rad': self.eta_rad,
            # ANGULAR_COMPONENTS
            'Rc_ang': self.Rc_ang,
            'eta_ang': self.eta_ang,
            'zeta': self.zeta,
        }
        return gvect

    @requires_parameters
    def __call__(self, key, positions, species, lattice_vectors, **kwargs):
        """ Calculate the gvector

        as defined in 10.1103/PhysRevLetter.98.146401
        """

        # define shorter names:
        n_species = self.number_of_species
        eta_rad = np.asarray(self.eta_rad)
        Rc_rad = self.Rc_rad
        # Rs0_rad = self.Rs0_rad
        # Rsst_rad = self.Rsst_rad
        Rs_rad = self.Rs_rad

        Rc_ang = self.Rc_ang
        # eta_ang = np.asarray(self.eta_ang)
        eta_ang = self.eta_ang
        # zeta = np.asarray(self.zeta)
        zeta = self.zeta
        lamb = np.asanyarray([1.0, -1.0])

        # ensure that everything is a numpy array
        positions = np.asarray(positions)
        species = np.asarray(species)
        lattice_vectors = np.asarray(lattice_vectors)

        if self.compute_dgvect:
            raise ValueError('derivative not implemented')
        n_atoms = len(positions)
        Gs = [np.empty(0) for x in range(n_atoms)]

        # compute how many cell replica we need in each direction
        if 'pbc' in kwargs:
            max_indices = replicas_max_idx(lattice_vectors,
                                           max(Rc_rad, Rc_ang),
                                           pbc=kwargs['pbc'])
        else:
            max_indices = replicas_max_idx(lattice_vectors, max(Rc_rad, Rc_ang))

        l_max, m_max, n_max = max_indices
        l_list = range(-l_max, l_max + 1)
        m_list = range(-m_max, m_max + 1)
        n_list = range(-n_max, n_max + 1)
        # create a matrix with all the idx of the extra cell we need
        replicas = np.asarray(list(itertools.product(l_list, m_list, n_list)))
        # map the index of the cell on to the lattice vectors
        # all needed translation for each atom in the unit cell
        replicas = replicas @ lattice_vectors
        n_replicas = len(replicas)
        # creating a tensor with the coordinate of all the needed atoms
        # and reshape it as positions tensor
        # shape: n_atoms, n_replicas, 3
        positions_extended = positions[:, np.newaxis, :] + replicas
        positions_extended = positions_extended.reshape(n_atoms * n_replicas, 3)
        # creating the equivalent species tensor
        # the order is
        # [...all atom 1 replicas.....,... all atom2 replicas....,]
        species = np.tile(species[:, np.newaxis],
                          (1, n_replicas)).reshape(n_atoms * n_replicas)
        # computing x_i - x_j between all the atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas, 3
        deltas = positions[:, np.newaxis, :] - positions_extended
        # using the deltas to compute rij for each atom
        # in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas
        rij = np.linalg.norm(deltas, axis=-1)

        # === RADIAL PART 1 ===
        # create a boolean mask to extrapolate only atoms inside the cutoff
        # shape: n_atoms, n_replicas
        # each row contains True value if the given j atom is inside the
        # cutoff wrt the i atom, logical "and" to exclude counting the
        # central atom
        radial_mask = np.logical_and(rij < Rc_rad, rij > 1e-8)

        # create tensors of parameters for the sampling to allow proper broadcasting
        # sampling_rad = np.arange(Rs0_rad, Rc_rad, Rsst_rad)
        sampling_rad = Rs_rad
        # ===  END RADIAL PART 1 ===

        # === ANGULAR PART 1 ===
        # same mask as the radial part
        # shape: n_atoms, n_replicas
        angular_mask = np.logical_and(rij < Rc_ang, rij > 1e-8)
        # === END ANGULAR PART 1 ===

        for idx in range(n_atoms):
            # === RADIAL PART 2 ===
            # for each relevant quantity for the radial part we extract only the
            # elemnts inside the cutoff
            # shape: atoms_inside_cutoff
            cutoff_species = species[radial_mask[idx]]
            cutoff_rij = rij[idx, radial_mask[idx]]

            # for each atom inside the cutoff we compute the corresponding
            # sampling
            # shape:n_atom in the cutoff, n_ etas, radial samples
            per_atom_contrib = G_radial(cutoff_rij[:, np.newaxis, np.newaxis],
                                        sampling_rad[np.newaxis, np.newaxis, :],
                                        eta_rad[np.newaxis, :, np.newaxis], Rc_rad)
            for atom_kind in range(n_species):
                # for each atom_kind we take the idxs of all atoms of that
                # kind inside the cutoff
                species_idxs = np.where(cutoff_species == atom_kind)
                # we use the indexs to extract the g's contribution
                # of that species
                species_per_atom_contrib = per_atom_contrib[species_idxs]
                # contract over all the contibutions
                kind_radial_g = species_per_atom_contrib.sum(0).reshape(-1)
                # creation of the gs
                Gs[idx] = np.append(Gs[idx], kind_radial_g)
            # ===  END RADIAL PART 2 ===

            # === ANGULAR PART 2 ===
            # same extraction as the radial part
            # shape: atoms_inside_cutoff, 3
            # cutoff_positions = positions_extended[angular_mask[idx]]
            cutoff_deltas = deltas[idx, angular_mask[idx]]
            # shape: atoms_inside_cutoff
            cutoff_species = species[angular_mask[idx]]
            cutoff_rij = rij[idx, angular_mask[idx]]

            for atom_kind_1 in range(n_species):
                for atom_kind_2 in range(atom_kind_1, n_species):
                    # prevent double counting if atom i is equal tom atom j
                    # TODO in emine version angular prefactor = 1 if i = j else 2
                    # decide which we want to keep
                    prefactor = 1.0 if atom_kind_1 != atom_kind_2 else 0.5
                    # for each atom_kind we take the idxs of all atoms of that
                    # kind inside the cutoff, we do this for both the species
                    species_idxs_1 = np.where(cutoff_species == atom_kind_1)
                    species_idxs_2 = np.where(cutoff_species == atom_kind_2)
                    # in the same way we extract all the quanities needed to compute
                    # the angular contribution
                    cutoff_deltas_ij = cutoff_deltas[species_idxs_1]
                    cutoff_deltas_ik = cutoff_deltas[species_idxs_2]
                    cutoff_r_ij = cutoff_rij[species_idxs_1]
                    cutoff_r_ik = cutoff_rij[species_idxs_2]
                    # compute the distance between jk
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    cutoff_r_jk = np.linalg.norm(cutoff_deltas_ij[:, np.newaxis, :]
                                                 - cutoff_deltas_ik,
                                                 axis=-1)
                    # = computation of the angle between ikj triplet =
                    # numerator: ij dot ik
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    a = np.sum(cutoff_deltas_ij[:, np.newaxis, :] * cutoff_deltas_ik, 2)
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
                    # computation of all the elements
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff, eta_ang_elements,
                    #        zeta_elements, 2
                    kind1_kind2_angular_g = G_angular_BP(
                        cutoff_r_ij[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                        cutoff_r_ik[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis],
                        cutoff_r_jk[:, :, np.newaxis, np.newaxis, np.newaxis],
                        theta_ijk[:, :, np.newaxis, np.newaxis, np.newaxis],
                        eta_ang[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis],
                        zeta[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis],
                        lamb[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :],
                        Rc_ang)
                    # creation of a mask to esclude counting of j==k
                    # shape: n_atom_species1_inside_cutoff,
                    #        n_atom_species2_inside_cutoff
                    f = np.logical_or(
                        np.abs(cutoff_r_ij[:, np.newaxis] - cutoff_r_ik) > 1e-5,
                        cos_theta_ijk < .99999)
                    # collection of the contributions:
                    # shape: n_of_triplet 1x2,
                    #        eta_ang_elements, zeta_elements, 2
                    kind1_kind2_angular_g = kind1_kind2_angular_g[f]
                    # contraction over the number of triplet, flattenig and adding it
                    # to the total G
                    # when reshaping the unroll is done right to left
                    # so the sequence is lambda, for each zeta, for each eta_ang
                    kind1_kind2_angular_g = prefactor * np.sum(
                        kind1_kind2_angular_g, 0).reshape(-1)
                    Gs[idx] = np.append(Gs[idx], kind1_kind2_angular_g)
            # === END ANGULAR PART 2 ===

        output = {'key': key, 'Gvect': np.asarray(Gs).tolist()}
        return output
