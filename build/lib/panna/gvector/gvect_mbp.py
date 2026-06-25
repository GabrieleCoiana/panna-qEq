###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import itertools
import logging
from functools import partial

import numpy as np
import tensorflow as tf

from .helper_functions import G_angular_mBP, G_radial, GdG_angular_mBP, GdG_radial
from .gvect_base import GvectBase, requires_parameters
from .pbc import replicas_max_idx

# logger
logger = logging.getLogger('panna')


class GvectmBP(GvectBase):
    """ Modified Behler Parinello descriptor

    Parameters
    ----------
      compute_dgvect : boolean
      sparse_dgvect : boolean
      species : string of comma separated values
        sequence of species eg: C,H,N,O
      pbc_directions: optional
        pbc directions to override those in the json

    Notes
    -----
      gvectmbp.units must be setted before usage!
      for a list of parameters that must be set:
      gvectmbp.descriptor_parameters.
      If you are adding a new parameter you need to add it in the following functions too:
      _gvector_parameters dictionary
      parse_parameters func
      gvect func
      __call__ func
    """

    name = 'mBP'
    doi = '10.1016/j.cpc.2020.107402'

    _gvect_parameters = {
        # RADIAL_COMPONENTS
        'Rc_rad': lambda self, x: x * self.unit2A,
        'Rs_rad': lambda self, x: x * self.unit2A,
        'eta_rad': lambda self, x: x / (self.unit2A * self.unit2A),
        # ANGULAR_COMPONENTS
        'Rc_ang': lambda self, x: x * self.unit2A,
        'eta_ang': lambda self, x: x / (self.unit2A * self.unit2A),
        'Rs_ang': lambda self, x: x * self.unit2A,
        'Thetas': lambda self, x: x,
        'zeta': lambda self, x: x,
    }

    def parse_parameters(self, gv_param):
        self.units = gv_param.get('gvect_parameters_unit', 'angstrom')
        # RADIAL_COMPONENTS
        # pylint: disable=invalid-name,attribute-defined-outside-init

        Rc_rad = gv_param.getfloat('Rc_rad')
        self.update_parameter('Rc_rad', Rc_rad)

        Rs0_rad = gv_param.getfloat('Rs0_rad', 0.0)
        RsN_rad = gv_param.getint('RsN_rad', None)
        Rs_rad = gv_param.get_comma_list_floats('Rs_rad_list', "")
        if len(Rs_rad) >= 1:
            logger.info('Radial Gaussian centers are set by Rs_rad_list')
            Rs_rad = np.asarray(Rs_rad)
        else:
            logger.info('Radial Gaussian centers are set by'
                        ' Rs0_rad, Rc_rad, RsN_rad')
            Rsst_rad = (Rc_rad - Rs0_rad) / RsN_rad
            Rs_rad = np.arange(Rs0_rad, Rc_rad, Rsst_rad)

        self.update_parameter('Rs_rad', Rs_rad)

        eta_rad = gv_param.get_comma_list_floats('eta_rad', None)
        if len(eta_rad) > 1:
            logger.info('eta_rad is determined by a list of eta_rad values')
            if len(eta_rad) != len(Rs_rad):
                logger.info('eta_rad and Rs_rad sizes do not match -'
                            ' - tiling the first element eta_rad'
                            ' to the size of Rs_rad')
                eta_rad = np.tile(np.asarray(eta_rad[0]), len(Rs_rad))
        else:
            eta_rad = np.tile(np.asarray(eta_rad), len(Rs_rad))
        self.update_parameter('eta_rad', eta_rad)

        # ANGULAR_COMPONENTS

        ThetasN = gv_param.getint('ThetasN', None)
        Thetas = gv_param.get_comma_list_floats('Thetas_list', '')  # in degrees
        if len(Thetas) >= 1:
            logger.info('Angular descriptor centers are set by Thetas_list')
            Thetas = np.asarray(Thetas) / 180.0 * np.pi
        else:
            logger.info('Angular descriptor centers are set by ThetasN')
            step_theta = np.pi / ThetasN
            Thetas = np.arange(0, np.pi, step_theta) + .5 * step_theta

        self.update_parameter('Thetas', Thetas)

        zeta = gv_param.get_comma_list_integers('zeta', None)
        if len(zeta) > 1:
            logger.info('zeta is determined by a list of zeta values')
            if len(zeta) != len(Thetas):
                logger.info('zeta and Thetas sizes do not match -'
                            ' tiling the first element of zeta!')
                zeta = np.tile(np.asarray(zeta[0]), len(Thetas))
        else:  # single zeta value
            zeta = np.tile(np.asarray(zeta), len(Thetas))

        self.update_parameter('zeta', zeta)

        # RADIAL_ANGULAR COMPONENTS
        Rc_ang = gv_param.getfloat('Rc_ang')
        self.update_parameter('Rc_ang', Rc_ang)

        Rs0_ang = gv_param.getfloat('Rs0_ang', None)
        RsN_ang = gv_param.getint('RsN_ang', None)
        Rs_ang = gv_param.get_comma_list_floats('Rs_ang_list', "")
        if len(Rs_ang) >= 1:
            logger.info('Radial-angular Gaussian centers are set by Rs_ang_list')
            Rs_ang = np.asarray(Rs_ang)
        else:
            logger.info('Radial-angular Gaussian centers are '
                        'set by Rs0_ang, Rc_ang, RsN_ang')
            Rsst_ang = (Rc_ang - Rs0_ang) / RsN_ang
            Rs_ang = np.arange(Rs0_ang, Rc_ang, Rsst_ang)

        self.update_parameter('Rs_ang', Rs_ang)

        eta_ang = gv_param.get_comma_list_floats('eta_ang', None)
        if len(eta_ang) > 1:
            logger.info('eta_ang is determined by a list of eta_ang values')
            if len(eta_ang) != len(Rs_ang):
                logger.info('eta_ang and Rs_ang sizes do not match - '
                            'tiling the first element of eta_ang!')
                eta_ang = np.tile(np.asarray(eta_ang[0]), len(Rs_ang))
        else:  # single eta_ang value
            eta_ang = np.tile(np.asarray(eta_ang), len(Rs_ang))
        self.update_parameter('eta_ang', eta_ang)

        # we do not support this anymore
        # Rsst_ang = gv_par.getfloat('Rsst_ang', -1)

    @property
    @requires_parameters
    def gsize(self):
        ns = self.number_of_species
        return int(ns * len(self.Rs_rad)
                   + 0.5 * ns * (ns + 1) * len(self.Rs_ang) * len(self.Thetas))

    @property
    @requires_parameters
    def grad_size(self):
        """g radial size"""
        ns = self.number_of_species
        return int(ns * len(self.Rs_rad))

    @property
    @requires_parameters
    def gvect_v0(self):
        """ extract the panna potential with this format
        if you intend to use with OPENKIM
        restriction: equidistant Rs etc.
        """
        parameters = {
            # RADIAL_COMPONENTS
            'eta_rad': self.eta_rad[0],
            'Rc_rad': self.Rc_rad,
            'Rs0_rad': self.Rs_rad[0],
            'RsN_rad': len(self.Rs_rad),
            'Rsst_rad': (self.Rc_rad - self.Rs_rad[0]) / float(len(self.Rs_rad)),
            # ANGULAR_COMPONENTS
            'eta_ang': self.eta_ang[0],
            'Rc_ang': self.Rc_ang,
            'Rs0_ang': self.Rs_ang[0],
            'RsN_ang': len(self.Rs_ang),
            'Rsst_ang': (self.Rc_ang - self.Rs_ang[0]) / float(len(self.Rs_ang)),
            'zeta': self.zeta[0],
            'ThetasN': len(self.Thetas)
        }
        return parameters

    @property
    @requires_parameters
    def gvect(self):
        """ extract the panna potential this way for all other purposes
        """
        parameters = {
            'eta_rad': self.eta_rad.tolist(),
            'Rc_rad': self.Rc_rad,
            'RsN_rad': len(self.Rs_rad),
            'RsN_ang': len(self.Rs_ang),
            'eta_ang': self.eta_ang.tolist(),
            'Rc_ang': self.Rc_ang,
            'Rs_rad': self.Rs_rad.tolist(),
            'Rs_ang': self.Rs_ang.tolist(),
            'Thetas': self.Thetas.tolist(),
            'zeta': self.zeta.tolist(),
            'ThetasN': len(self.Thetas),
        }
        return parameters

    @requires_parameters
    def __call__(self, key, positions, species, lattice_vectors, **kwargs):
        """Calculate the gvector based on given parameters, using list

        Args:
            key: key of the simulation
            positions: list of atomic positions

            species: list of species as idx

            lattice_vectors: list [a1, a2, a3]

            For all the other parameters refer to the article

        kwargs:
            pbc: Normally pbc are recovered from file,
                 if in the lattice_vectors a direction is set to zero
                 then no pbc is applied in that direction.
                 This argument allow you to turn off specific directions
                 by passing an array of 3 logical value (one for each
                 direction). False value turn off that specific direction
                 eg. pbc = [True, False, True] => pbc along a1 and a3

        Return:
            dict, class has two flags: compute_dgvect, sparse_dgvect
            if False, False:
                key, Gvect
            elif True, False:
                key, Gvect, dGvect
            elif True, True:
                key, Gvect, dGvect_val, dGvect_ind

            where:
                key: same key passed
                Gvect: list of gvector, one per atom, shape (natoms, gsize)
                dGvect: list of gvector derivatives, one per atom
                dGvect_val: sparse format of dGvect, values
                dGvect_ind: sparse format of dGvect, indeces


        note:
            All kind of PBC : 1D, 2D and 3D are implemented..
            Check the function "replicas_max_idx"

        """
        output = {'key': key}

        # define shorter names:
        # and make sure arrays are arrays
        n_species = self.number_of_species
        eta_rad = np.asarray(self.eta_rad)
        Rc_rad = self.Rc_rad
        Rs_rad = np.asarray(self.Rs_rad)
        eta_ang = np.asarray(self.eta_ang)
        Rc_ang = self.Rc_ang
        Rs_ang = np.asarray(self.Rs_ang)
        zeta = np.asarray(self.zeta)
        Thetas = np.asarray(self.Thetas)
        gradsize = self.grad_size
        n_Rad = len(self.Rs_rad)
        n_Ang = len(self.Rs_ang) * len(Thetas)
        # np.savetxt("aa_rsrad.dat",Rs_rad)
        # np.savetxt("aa_thetas.dat",Thetas)
        # np.savetxt("aa_rsang.dat",Rs_ang)

        # ensure that everything is a numpy array
        positions = np.asarray(positions)
        nat_species = np.asarray(species)
        lattice_vectors = np.asarray(lattice_vectors)
        n_atoms = len(positions)
        Gs = [np.empty(0) for x in range(n_atoms)]
        dGs_dense = np.zeros((n_atoms, self.gsize, n_atoms, 3))

        # wrap the atoms in the unit cell
        if (lattice_vectors != 0).any():
            crystal_coord = positions @ np.linalg.inv(lattice_vectors)
            crystal_coord = crystal_coord % 1.0
            positions = crystal_coord @ lattice_vectors

        # compute how many cell replica we need in each direction
        if self.pbc_directions is not None:
            max_indices = replicas_max_idx(lattice_vectors,
                                           max(Rc_rad, Rc_ang),
                                           pbc=self.pbc_directions)
        else:
            max_indices = replicas_max_idx(lattice_vectors, max(Rc_rad, Rc_ang))
        l_max, m_max, n_max = max_indices
        l_list = range(-l_max, l_max + 1)
        m_list = range(-m_max, m_max + 1)
        n_list = range(-n_max, n_max + 1)
        # create a matrix with all the idx of the extra cell we need
        replicas = np.asarray(list(itertools.product(l_list, m_list, n_list)))

        # translation vectors needed to map each atom in the unit cell to its replica
        # @ means matrix multiplication
        replicas = replicas @ lattice_vectors
        n_replicas = len(replicas)
        # create a tensor with the coordinate of all the atoms and needed replicas
        # and reshape it like positions tensor
        # shape: n_atoms, n_replicas, 3
        positions_extended = positions[:, np.newaxis, :] + replicas
        # Reshape default order is 'C',i.e. read as last index is changing fastest
        positions_extended = positions_extended.reshape(n_atoms * n_replicas, 3)
        # The resulting order is
        # [...position of atom 1 replicas... , ...position of atom2 replicas... , ]
        # creating the equivalent species tensor
        species = np.tile(nat_species[:, np.newaxis],
                          (1, n_replicas)).reshape(n_atoms * n_replicas)
        # [ species of atom1 replicas, species of atom2 replicas, ... ]
        # other way around: for each replica remember what
        # its index in the unit cell was
        my_idx_in_uc = np.tile(np.arange(n_atoms)[:, np.newaxis],
                               (1, n_replicas)).reshape(n_atoms * n_replicas)
        # [ 0, 1, 2, .., natom, 0, 1, 2, .., natom]
        # computing x_i - x_j between all the atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_replicas * n_atoms , 3
        deltas = positions_extended - positions[:, np.newaxis, :]
        # using the deltas to compute rij for each
        # atom in the unit cell and all the atom
        # in the cell + replica tensor
        # shape: n_atoms, n_all_replicas
        rij = np.linalg.norm(deltas, axis=-1)

        # === RADIAL PART 1 ===
        # create a boolean mask to extract only atoms inside the cutoff of a given atom
        # shape: n_atoms, n_all_replicas
        # each row contains True value if the given j atom is inside the
        # cutoff wrt the i atom, logical "and" to exclude counting the
        # central atom
        radial_mask = np.logical_and(rij < Rc_rad, rij > 1e-8)

        # create tensors of parameters for the sampling to allow proper broadcasting
        # sampling_rad = np.arange(Rs0_rad, Rc_rad, Rsst_rad)
        sampling_rad = Rs_rad[np.newaxis, :]
        eta_rad = eta_rad[np.newaxis, :]
        # ===  END RADIAL PART 1 ===

        # === ANGULAR PART 1 ===
        # same mask as the radial part with angular cutoff
        # shape: n_atoms, n_all_replicas
        angular_mask = np.logical_and(rij < Rc_ang, rij > 1e-8)

        # create tensors of parameters for the sampling to allow proper broadcasting
        # angle centers are shifted by half differently than mBP reference.
        # step_theta = np.pi / ThetasN
        # sampling_theta = np.arange(0, np.pi, step_theta) + .5 * step_theta
        # already cos/sin here?
        cos_sampling_theta = np.cos(Thetas)
        sin_sampling_theta = np.sin(Thetas)
        cos_sampling_theta = cos_sampling_theta[np.newaxis, np.newaxis, np.newaxis, :]
        sin_sampling_theta = sin_sampling_theta[np.newaxis, np.newaxis, np.newaxis, :]
        zeta = zeta[np.newaxis, np.newaxis, np.newaxis, :]

        # sampling_rad_ang = np.arange(Rs0_ang, Rc_ang, Rsst_ang)
        sampling_rad_ang = Rs_ang[np.newaxis, np.newaxis, :, np.newaxis]
        eta_ang = eta_ang[np.newaxis, np.newaxis, :, np.newaxis]
        # === END ANGULAR PART 1 ===

        if self.compute_dgvect:
            for idx in range(n_atoms):
                # === RADIAL PART 2 ===
                # shape: (atoms in cutoff , 3)
                cutoff_deltas = deltas[idx, radial_mask[idx]]
                # shape: (atoms in cutoff)
                cutoff_rij = rij[idx, radial_mask[idx]]
                cutoff_species = species[radial_mask[idx]]
                cutoff_idx_in_uc = my_idx_in_uc[radial_mask[idx]]
                # shape:
                # per_atom_G (atoms in cutoff , G-radial size for a single pair)
                # per_atom_dG (atoms in cutoff, G-radial size for a single pair)
                per_atom_G, per_atom_dG = GdG_radial(cutoff_rij[:, np.newaxis],
                                                     sampling_rad, eta_rad, Rc_rad)
                # per_atom_dG
                # (atoms in cutoff, G-radial size for a single pair , 3)
                per_atom_dG = per_atom_dG[:, :, np.newaxis] \
                    * cutoff_deltas[:, np.newaxis, :]

                # We will need this below:
                # dG(r_ij) as we calculated above is not yet complete,
                # coz we are looking into dG_i/dr_i or dG_i/dr_j
                # when j=i, it has + sign, when j!=i && j!=i' (replica of i)
                # it has - sign;
                # when j=i', a replica of i, dG(r_ii') = dG(r_i - r_i - nT)
                # where T is a lattice vector, so dG/dri=0.
                # So we can just flip the sign of only +1 component where j=i
                # (dGi/dr_i) later.
                # multiplier = np.where(np.arange(n_atoms)==idx,1,-1)
                # multiplier = np.tile(multiplier[:, np.newaxis],
                # (1, n_replicas)).reshape(n_atoms * n_replicas)
                # [..first atom replicas ... second atom replicas ..
                # idx atoms replicas .. ]
                # exactly as positions and species so that we can use the
                # radial mask again
                # multiplier = multiplier[radial_mask[idx]]
                # per_atom_dG =
                # per_atom_dG * multiplier[:,np.newaxis,np.newaxis]

                for atom_kind in range(n_species):
                    # for each atom_kind we take the idxs of all atoms of
                    # that kind inside the cutoff
                    species_idx = np.nonzero(cutoff_species == atom_kind)
                    # we use the indeces to extract the g's contribution of
                    # that species and contract
                    s_per_atom_G = per_atom_G[species_idx].sum(0)
                    Gs[idx] = np.append(Gs[idx], s_per_atom_G)
                    #
                # dG/dR does not have to know of species
                # m = -1 if j is idx replica, else  m= +1
                multiplier = np.where(cutoff_idx_in_uc == idx, -1, 1)
                for j in range(np.shape(per_atom_dG)[0]):
                    if cutoff_species[j] > (n_species - 1):
                        # this is not detected automatically by numpy
                        raise ValueError('index error in index specie')
                    dGs_dense[idx,
                              n_Rad * cutoff_species[j]:n_Rad * (cutoff_species[j] + 1),
                              cutoff_idx_in_uc[j],
                              :] -= \
                        per_atom_dG[j, :, :] * multiplier[j]
                    dGs_dense[idx,
                              n_Rad * cutoff_species[j]:n_Rad * (cutoff_species[j] + 1),
                              idx,
                              :] += \
                        per_atom_dG[j, :, :] * multiplier[j]

                #  ===  END RADIAL PART 2 ===

                # === ANGULAR PART 2 ===
                # same extraction as the radial part
                # shape: atoms_inside_cutoff, 3
                # cutoff_positions = positions_extended[angular_mask[idx]]
                cutoff_deltas = deltas[idx, angular_mask[idx]]
                # shape: atoms_inside_cutoff
                cutoff_species = species[angular_mask[idx]]
                cutoff_r = rij[idx, angular_mask[idx]]
                cutoff_idx_in_uc = my_idx_in_uc[angular_mask[idx]]
                # cutoff_idx_ia_uc = my_idx_in_uc[angular_mask[idx]]
                for atom_kind_1 in range(n_species):
                    for atom_kind_2 in range(atom_kind_1, n_species):
                        # prevent double counting if atom i is equal to atom j
                        # prefactor = 1.0 if atom_kind_1 != atom_kind_2 else 0.5
                        prefactor = (2.0 - int(atom_kind_1 == atom_kind_2)) * 0.5
                        # for each atom_kind we take the idxs of all
                        # atoms of that
                        # kind inside the cutoff, we do this for both the
                        # species
                        species_idxs_1 = np.where(cutoff_species == atom_kind_1)[0]
                        species_idxs_2 = np.where(cutoff_species == atom_kind_2)[0]
                        n1 = len(species_idxs_1)
                        n2 = len(species_idxs_2)
                        # Enter only if atoms exist
                        if n1 * n2 > 0:
                            sp_shift = int((n_species - 1) * atom_kind_1
                                           - (atom_kind_1 - 1) * atom_kind_1 / 2
                                           + atom_kind_2)
                            # in the same way we extract all the quanities needed
                            # to compute the angular contribution
                            cutoff_deltas_ij = cutoff_deltas[species_idxs_1]
                            cutoff_deltas_ik = cutoff_deltas[species_idxs_2]
                            cutoff_r_ij = cutoff_r[species_idxs_1]
                            cutoff_r_ik = cutoff_r[species_idxs_2]
                            # = computation of the angle between ij-ik triplet =
                            # numerator: ij dot ik
                            # shape: n_atom_species1_inside_cutoff,
                            #        n_atom_species2_inside_cutoff
                            a = np.sum(
                                cutoff_deltas_ij[:, np.newaxis, :] * cutoff_deltas_ik,
                                2)
                            # denominator: |rij| * |rik|
                            # shape: n_atom_species1_inside_cutoff,
                            #        n_atom_species2_inside_cutoff
                            b = cutoff_r_ij[:, np.newaxis] * cutoff_r_ik
                            # element by element ratio
                            cos_theta_jik = a / b
                            # correct numerical error
                            cos_theta_jik[cos_theta_jik >= 1.0] = 1.0
                            cos_theta_jik[cos_theta_jik <= -1.0] = -1.0
                            # compute the angle
                            # shape: n_atom_species1_inside_cutoff,
                            #        n_atom_species2_inside_cutoff
                            theta_jik = np.arccos(cos_theta_jik)
                            # computation of all the elements
                            # shape: n_atom_species1_inside_cutoff,
                            #        n_atom_species2_inside_cutoff,
                            #        radial_ang_sampling, ang_sampling
                            k1_k2_ang_G, dG1, dG2, dG3 = GdG_angular_mBP(
                                cutoff_r_ij[:, np.newaxis, np.newaxis, np.newaxis],
                                cutoff_r_ik[np.newaxis, :, np.newaxis, np.newaxis],
                                theta_jik[:, :, np.newaxis, np.newaxis],
                                sampling_rad_ang, cos_sampling_theta,
                                sin_sampling_theta, eta_ang, zeta, Rc_ang, prefactor)
                            # all dGs have the same shape of G right now:
                            # (n_atom_spec1_in_cut,
                            #  n_atom_spec2_in_cut,
                            #  rad_ang_sampling x ang_sampling)

                            # creation of a mask to exclude counting of j==k
                            # shape: n_atom_species1_inside_cutoff,
                            #        n_atom_species2_inside_cutoff
                            f = np.logical_or(
                                np.abs(cutoff_r_ij[:, np.newaxis] - cutoff_r_ik) > 1e-5,
                                cos_theta_jik < .99999)
                            # shape: n_of_triplet 1x2,
                            #        radial_ang_sampling x ang_sampling
                            k1_k2_ang_G = k1_k2_ang_G[f, :]
                            # contract over the number of triplet for each i,
                            # already flattened for Rs and Theta_s,
                            k1_k2_ang_G = np.reshape(np.sum(k1_k2_ang_G, 0), (-1))
                            # append to G_i
                            Gs[idx] = np.append(Gs[idx], k1_k2_ang_G)
                            #
                            dG1 = np.reshape(dG1, (n1, n2, n_Ang)) * f[:, :, np.newaxis]
                            dG2 = np.reshape(dG2, (n1, n2, n_Ang)) * f[:, :, np.newaxis]
                            dG3 = np.reshape(dG3, (n1, n2, n_Ang)) * f[:, :, np.newaxis]
                            # if idx==0:
                            #     print(dG1[:,:,0],dG2[:,:,0],dG3[:,:,0])
                            jcontrib = \
                                dG1[:, :, :, np.newaxis] \
                                * cutoff_deltas_ij[:, np.newaxis, np.newaxis, :]
                            jcontrib += \
                                dG2[:, :, :, np.newaxis] \
                                * cutoff_deltas_ik[np.newaxis, :, np.newaxis, :]
                            kcontrib = \
                                dG2[:, :, :, np.newaxis] \
                                * cutoff_deltas_ij[:, np.newaxis, np.newaxis, :]
                            kcontrib += \
                                dG3[:, :, :, np.newaxis] \
                                * cutoff_deltas_ik[np.newaxis, :, np.newaxis, :]
                            # all the prefactor business is already done in the GdG call
                            for j, nj in enumerate(species_idxs_1):
                                for k, nk in enumerate(species_idxs_2):
                                    dGs_dense[idx,
                                              gradsize + sp_shift * n_Ang:
                                              gradsize + (sp_shift + 1) * n_Ang,
                                              cutoff_idx_in_uc[nj],
                                              :] += jcontrib[j, k, :, :]
                                    dGs_dense[idx,
                                              gradsize + sp_shift * n_Ang:
                                              gradsize + (sp_shift + 1) * n_Ang,
                                              cutoff_idx_in_uc[nk],
                                              :] += kcontrib[j, k, :, :]
                                    dGs_dense[idx,
                                              gradsize + sp_shift * n_Ang:
                                              gradsize + (sp_shift + 1) * n_Ang,
                                              idx,
                                              :] -= jcontrib[j, k, :, :] \
                                        + kcontrib[j, k, :, :]
                        else:
                            Gs[idx] = np.append(Gs[idx], np.zeros(n_Ang))

                # === END ANGULAR PART 2 ===
            # np.savetxt('newv_G.txt',Gs,  fmt='%12.8f' )
            # np.savetxt('newv_dG.txt',dGs_dense[0,:,0,:], fmt='%12.8f' )
            # np.savetxt('newv_dG.txt',dGs_dense[0,:,1,:], fmt='%12.8f' )
            # np.savetxt('newv_dG.txt',dGs_dense[0,:,2,:], fmt='%12.8f' )
            # np.savetxt('newv_dG.txt',dGs_dense[1,:,1,:], fmt='%12.8f' )
            # np.savetxt('newv_dG.txt',dGs_dense[1,:,2,:], fmt='%12.8f' )
            # np.savetxt('newv_dG.txt',dGs_dense[2,:,2,:], fmt='%12.8f' )

            output['Gvect'] = np.asarray(Gs).tolist()
            if self.sparse_dgvect:
                idxes = np.where(dGs_dense != 0.0)
                values = dGs_dense[idxes]
                i, j, k, direction = idxes
                idx_1 = i * self.gsize + j
                idx_2 = 3 * k + direction
                dgdx_indices = np.vstack([idx_1, idx_2]).T
                output['dGvect_val'] = values.tolist()
                output['dGvect_ind'] = dgdx_indices.tolist()
                return output
            output['dGvect'] = dGs_dense.tolist()
            return output
        else:
            for idx in range(n_atoms):
                cutoff_species = species[radial_mask[idx]]
                cutoff_rij = rij[idx, radial_mask[idx]]
                per_atom_contrib = G_radial(cutoff_rij[:, np.newaxis], sampling_rad,
                                            eta_rad, Rc_rad)
                for atom_kind in range(n_species):
                    species_idxs = np.where(cutoff_species == atom_kind)
                    species_per_atom_contrib = per_atom_contrib[species_idxs]
                    kind_radial_g = species_per_atom_contrib.sum(0)
                    Gs[idx] = np.append(Gs[idx], kind_radial_g)
                # ===  END RADIAL PART 2 ===

                # === ANGULAR PART 2 ===
                cutoff_deltas = deltas[idx, angular_mask[idx]]
                cutoff_species = species[angular_mask[idx]]
                cutoff_rij = rij[idx, angular_mask[idx]]
                for atom_kind_1 in range(n_species):
                    for atom_kind_2 in range(atom_kind_1, n_species):
                        prefactor = 1.0 if atom_kind_1 != atom_kind_2 else 0.5
                        species_idxs_1 = np.where(cutoff_species == atom_kind_1)
                        species_idxs_2 = np.where(cutoff_species == atom_kind_2)
                        cutoff_deltas_ij = cutoff_deltas[species_idxs_1]
                        cutoff_deltas_ik = cutoff_deltas[species_idxs_2]
                        cutoff_r_ij = cutoff_rij[species_idxs_1]
                        cutoff_r_ik = cutoff_rij[species_idxs_2]
                        a = np.sum(cutoff_deltas_ij[:, np.newaxis, :]
                                   * cutoff_deltas_ik, 2)
                        b = cutoff_r_ij[:, np.newaxis] * cutoff_r_ik
                        cos_theta_ijk = a / b
                        cos_theta_ijk[cos_theta_ijk >= 1.0] = 1.0
                        cos_theta_ijk[cos_theta_ijk <= -1.0] = -1.0
                        theta_ijk = np.arccos(cos_theta_ijk)
                        kind1_kind2_angular_g = G_angular_mBP(
                            cutoff_r_ij[:, np.newaxis, np.newaxis, np.newaxis],
                            cutoff_r_ik[np.newaxis, :, np.newaxis, np.newaxis],
                            theta_ijk[:, :, np.newaxis, np.newaxis],
                            sampling_rad_ang, Thetas, eta_ang, zeta, Rc_ang)
                        f = np.logical_or(
                            np.abs(cutoff_r_ij[:, np.newaxis] - cutoff_r_ik) > 1e-5,
                            cos_theta_ijk < .99999)
                        kind1_kind2_angular_g = kind1_kind2_angular_g[f, :, :]
                        kind1_kind2_angular_g = prefactor \
                            * np.sum(kind1_kind2_angular_g, 0).reshape(-1)
                        Gs[idx] = np.append(Gs[idx], kind1_kind2_angular_g)
                # === END ANGULAR PART 2 ===

            output['Gvect'] = np.asarray(Gs).tolist()
            return output

    @tf.function(jit_compile=True,
        input_signature=[[tf.TensorSpec(shape=(), dtype=tf.int32),\
                         tf.TensorSpec(shape=(None), dtype=tf.int32),\
                         tf.TensorSpec(shape=(None,3), dtype=tf.float32),\
                         tf.TensorSpec(shape=(None,None,None), dtype=tf.int32),\
                         tf.TensorSpec(shape=(None,None,2), dtype=tf.int32),\
                         tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32),\
                         tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),\
                         tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),\
                         tf.TensorSpec(shape=(None,None,None), dtype=tf.float32)]])

    def tf_g(self, x):
        """Operating on each element of the batch.

        x elements: 0 - nat
                    1 - species
                    2 - positions
                    3 - indices of nn
                    4 - number of indices per atom
                    5 - ij vectors
                    6 - ij radii
                    7 - mask of smaller cutoff
                    8 - mask or larger cutoff
        """
        nats_padded = tf.shape(x[1])[0]
        # Temporary flag to choose code version
        RMOD = 2

        # define shorter names
        # and make sure arrays are arrays
        n_species = self.number_of_species
        eta_rad = np.asarray(self.eta_rad, dtype=np.float32)
        Rc_rad = np.asarray(self.Rc_rad, dtype=np.float32)
        Rs_rad = np.asarray(self.Rs_rad, dtype=np.float32)
        if len(self.eta_ang)==len(self.Rs_ang):
            eta_ang = np.asarray(self.eta_ang, dtype=np.float32)
        else:
            eta_ang = self.eta_ang*np.ones(len(self.Rs_ang), dtype=np.float32)
        Rc_ang = np.asarray(self.Rc_ang, dtype=np.float32)
        Rs_ang = np.asarray(self.Rs_ang, dtype=np.float32)
        if len(self.zeta)==len(self.Thetas):
            zeta = np.asarray(self.zeta, dtype=np.float32)
        else:
            zeta = self.zeta*np.ones(len(self.Thetas), dtype=np.float32)
        Thetas = np.asarray(self.Thetas, dtype=np.float32)
        gsize = self.gsize
        n_Rad = len(self.Rs_rad)
        n_Ang_r = len(self.Rs_ang)
        n_Ang_a = len(Thetas)
        n_Ang = n_Ang_r * n_Ang_a
        angsize = n_Ang * int(n_species*(n_species+1)/2)

        # Cut inputs to the actual atoms
        n_atoms = x[0]
        species = x[1][:n_atoms]
        positions = x[2][:n_atoms]
        nn_inds = x[3][:n_atoms]
        nn_num = x[4][:n_atoms]   # WOULD IT BE USEFUL TO USE THIS?
        nn_vecs = x[5][:n_atoms]
        nn_r = x[6][:n_atoms]
        if Rc_ang<=Rc_rad:
            maskang = x[7][:n_atoms]
            maskrad = x[8][:n_atoms]
        else:
            maskang = x[8][:n_atoms]
            maskrad = x[7][:n_atoms]

        rij = tf.reshape(nn_r,[n_atoms,n_species,-1,1])
        nnmax = tf.shape(rij)[2]
        maskrad = tf.reshape(maskrad,[n_atoms,n_species,-1,1])
        centers = tf.reshape(Rs_rad, [1,1,1,-1])
        etas = tf.reshape(eta_rad, [1,1,1,-1])
        if self.compute_dgvect:
            Grad, dGrad_t = tf_GdG_radial(rij,centers,etas,Rc_rad)
            Grad = tf.reshape(tf.reduce_sum(maskrad*Grad, axis=2),[n_atoms,-1])
            # Not putting the multiplier as I don't think it's needed...
            # Multiplying with the displacements
            dGrad_t = tf.reshape(maskrad*dGrad_t,[n_atoms,n_species,-1,n_Rad,1]) *\
                      tf.reshape(nn_vecs,[n_atoms,n_species,-1,1,3])
            # Summing all derivatives w.r.t i
            dGrad = tf.reshape(tf.reduce_sum(dGrad_t, axis=2),[n_atoms,-1,3])
            # Putting in the right place
            dGrad = tf.einsum('ijl,ik->ijkl',dGrad,tf.eye(n_atoms))
            # Adding all the derivatives wrt j
            # For each nn we scatter to the atom in the right position full list
            # Then we reshuffle this to be the right contribution in the dG matrix
            ind1 = tf.tile(tf.reshape(tf.range(n_atoms),[-1,1,1]),[1,n_species,nnmax])
            ind2 = tf.tile(tf.reshape(tf.range(n_species),[1,-1,1]),[n_atoms,1,nnmax])
            inds = tf.stack([ind1,ind2,nn_inds],axis=-1)
            dGradj = tf.zeros([n_atoms,n_species,n_atoms,n_Rad,3])
            dGradj = tf.tensor_scatter_nd_add(dGradj,inds,dGrad_t)
            dGradj = tf.reshape(tf.transpose(dGradj, perm=[0,1,3,2,4]),[n_atoms,-1,n_atoms,3])
            dGrad = dGrad - dGradj
        else:
            Grad = tf_G_radial(rij,centers,etas,Rc_rad)
            Grad = tf.reshape(tf.reduce_sum(maskrad*Grad, axis=2),[n_atoms,-1])

        
        # In this version, we use map_fn on GdG_ang, and process each atom on its own
        # We also do all of the processing inside GdG
        # This clearly uses less memory, but might parallelize less
        # Shapes should be [nat,nsp,nsp,nn,nn,gangR,gangA]
        # but the first dimension is mapped over, so we reshape:
        Rs_ang = tf.reshape(Rs_ang,[1,1,1,1,-1,1])
        eta_ang = tf.reshape(eta_ang,[1,1,1,1,-1,1])
        zeta = tf.reshape(zeta,[1,1,1,1,1,-1])
        CTh = tf.reshape(tf.cos(Thetas),[1,1,1,1,1,-1])
        STh = tf.reshape(tf.sin(Thetas),[1,1,1,1,1,-1])
        # Mask to remove the j==k case
        maskjk = tf.reshape(tf.ones([n_species,n_species,nnmax,nnmax]) -\
                          (tf.reshape(tf.eye(n_species),[n_species,n_species,1,1]) *\
                            tf.reshape(tf.eye(nnmax),[1,1,nnmax,nnmax])), \
                          [n_species,n_species,nnmax,nnmax,1,1])
        # Input for the mapped function:
        input_stack = [nn_r,nn_vecs,maskang,nn_inds,tf.range(n_atoms)]
        if self.compute_dgvect:
            angf = tf_GdG_angular_mBP_1at
        else:
            angf = tf_G_angular_mBP_1at
        angf = partial(angf, Rs_ang, CTh, STh, eta_ang, zeta, \
                   Rc_ang, n_Ang_r, n_Ang_a, maskjk, n_atoms, n_species, nnmax)
        # Defining partial and signature for [without initial n_atoms]:
        angf = tf.function(angf, input_signature=[[
                        tf.TensorSpec(shape=[None,None],dtype=tf.float32),\
                        tf.TensorSpec(shape=[None,None,3],dtype=tf.float32),\
                        tf.TensorSpec(shape=[None,None],dtype=tf.float32),\
                        tf.TensorSpec(shape=[None,None],dtype=tf.int32),\
                        tf.TensorSpec(shape=[],dtype=tf.int32)]],\
                        jit_compile=True)

        # Both map_fn and vectorized_map work here:
        # Vect. seems to have better performance, so it is enabled here
        # But map uses less memory, so we might enable it with a switch in the future 
        if self.compute_dgvect:
            Gang, dGang = tf.vectorized_map(angf,input_stack)
        else:
            Gang = tf.vectorized_map(angf,input_stack)

        # Grad = tf.zeros([n_atoms,n_Rad])
        # Gang = tf.zeros([n_atoms,n_Ang])
        Gs = tf.concat([Grad,Gang], axis=1)
        padded_gs = tf.concat([Gs, tf.zeros([nats_padded-n_atoms,gsize])], axis=0)
        if not self.compute_dgvect:
            return padded_gs

        # dGrad = tf.zeros([n_atoms,n_Rad,n_atoms,3])
        # dGang = tf.zeros([n_atoms,n_Ang,n_atoms,3])
        dGs = tf.concat([dGrad,dGang], axis=1)
        padded_dgs = tf.concat([tf.reshape(dGs,[-1]), \
                             tf.zeros((nats_padded**2-n_atoms**2)*gsize*3)],axis=0)
        #Temp in place of gs
        # padded_gs = tf.random.normal([nats_padded,gsize])
        # padded_dgs = tf.random.normal([nats_padded*gsize*nats_padded*3])
        return [padded_gs, padded_dgs]

@tf.function(jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),\
                     tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_cutoffR(r_ij, r_c):
    return 0.5 * (1.0 + tf.cos(np.pi * r_ij / r_c))


@tf.function(jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),\
                     tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_dcutoffR(r_ij, r_c):
    return 0.5 * np.pi * tf.sin(np.pi * r_ij / r_c) / r_c


@tf.function(jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),\
                     tf.TensorSpec(shape=(1,1,1,None), dtype=tf.float32),\
                     tf.TensorSpec(shape=(1,1,1,None), dtype=tf.float32),\
                     tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_G_radial(r_ij, R_s, eta_rad, Rc_rad):
    return tf.exp(-eta_rad * (r_ij - R_s)**2) * tf_cutoffR(r_ij, Rc_rad)


@tf.function(jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None,None,None,1), dtype=tf.float32),\
                     tf.TensorSpec(shape=(1,1,1,None), dtype=tf.float32),\
                     tf.TensorSpec(shape=(1,1,1,None), dtype=tf.float32),\
                     tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_GdG_radial(r_ij, R_s, eta_rad, Rc_rad):
    sm = 1e-20
    Gauss = tf.exp(-eta_rad * (r_ij - R_s)**2)
    cutoff = tf_cutoffR(r_ij, Rc_rad)
    g_rad = Gauss * cutoff
    dg_rad = (tf_dcutoffR(r_ij, Rc_rad) + \
              (2.0 * eta_rad * cutoff * (r_ij - R_s))) *\
             Gauss / (r_ij + sm)
    return g_rad, dg_rad


@tf.function(jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),\
                     tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_cutoffA_1a(r_ij, r_c):
    return 0.5 * (1.0 + tf.cos(np.pi * r_ij / r_c))


@tf.function(jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),\
                     tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_dcutoffA_1a(r_ij, r_c):
    return 0.5 * np.pi * tf.sin(np.pi * r_ij / r_c) / r_c


# Version working on 1 atom at a time, for map_fn/vectorized_map
def tf_G_angular_mBP_1at(R_p, cs, ss, eta_ang, zeta, Rc_ang, n_Ang_r, n_Ang_a, mask2, \
                           n_atoms, n_species, nnmax, x):
    r = x[0]
    vecs = x[1]
    mask1 = x[2]
    nn_inds = x[3]
    n = x[4]
    # the correction to make second derivative continuous
    eps = 1e-3
    # Small value for normalization
    sm = 1e-20
    r_ij = tf.reshape(r,[n_species,1,nnmax,1,1,1])
    r_ik = tf.reshape(r,[1,n_species,1,nnmax,1,1])
    rij_vec = tf.reshape(vecs,[n_species,1,nnmax,1,1,1,3])
    rik_vec = tf.reshape(vecs,[1,n_species,1,nnmax,1,1,3])
    costheta_jik = tf.reduce_sum(rij_vec*rik_vec, axis=6) / (r_ij*r_ik + sm)
    mask = tf.reshape(mask1,[n_species,1,nnmax,1,1,1]) *\
           tf.reshape(mask1,[1,n_species,1,nnmax,1,1]) *\
           mask2
    # radial center of gaussian
    r_cent = (r_ij + r_ik) * 0.5 - R_p
    ct = costheta_jik  # cos_theta_jik
    st = tf.sqrt(1 - ct**2 + eps * ss**2)  # sin_theta_jik_approx
    # normalization of eps correction
    norm = 1.0 / (1. + tf.sqrt(1. + eps * ss**2))
    # components of G = Gauss x CosTerm x CutTerms
    Gauss = 2.0 * tf.exp(-eta_ang * r_cent**2)
    onepcos = 1.0 + ct * cs + st * ss
    CosTerm = (onepcos * norm)**zeta
    Cut = tf_cutoffA_1a(r, Rc_ang)
    Cut1 = tf.reshape(Cut,[n_species,1,nnmax,1,1,1])
    Cut2 = tf.reshape(Cut,[1,n_species,1,nnmax,1,1])
    G = mask * Gauss * CosTerm * Cut1 * Cut2
    G = tf.reshape(tf.reduce_sum(G,[2,3]),[n_species,n_species,-1])
    Gang = tf.concat([(2.0 - int(s1==s2)) * 0.5 * G[s1,s2] \
                       for s1 in range(n_species) \
                         for s2 in range(s1,n_species)],axis=0)
    return Gang


# Version working on 1 atom at a time, for map_fn/vectorized_map
def tf_GdG_angular_mBP_1at(R_p, cs, ss, eta_ang, zeta, Rc_ang, n_Ang_r, n_Ang_a, mask2, \
                           n_atoms, n_species, nnmax, x):
    r = x[0]
    vecs = x[1]
    mask1 = x[2]
    nn_inds = x[3]
    n = x[4]
    # the correction to make second derivative continuous
    eps = 1e-3
    # Small value for normalization
    sm = 1e-20
    r_ij = tf.reshape(r,[n_species,1,nnmax,1,1,1])
    r_ik = tf.reshape(r,[1,n_species,1,nnmax,1,1])
    invr = 1.0 / (r + sm)
    invr_ij = tf.reshape(invr,[n_species,1,nnmax,1,1,1])
    invr_ik = tf.reshape(invr,[1,n_species,1,nnmax,1,1])
    rij_vec = tf.reshape(vecs,[n_species,1,nnmax,1,1,1,3])
    rik_vec = tf.reshape(vecs,[1,n_species,1,nnmax,1,1,3])
    costheta_jik = tf.reduce_sum(rij_vec*rik_vec, axis=6) / (r_ij*r_ik + sm)
    mask = tf.reshape(mask1,[n_species,1,nnmax,1,1,1]) *\
           tf.reshape(mask1,[1,n_species,1,nnmax,1,1]) *\
           mask2
    # radial center of gaussian
    r_cent = (r_ij + r_ik) * 0.5 - R_p
    ct = costheta_jik  # cos_theta_jik
    st = tf.sqrt(1 - ct**2 + eps * ss**2)  # sin_theta_jik_approx
    # normalization of eps correction
    norm = 1.0 / (1. + tf.sqrt(1. + eps * ss**2))
    # components of G = Gauss x CosTerm x CutTerms
    Gauss = 2.0 * tf.exp(-eta_ang * r_cent**2)
    onepcos = 1.0 + ct * cs + st * ss
    CosTerm = (onepcos * norm)**zeta
    Cut = tf_cutoffA_1a(r, Rc_ang)
    Cut1 = tf.reshape(Cut,[n_species,1,nnmax,1,1,1])
    Cut2 = tf.reshape(Cut,[1,n_species,1,nnmax,1,1])
    dCut = tf_dcutoffA_1a(r, Rc_ang)
    dCut1 = tf.reshape(dCut,[n_species,1,nnmax,1,1,1])
    dCut2 = tf.reshape(dCut,[1,n_species,1,nnmax,1,1])
    G = mask * Gauss * CosTerm
    Cut1xCut2 = Cut1 * Cut2
    # derivatives
    # dGi/drj = A r_ij + B r_ik
    dcos = zeta * Cut1xCut2 / onepcos * (cs - ss * ct / st)
    etarcut = eta_ang * r_cent * Cut1xCut2
    # A
    dG1 = - G * invr_ij * \
        (etarcut
         + dcos * ct * invr_ij
         + dCut1 * Cut2)
    # B
    dG2 = (G * dcos) * invr_ij * invr_ik
    # A'
    dG3 = - G * invr_ik *\
        (etarcut
         + dcos * ct * invr_ik
         + dCut2 * Cut1)
    # Fixing the G values
    G *= Cut1xCut2
    G = tf.reshape(tf.reduce_sum(G,[2,3]),[n_species,n_species,-1])
    Gang = tf.concat([(2.0 - int(s1==s2)) * 0.5 * G[s1,s2] \
                       for s1 in range(n_species) \
                         for s2 in range(s1,n_species)],axis=0)
    jterm = tf.expand_dims(dG1,-1)*rij_vec + \
            tf.expand_dims(dG2,-1)*rik_vec
    kterm = tf.expand_dims(dG2,-1)*rij_vec + \
            tf.expand_dims(dG3,-1)*rik_vec
    # Summing all terms for derivative w.r.t. i
    dGang = tf.concat([(-2.0 + int(s1==s2)) * 0.5 * \
                        tf.reduce_sum(jterm[s1,s2]+kterm[s1,s2], axis=[0,1]) \
                          for s1 in range(n_species) \
                            for s2 in range(s1,n_species)],axis=0)
    # Putting only at the index of the atom
    dGang = tf.reshape(dGang,[-1,1,3])*tf.reshape(tf.one_hot(n,n_atoms),[1,n_atoms,1])
    # Adding all the derivatives wrt j and k
    # For each nn we scatter to the atom in the right position full list
    # Then we reshuffle this to be the right contribution in the dG matrix
    ind1 = tf.tile(tf.reshape(tf.range(n_species),[-1,1,1]),[1,n_species,nnmax])
    ind2 = tf.tile(tf.reshape(tf.range(n_species),[1,-1,1]),[n_species,1,nnmax])
    ind3 = tf.tile(tf.reshape(nn_inds,[n_species,1,nnmax]),[1,n_species,1])
    inds = tf.stack([ind1,ind2,ind3],axis=-1)
    dGangjke = tf.zeros([n_species,n_species,n_atoms,n_Ang_r,n_Ang_a,3])
    dGangjke = tf.tensor_scatter_nd_add(dGangjke,inds,tf.reduce_sum(jterm,axis=3))
    # k part (is there a way to do more together?)
    ind3 = tf.tile(tf.reshape(nn_inds,[1,n_species,nnmax]),[n_species,1,1])
    inds = tf.stack([ind1,ind2,ind3],axis=-1)
    dGangjke = tf.tensor_scatter_nd_add(dGangjke,inds,tf.reduce_sum(kterm,axis=2))
    dGangjk = tf.concat([(2.0 - int(s1==s2)) * 0.5 * dGangjke[s1,s2] \
                        for s1 in range(n_species) \
                            for s2 in range(s1,n_species)],axis=0)
    dGangjk = tf.reshape(dGangjk,[-1,n_atoms,n_Ang_r*n_Ang_a,3])
    dGangjk = tf.reshape(tf.transpose(dGangjk, perm=[0,2,1,3]),[-1,n_atoms,3])
    dGang = dGang + dGangjk
    # Gang = tf.zeros([n_Ang_r*n_Ang_a])
    # dGang = tf.zeros([n_Ang_r*n_Ang_a,n_atoms,3])
    return Gang, dGang
