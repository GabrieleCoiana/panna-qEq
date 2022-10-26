###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import json
import os
import unittest

import numpy as np

from lib import ExampleJsonWrapper
from gvector import GvectBP, GvectmBP


class TestGvector(unittest.TestCase):
    def test_1(self):
        """
        test mBP with derivative
        """
        example_location = os.path.join('tests', 'data', 'gvector', 'examples')
        ref_location = os.path.join('tests', 'data', 'gvector', 'references')
        gvect_func = GvectmBP(
            compute_dgvect=True,
            species='H, C, N, O',
            param_unit='angstrom',
            pbc_directions=None,
            Rc_rad=4.6,
            Rs0_rad=0.5,
            RsN_rad=16,
            eta_rad=16,
            Rc_ang=3.1,
            ThetasN=8,
            Rs0_ang=0.5,
            RsN_ang=4,
            eta_ang=16,
            zeta=50,
            sparse_dgvect=False)

        for example_file in os.listdir(example_location):
            example_number = example_file.split('.')[0]
            example = ExampleJsonWrapper(
                os.path.join(example_location, example_file),
                gvect_func.species_idx_2str)

            _dummy, gvects, dgvects = gvect_func(
                example.key, example.angstrom_positions,
                example.species_indexes, example.angstrom_lattice_vectors)
            ref_gvects = np.load(
                os.path.join(ref_location, example_number + '_gvects.npy'))
            ref_dgvects = np.load(
                os.path.join(ref_location, example_number + '_dgvects.npy'))

            # test g
            np.testing.assert_array_almost_equal(gvects, ref_gvects)

            # test dg, shape need to be adapted because the sotored npy array
            # are a little different
            dgvects = np.asarray(dgvects)
            dgvects = dgvects.reshape(*dgvects.shape[:-2],
                                      dgvects.shape[-1] * dgvects.shape[-2])
            np.testing.assert_array_almost_equal(dgvects, ref_dgvects)

    def test_2(self):
        """
        test mBP without derivative
        """
        example_location = os.path.join('tests', 'data', 'gvector', 'examples')
        ref_location = os.path.join('tests', 'data', 'gvector', 'references')
        gvect_func = GvectmBP(
            compute_dgvect=False,
            species='H, C, N, O',
            param_unit='angstrom',
            pbc_directions=None,
            Rc_rad=4.6,
            Rs0_rad=0.5,
            RsN_rad=16,
            eta_rad=16,
            Rc_ang=3.1,
            ThetasN=8,
            Rs0_ang=0.5,
            RsN_ang=4,
            eta_ang=16,
            zeta=50,
            sparse_dgvect=False)

        for example_file in os.listdir(example_location):
            example_number = example_file.split('.')[0]
            example = ExampleJsonWrapper(
                os.path.join(example_location, example_file),
                gvect_func.species_idx_2str)

            _dummy, gvects = gvect_func(
                example.key, example.angstrom_positions,
                example.species_indexes, example.angstrom_lattice_vectors)
            ref_gvects = np.load(
                os.path.join(ref_location, example_number + '_gvects.npy'))

            # test g
            np.testing.assert_array_almost_equal(gvects, ref_gvects)

    def test_3(self):
        """
        test BP without derivative
        This test is very incomplete, there is no reference, it just run the
        code
        """
        example_location = os.path.join('tests', 'data', 'gvector', 'examples')
        # ref_location = not available
        gvect_func = GvectBP(
            compute_dgvect=False,
            species='H, C, N, O',
            param_unit='angstrom',
            pbc_directions=None,
            Rc=4.6,
            Rs0=0.5,
            RsN=16,
            eta=[9, 12],
            eta_ang=[9, 12, 20],
            zeta=[40.7, 52.3, 60],
            sparse_dgvect=False)

        for example_file in os.listdir(example_location):
            # example_number = example_file.split('.')[0]
            example = ExampleJsonWrapper(
                os.path.join(example_location, example_file),
                gvect_func.species_idx_2str)

            _dummy, gvects = gvect_func(
                example.key, example.angstrom_positions,
                example.species_indexes, example.angstrom_lattice_vectors)
            # ref_gvects = not available

            # test g
            # not available

if __name__ == '__main__':
    unittest.main()
