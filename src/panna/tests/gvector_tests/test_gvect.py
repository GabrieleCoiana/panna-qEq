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
import configparser

import numpy as np

if __package__ != 'panna':
    from lib import ExampleJsonWrapper
    from gvector import GvectBP, GvectmBP
    from lib.parser_callable import converters
else:
    from .lib import ExampleJsonWrapper
    from .gvector import GvectBP, GvectmBP
    from .lib.parser_callable import converters


class TestGvector(unittest.TestCase):
    def test_1(self):
        """
        test mBP with derivative
        """
        example_location = os.path.join('tests', 'data', 'gvector', 'examples')
        ref_location = os.path.join('tests', 'data', 'gvector', 'references')
        config_file = os.path.join('tests', 'data', 'gvector', 'gvect_mBP1.ini')
        gvect_func = GvectmBP(
            compute_dgvect=True,
            species='H, C, N, O',
            pbc_directions=None,
            sparse_dgvect=False)
        config = configparser.ConfigParser(converters=converters)
        config.read(config_file)
        gvect_func.parse_parameters(config['GVECT_PARAMETERS'])

        for example_file in os.listdir(example_location):
            example_number = example_file.split('.')[0]
            example = ExampleJsonWrapper(
                os.path.join(example_location, example_file),
                gvect_func.species_idx_2str)

            output = gvect_func(
                example.key, example.angstrom_positions,
                example.species_indexes, example.angstrom_lattice_vectors)
            ref_gvects = np.load(
                os.path.join(ref_location, example_number + '_gvects.npy'))
            ref_dgvects = np.load(
                os.path.join(ref_location, example_number + '_dgvects.npy'))

            # test g
            np.testing.assert_array_almost_equal(output['Gvect'], ref_gvects)

            # test dg
            dgvects = np.asarray(output['dGvect'])
            np.testing.assert_array_almost_equal(dgvects, ref_dgvects)


    def test_2(self):
        """
        test mBP without derivative
        """
        example_location = os.path.join('tests', 'data', 'gvector', 'examples')
        ref_location = os.path.join('tests', 'data', 'gvector', 'references')
        config_file = os.path.join('tests', 'data', 'gvector', 'gvect_mBP1.ini')
        gvect_func = GvectmBP(
            compute_dgvect=False,
            species='H, C, N, O',
            pbc_directions=None,
            sparse_dgvect=False)
        config = configparser.ConfigParser(converters=converters)
        config.read(config_file)
        gvect_func.parse_parameters(config['GVECT_PARAMETERS'])

        for example_file in os.listdir(example_location):
            example_number = example_file.split('.')[0]
            example = ExampleJsonWrapper(
                os.path.join(example_location, example_file),
                gvect_func.species_idx_2str)

            output = gvect_func(
                example.key, example.angstrom_positions,
                example.species_indexes, example.angstrom_lattice_vectors)
            ref_gvects = np.load(
                os.path.join(ref_location, example_number + '_gvects.npy'))

            # test g
            np.testing.assert_array_almost_equal(output['Gvect'], ref_gvects)

    def test_3(self):
        """
        test BP without derivative
        """
        example_location = os.path.join('tests', 'data', 'gvector', 'examples')
        ref_location = os.path.join('tests', 'data', 'gvector', 'referencesBP')
        config_file = os.path.join('tests', 'data', 'gvector', 'gvect_BP.ini')
        gvect_func = GvectBP(
            compute_dgvect=False,
            species='H, C, N, O',
            pbc_directions=None,
            sparse_dgvect=False)
        config = configparser.ConfigParser(converters=converters)
        config.read(config_file)
        gvect_func.parse_parameters(config['GVECT_PARAMETERS'])

        for example_file in os.listdir(example_location):
            example_number = example_file.split('.')[0]
            example = ExampleJsonWrapper(
                os.path.join(example_location, example_file),
                gvect_func.species_idx_2str)

            output = gvect_func(
                example.key, example.angstrom_positions,
                example.species_indexes, example.angstrom_lattice_vectors)
            ref_gvects = np.load(
                os.path.join(ref_location, example_number + '_gvects.npy'))

            # test g
            np.testing.assert_array_almost_equal(output['Gvect'], ref_gvects)

if __name__ == '__main__':
    unittest.main()
