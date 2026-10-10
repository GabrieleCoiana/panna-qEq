###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import unittest
import os
import shutil
import numpy as np

if __package__ != 'panna':
    from gvector.binary_encoder import binary_encoder
    from lib.example_bin import load_example
    from tests.utils import ROOT_FOLDER
else:
    from .gvector.binary_encoder import binary_encoder
    from .lib.example_bin import load_example
    from .tests.utils import ROOT_FOLDER


class TestIOBinary(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    def test_1(self):
        """ create a binary file
        no derivative
        no forces
        no per_atom quantity
        """
        test_data_dir = ROOT_FOLDER + 'tests/binary_io_tmp_1'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)
        os.chdir(test_data_dir)
        # create dataset
        example_dict = {
            'key': 'test_file.example',
            'species': np.asarray([0, 0, 1, 1, 2]),
            'Gvect': np.arange(50).reshape(5, 10),
            'E': 100
        }
        binary_encoder(example_dict, '.')
        example = load_example(example_dict['key'] + '.bin')
        np.testing.assert_equal(example_dict['E'], example.true_energy)
        np.testing.assert_array_equal(example_dict['species'],
                                      example.species_vector)
        np.testing.assert_array_equal(example_dict['Gvect'], example.gvects)
        with self.assertRaises(Exception):
            example.dgvects
        with self.assertRaises(Exception):
            example.forces
        with self.assertRaises(Exception):
            example.per_atom_quantity

    def test_2(self):
        """ create a binary file
        derivative and forces
        no per_atom quantity
        """
        test_data_dir = ROOT_FOLDER + 'tests/binary_io_tmp_2'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)
        os.chdir(test_data_dir)
        # create dataset
        example_dict = {
            'key': 'test_file_2.example',
            'species': np.asarray([0, 0, 1, 1, 2]),
            'Gvect': np.arange(50).reshape(5, 10),
            'dGvect': np.arange(750).reshape(5, 10, 15),
            'forces': np.arange(15).reshape(5, 3),
            'E': 100
        }
        binary_encoder(example_dict, '.')
        example = load_example(example_dict['key'] + '.bin')
        np.testing.assert_equal(example_dict['E'], example.true_energy)
        np.testing.assert_array_equal(example_dict['species'],
                                      example.species_vector)
        np.testing.assert_array_equal(example_dict['Gvect'], example.gvects)
        np.testing.assert_array_equal(example_dict['dGvect'], example.dgvects)
        np.testing.assert_array_equal(example_dict['forces'].flatten(),
                                      example.forces)
        with self.assertRaises(Exception):
            example.per_atom_quantity

    def test_3(self):
        """ create a binary file
        derivative and forces and per_atom quantity
        """
        test_data_dir = ROOT_FOLDER + 'tests/binary_io_tmp_3'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)
        os.chdir(test_data_dir)
        # create dataset
        example_dict = {
            'key': 'test_file_3.example',
            'species': np.asarray([0, 0, 1, 1, 2]),
            'Gvect': np.arange(50).reshape(5, 10),
            'dGvect': np.arange(750).reshape(5, 10, 15),
            'forces': np.arange(15).reshape(5, 3),
            'per_atom_quantity': np.arange(5),
            'E': 100
        }
        binary_encoder(example_dict, '.')
        example = load_example(example_dict['key'] + '.bin')
        np.testing.assert_equal(example_dict['E'], example.true_energy)
        np.testing.assert_array_equal(example_dict['species'],
                                      example.species_vector)
        np.testing.assert_array_equal(example_dict['Gvect'], example.gvects)
        np.testing.assert_array_equal(example_dict['dGvect'], example.dgvects)
        np.testing.assert_array_equal(example_dict['forces'].flatten(),
                                      example.forces)
        np.testing.assert_array_equal(example_dict['per_atom_quantity'],
                                      example.per_atom_quantity)

    def test_4(self):
        """ create a binary file
        per_atom quantity
        no derivative
        no forces
        """
        test_data_dir = ROOT_FOLDER + 'tests/binary_io_tmp_4'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)
        os.chdir(test_data_dir)
        # create dataset
        example_dict = {
            'key': 'test_file.example',
            'species': np.asarray([0, 0, 1, 1, 2]),
            'Gvect': np.arange(50).reshape(5, 10),
            'per_atom_quantity': np.arange(5),
            'E': 100
        }
        binary_encoder(example_dict, '.')
        example = load_example(example_dict['key'] + '.bin')
        np.testing.assert_equal(example_dict['E'], example.true_energy)
        np.testing.assert_array_equal(example_dict['species'],
                                      example.species_vector)
        np.testing.assert_array_equal(example_dict['Gvect'], example.gvects)
        np.testing.assert_array_equal(example_dict['per_atom_quantity'],
                                      example.per_atom_quantity)
        with self.assertRaises(Exception):
            example.dgvects
        with self.assertRaises(Exception):
            example.forces


if __name__ == '__main__':
    unittest.main()
