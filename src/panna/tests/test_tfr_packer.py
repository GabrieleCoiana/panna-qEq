###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import shutil
import unittest

import numpy as np

import gvector
import tfr_packer

from tests.utils import ROOT_FOLDER


class Test_Tft_Packer(unittest.TestCase):
    def setUp(self):
        n_atoms = 10
        g_size = 5

        example_dict = {
            'key':
            'test_file.example',
            'species':
            np.arange(n_atoms).reshape(n_atoms),
            'Gvect':
            np.arange(n_atoms * g_size).reshape(n_atoms, g_size),
            'dGvect':
            np.arange(n_atoms * g_size * n_atoms * 3).reshape(
                n_atoms, g_size, n_atoms * 3),
            'forces':
            np.arange(n_atoms * 3),
            'per_atom_quantity':
            np.arange(n_atoms),
            'E':
            n_atoms * (n_atoms + 1) / 2,
        }
        self.example = example_dict
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    def test_tfr_packer(self):
        """
          test: creating binary
        """
        # TESTING FOR TFRS PREPARATION
        test_data_dir = ROOT_FOLDER + 'tests/test_tfr_packer'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        # create the example
        os.makedirs('bin')
        gvector.write_routine.compute_binary(self.example, 'bin')

        # compacting the data
        tfrs_parameters = type(
            'Tfrs', (object, ),
            dict(in_path='bin',
                 out_path='tfrs_w_d',
                 elements_per_file=2,
                 prefix='',
                 num_species=4,
                 derivatives=True,
                 sparse_derivatives=False,
                 per_atom_quantity=True))()

        # packing with derivatives
        tfr_packer.main(tfrs_parameters)

        # TODO compute difference
        # is this necessray? maybe


if __name__ == '__main__':
    unittest.main()
