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
import gvect_calculator
from gvector import GvectmBP
from gvector import GvectBP
from neuralnet.example import load_example
from tests.utils import ROOT_FOLDER


class ParameterContainer():
    """empty class
    """


class Test_Gvector_Calculator(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    def test_1(self):
        '''
          test: mBP with derivatives
        '''
        test_data_dir = ROOT_FOLDER + '/tests/gvect_calculator_1'

        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/gvector_calculator/examples',
                   'examples')
        os.symlink(self.cwd + '/tests/data/gvector_calculator/bin_references',
                   'bin_references')

        gvect_func = GvectmBP(compute_dgvect=True,
                              sparse_dgvect=False,
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
                              zeta=50)

        folder_parameters = type(
            'FolderParameters', (object, ),
            dict(
                input_json_dir='./examples',
                binary_out_dir=os.path.join(test_data_dir, 'bin'),
                log_dir=test_data_dir,
            ))()

        misc_parameters = ParameterContainer()
        misc_parameters.per_atom_energy = False
        misc_parameters.long_range = False

        gvect_calculator.main(gvect_func, folder_parameters, 1,
                              misc_parameters)

        example_files_reference = os.listdir('bin_references')
        example_files_computed = os.listdir(folder_parameters.binary_out_dir)
        example_files_reference.sort()
        example_files_computed.sort()
        #compute difference
        for example_file_reference, example_file_computed in zip(
                example_files_reference, example_files_computed):
            example_computed = load_example(
                os.path.join(folder_parameters.binary_out_dir,
                             example_file_computed))
            example_reference = load_example(
                os.path.join('bin_references', example_file_reference))
            # test g
            np.testing.assert_array_equal(example_computed.gvects,
                                          example_reference.gvects)
            # test dg
            np.testing.assert_array_equal(example_computed.dgvects,
                                          example_reference.dgvects)

    def test_2(self):
        '''
          test: BP version of the G
        '''
        test_data_dir = ROOT_FOLDER + 'tests/gvect_calculator_2'

        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)
        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/gvector_calculator/examples',
                   'examples')
        #os.symlink('/tests/data/gvector_calculator/bin_references',
        #           'bin_references')

        test_data_dir = 'test_gvect_prep'
        gvect_func = GvectBP(compute_dgvect=False,
                             sparse_dgvect=False,
                             species='H, C, N, O',
                             param_unit='angstrom',
                             pbc_directions=None,
                             Rc=4.6,
                             Rs0=0.5,
                             RsN=16,
                             eta=[9, 12],
                             eta_ang=[9, 12, 20],
                             zeta=[40.7, 52.3, 60])

        folder_parameters = type(
            'FolderParameters', (object, ),
            dict(
                input_json_dir='./examples',
                binary_out_dir=os.path.join(test_data_dir, 'bin'),
                log_dir=test_data_dir,
            ))()

        misc_parameters = ParameterContainer()
        misc_parameters.per_atom_energy = False
        misc_parameters.long_range = False

        gvect_calculator.main(gvect_func, folder_parameters, 1,
                              misc_parameters)
        # TODO insert real tests when available
        # example_files_reference = os.listdir('bin_references')
        # example_files_computed = os.listdir(folder_parameters.binary_out_dir)
        # example_files_reference.sort()
        # example_files_computed.sort()
        #compute difference
        #for example_file_reference, example_file_computed in zip(
        #        example_files_reference, example_files_computed):
        #    example_computed = load_example(
        #        os.path.join(folder_parameters.binary_out_dir,
        #                     example_file_computed),
        #        gvect_func.number_of_species,
        #        derivatives=True)
        #    example_reference = load_example(
        #        os.path.join(folder_parameters.binary_out_dir,
        #                     example_file_reference),
        #        gvect_func.number_of_species,
        #        derivatives=True)
        #    # test g
        #    np.testing.assert_array_equal(example_computed.gvects,
        #                                  example_reference.gvects)
        #    # test dg
        #    np.testing.assert_array_equal(example_computed.dgvects,
        #                                  example_reference.dgvects)

    def test_3(self):
        '''
          test: mBP without derivatives
        '''
        test_data_dir = ROOT_FOLDER + 'tests/gvect_calculator_3'
        self.test_data_dir = test_data_dir
        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/gvector_calculator/examples',
                   'examples')
        os.symlink(self.cwd + '/tests/data/gvector_calculator/bin_references',
                   'bin_references')

        gvect_func = GvectmBP(compute_dgvect=False,
                              sparse_dgvect=False,
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
                              zeta=50)

        folder_parameters = type(
            'FolderParameters', (object, ),
            dict(
                input_json_dir='./examples',
                binary_out_dir=os.path.join(test_data_dir, 'bin'),
                log_dir=test_data_dir,
            ))()

        misc_parameters = ParameterContainer()
        misc_parameters.per_atom_energy = False
        misc_parameters.long_range = False

        gvect_calculator.main(gvect_func, folder_parameters, 1,
                              misc_parameters)

        example_files_reference = os.listdir('bin_references')
        example_files_computed = os.listdir(folder_parameters.binary_out_dir)
        example_files_reference.sort()
        example_files_computed.sort()
        #compute difference
        for example_file_reference, example_file_computed in zip(
                example_files_reference, example_files_computed):
            example_computed = load_example(
                os.path.join(folder_parameters.binary_out_dir,
                             example_file_computed))
            example_reference = load_example(
                os.path.join('bin_references', example_file_reference))
            # test g
            np.testing.assert_array_equal(example_computed.gvects,
                                          example_reference.gvects)


if __name__ == '__main__':
    unittest.main()
