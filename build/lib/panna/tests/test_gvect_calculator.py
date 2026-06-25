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
import sys
import io


from panna.lib.example_bin import load_example
from panna.gvect_calculator import main as gvect_calculator
from panna.tests.utils import ROOT_FOLDER


class ParameterContainer():
    """empty class
    """


class Test_Gvector_Calculator(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    # def test_1(self):
    #     '''
    #       test: mBP with derivatives
    #     '''
    #     test_data_dir = ROOT_FOLDER + '/tests/gvect_calculator_1'

    #     self.test_data_dir = test_data_dir

    #     if os.path.isdir(test_data_dir):
    #         shutil.rmtree(self.test_data_dir)

    #     os.makedirs(test_data_dir)

    #     os.chdir(test_data_dir)
    #     os.symlink(self.cwd + '/tests/data/gvector_calculator/examples',
    #                'examples')
    #     os.symlink(self.cwd + '/tests/data/gvector_calculator/bin_references',
    #                'bin_references')
    #     config_file = self.cwd + '/tests/data/gvector_calculator/gvect_mBP1.ini'

    #     # Run the gvect calculator, suppressing output
    #     suppress_text = io.StringIO()
    #     sys.stdout = suppress_text 
    #     with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
    #         gvect_calculator()
    #     sys.stdout = sys.__stdout__

    #     example_files_reference = os.listdir('bin_references')
    #     example_files_computed = os.listdir(os.path.join(test_data_dir, 'bin'))
    #     example_files_reference.sort()
    #     example_files_computed.sort()
    #     #compute difference
    #     for example_file_reference, example_file_computed in zip(
    #             example_files_reference, example_files_computed):
    #         example_computed = load_example(
    #             os.path.join(test_data_dir, 'bin',
    #                          example_file_computed))
    #         example_reference = load_example(
    #             os.path.join('bin_references', example_file_reference))
    #         # test g
    #         np.testing.assert_array_equal(example_computed.gvects,
    #                                       example_reference.gvects)
    #         # test dg
    #         np.testing.assert_array_equal(example_computed.dgvects,
    #                                       example_reference.dgvects)

    def test_2(self):
        '''
          test: mBP without derivatives
        '''
        test_data_dir = ROOT_FOLDER + '/tests/gvect_calculator_2'

        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/gvector_calculator/examples',
                   'examples')
        os.symlink(self.cwd + '/tests/data/gvector_calculator/bin_references',
                   'bin_references')
        config_file = self.cwd + '/tests/data/gvector_calculator/gvect_mBP2.ini'

        # Run the gvect calculator, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text 
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            gvect_calculator()
        sys.stdout = sys.__stdout__

        example_files_reference = os.listdir('bin_references')
        example_files_computed = os.listdir(os.path.join(test_data_dir, 'bin'))
        example_files_reference.sort()
        example_files_computed.sort()
        #compute difference
        for example_file_reference, example_file_computed in zip(
                example_files_reference, example_files_computed):
            example_computed = load_example(
                os.path.join(test_data_dir, 'bin',
                             example_file_computed))
            example_reference = load_example(
                os.path.join('bin_references', example_file_reference))
            # test g
            np.testing.assert_array_equal(example_computed.gvects,
                                          example_reference.gvects)


    def test_3(self):
        '''
          test: BP without derivatives
        '''
        test_data_dir = ROOT_FOLDER + '/tests/gvect_calculator_3'

        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/gvector_calculator/examples',
                   'examples')
        os.symlink(self.cwd + '/tests/data/gvector_calculator/bin_referencesBP',
                   'bin_referencesBP')
        config_file = self.cwd + '/tests/data/gvector_calculator/gvect_BP.ini'

        # Run the gvect calculator, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text 
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            gvect_calculator()
        sys.stdout = sys.__stdout__

        example_files_reference = os.listdir('bin_referencesBP')
        example_files_computed = os.listdir(os.path.join(test_data_dir, 'bin'))
        example_files_reference.sort()
        example_files_computed.sort()
        #compute difference
        for example_file_reference, example_file_computed in zip(
                example_files_reference, example_files_computed):
            example_computed = load_example(
                os.path.join(test_data_dir, 'bin',
                             example_file_computed))
            example_reference = load_example(
                os.path.join('bin_referencesBP', example_file_reference))
            # test g
            np.testing.assert_array_equal(example_computed.gvects,
                                          example_reference.gvects)


if __name__ == '__main__':
    unittest.main()
