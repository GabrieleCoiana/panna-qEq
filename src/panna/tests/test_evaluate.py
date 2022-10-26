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

from evaluate import main as evaluate

from tests.utils import ROOT_FOLDER


class _DataContainer():
    pass


class Test_Evaluate(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        try:
            shutil.rmtree(self.test_data_dir, ignore_errors=True)
            pass
        except AttributeError:
            pass

    def test_1(self):
        """ Testing for training features
        """
        test_data_dir = ROOT_FOLDER + 'tests/test_eval_1'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir),
                          ignore_errors=True)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/evaluation/test_1/evaluate_data',
                   'evaluate_data')
        os.symlink(self.cwd + '/tests/data/evaluation/test_1/saved_networks',
                   'saved_networks')

        # buiding the input
        io_parameters = _DataContainer()
        io_parameters.data_dir = './evaluate_data'
        io_parameters.train_dir = './train_output'
        io_parameters.eval_dir = './evaluate_output'

        io_parameters.example_format = 'TFR'

        io_parameters.networks_format = 'PANNA'
        io_parameters.networks_folder = './saved_networks'

        io_parameters.number_of_process = 1

        validation_parameters = _DataContainer()
        validation_parameters.batch_size = -1
        validation_parameters.single_step = True
        validation_parameters.compute_forces = False

        validation_parameters.step_number = None
        validation_parameters.subsampling = None
        validation_parameters.add_offset = False

        parallel_param = _DataContainer()
        parallel_param.number_of_process = 4

        tfr_parameters = _DataContainer()
        tfr_parameters.sparse_derivatives = False
        tfr_parameters.g_size = 384

        parameters = (io_parameters, parallel_param, validation_parameters,
                      tfr_parameters)

        # Run the evaluation
        evaluate(parameters)

        # Read the output
        outE = []
        with open("./evaluate_output/energies.dat", "r") as f:
            f.readline()
            for line in f:
                line = line.split(' ')
                outE.append(float(line[3]))
        # Checking against reference
        # [values and threshold hardcoded for now... better ideas?]
        refE = np.asarray([-9.037636215782056, -9.42446958788558])
        np.testing.assert_allclose(np.asarray(outE), refE)

    def test_2(self):
        """ Testing for training features
        """
        test_data_dir = ROOT_FOLDER + './tests/test_eval_2'
        self.test_data_dir = test_data_dir
        if os.path.isdir(test_data_dir):
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir),
                          ignore_errors=True)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/evaluation/test_2/evaluate_data',
                   'evaluate_data')
        os.symlink(self.cwd + '/tests/data/evaluation/test_2/saved_networks',
                   'saved_networks')

        # buiding the input
        io_parameters = _DataContainer()
        io_parameters.data_dir = './evaluate_data'
        io_parameters.train_dir = './train_output'
        io_parameters.eval_dir = './evaluate_output'

        io_parameters.example_format = 'TFR'

        io_parameters.networks_format = 'PANNA'
        io_parameters.networks_folder = './saved_networks'

        io_parameters.number_of_process = 1

        validation_parameters = _DataContainer()
        validation_parameters.batch_size = -1
        validation_parameters.single_step = True
        validation_parameters.compute_forces = False

        validation_parameters.step_number = None
        validation_parameters.subsampling = None
        validation_parameters.add_offset = False

        parallel_param = _DataContainer()
        parallel_param.number_of_process = 4

        tfr_parameters = _DataContainer()
        tfr_parameters.sparse_derivatives = False
        tfr_parameters.g_size = 384
        parameters = (io_parameters, parallel_param, validation_parameters,
                      tfr_parameters)

        # Run the evaluation
        evaluate(parameters)
        # Read the output
        outE = []
        with open("./evaluate_output/energies.dat", "r") as f:
            f.readline()
            for line in f:
                outE.append(float(line.split(' ')[3]))

        refE = np.asarray([
            -0.0036183453123613,
            -0.4847124718749001,
            0.1212132484374138,
        ])
        # this is a little too loose, why?
        np.testing.assert_allclose(np.asarray(outE), refE, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
