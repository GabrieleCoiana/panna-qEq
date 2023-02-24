###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import configparser
import train
import shutil
import unittest
import numpy as np
import sys
import io


if __package__ != 'panna':
    from train import main as train
    from tests.utils import ROOT_FOLDER
else:
    from .train import main as train
    from .tests.utils import ROOT_FOLDER


class Test_train(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        # comment this line to not delete the outputs!!
        try:
            shutil.rmtree(self.test_data_dir, ignore_errors=True)
        except AttributeError:
            pass

    def test_1(self):
        """ Testing for training from saved network
        """
        test_data_dir = ROOT_FOLDER + 'tests/test_train_1'

        self.test_data_dir = test_data_dir
        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)
        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/train/train_data',
                   'train_data')
        os.symlink(self.cwd + '/tests/data/train/starting_network',
                   'starting_network')
        config_file = self.cwd + '/tests/data/train/train1.ini'

        # Run the training, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text 
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            train()
        sys.stdout = sys.__stdout__

        # Read the output
        outE = []
        with open("./delta_e.dat") as f:
            for Es in f:
                tmp = [float(x) for x in Es.split()]
                tmp.sort()
                outE.append(np.asarray(tmp))
        # Checking against reference
        # values and threshold hardcoded for now
        # 1e-5 allows for small GPU/CPU inconsistencies
        refE_0 = np.asarray([-0.08596611, 0.13045311])
        np.testing.assert_allclose(outE[0], refE_0, atol=1e-5)
        np.testing.assert_allclose(outE[-1], np.zeros(2), atol=1e-5)


    def test_2(self):
        """ Testing for training from checkpoint
        """
        test_data_dir = ROOT_FOLDER + 'tests/test_train_2'

        self.test_data_dir = test_data_dir
        if os.path.isdir(test_data_dir):
            shutil.rmtree(self.test_data_dir)
        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/train/train_data',
                   'train_data')
        shutil.copytree(self.cwd + '/tests/data/train/train', './train')
        config_file = self.cwd + '/tests/data/train/train2.ini'

        # Run the training, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text 
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file,
                                              '--communication_port', '22223']):
            train()
        sys.stdout = sys.__stdout__

        # Read the output
        outE = []
        with open("./delta_e.dat") as f:
            for Es in f:
                tmp = [float(x) for x in Es.split()]
                tmp.sort()
                outE.append(np.asarray(tmp))
        # Checking against reference
        # values and threshold hardcoded for now
        # 1e-5 allows for small GPU/CPU inconsistencies
        refE_0 = np.asarray([-0.08596611, 0.13045311])
        np.testing.assert_allclose(outE[0], refE_0, atol=1e-5)
        np.testing.assert_allclose(outE[-1], np.zeros(2), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
