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
import logging
logging.getLogger('panna').setLevel(logging.ERROR)

from panna.train import main as train
from panna.tests.utils import ROOT_FOLDER

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
        out = []
        with open("./metrics.dat") as f:
            for o in f.readlines()[1:]:
                out.append(float(o.split()[1]))
        # Checking against reference
        # values and threshold hardcoded for now
        # 1e-4 allows for small GPU/CPU inconsistencies
        # print(out)
        ref = np.asarray([0.0012613933067768812, 0.0, 8.138021075865254e-05, \
                           0.0, 8.138021075865254e-05])
        np.testing.assert_allclose(out, ref, atol=1e-4)


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
        out = []
        with open("./train/metrics.dat") as f:
            for o in f.readlines()[1:]:
                out.append(float(o.split()[1]))
        # Checking against reference
        # values and threshold hardcoded for now
        # 1e-4 allows for small GPU/CPU inconsistencies
        # print(out)
        ref = np.asarray([0.4349365234375, 0.296142578125, 0.1971435546875, \
                          0.1262613981962204, 0.0771891325712204])
        np.testing.assert_allclose(out, ref, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
