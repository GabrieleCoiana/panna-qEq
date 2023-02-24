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

if __package__ != 'panna':
    from evaluate import main as evaluate
    from tests.utils import ROOT_FOLDER
else:
    from .evaluate import main as evaluate
    from .tests.utils import ROOT_FOLDER


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
        """ Testing evaluation from checkpoint
        """
        test_data_dir = ROOT_FOLDER + 'tests/test_eval_1'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir),
                          ignore_errors=True)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/evaluation/evaluate_data',
                   'evaluate_data')
        os.symlink(self.cwd + '/tests/data/evaluation/train',
                   'train')
        config_file = self.cwd + '/tests/data/evaluation/evaluate1.ini'

        # Run the evaluation, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            evaluate()
        sys.stdout = sys.__stdout__

        # Read the output
        outE = []
        with open("./evaluate_output/0.dat", "r") as f:
            f.readline()
            for line in f:
                line = line.split(' ')
                outE.append(float(line[3]))
        # Checking against reference
        refE = np.asarray([-2078.2035642660258, -2078.37483961411])
        np.testing.assert_allclose(np.asarray(outE), refE)

    def test_2(self):
        """ Testing evaluation from saved network
        """
        test_data_dir = ROOT_FOLDER + 'tests/test_eval_2'
        self.test_data_dir = test_data_dir

        if os.path.isdir(test_data_dir):
            shutil.rmtree(os.path.join(self.cwd, self.test_data_dir),
                          ignore_errors=True)

        os.makedirs(test_data_dir)

        os.chdir(test_data_dir)
        os.symlink(self.cwd + '/tests/data/evaluation/evaluate_data',
                   'evaluate_data')
        os.symlink(self.cwd + '/tests/data/evaluation/saved_network',
                   'saved_network')
        config_file = self.cwd + '/tests/data/evaluation/evaluate2.ini'

        # Run the evaluation, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            evaluate()
        sys.stdout = sys.__stdout__

        # Read the output
        outE = []
        with open("./evaluate_output/PANNA_scaffold.dat", "r") as f:
            f.readline()
            for line in f:
                line = line.split(' ')
                outE.append(float(line[3]))
        # Checking against reference
        refE = np.asarray([-2078.2035642660258, -2078.37483961411])
        np.testing.assert_allclose(np.asarray(outE), refE)


if __name__ == '__main__':
    unittest.main()
