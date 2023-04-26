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
import logging
logging.getLogger('panna').setLevel(logging.ERROR)

from panna.evaluate import main as evaluate
from panna.tests.utils import ROOT_FOLDER


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
        os.symlink(self.cwd + '/tests/data/evaluation/train1.ini',
                   'train1.ini')
        config_file = self.cwd + '/tests/data/evaluation/evaluate1.ini'

        # Run the evaluation, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            evaluate()
        sys.stdout = sys.__stdout__

        # Read the output
        with open("./evaluate_output/epoch_6_step_600.dat", "r") as f:
            out = [float(l.split()[3]) for l in f.readlines()[1:]]
        out = np.sort(np.asarray(out))
        # Checking against reference
        ref = np.asarray([-2078.50537109375, -2078.11767578125])
        np.testing.assert_allclose(out, ref, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
