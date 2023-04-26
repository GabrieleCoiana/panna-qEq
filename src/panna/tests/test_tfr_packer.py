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
import sys
import io

import numpy as np

from panna.tfr_packer import main as tfr_packer
from panna.tests.utils import ROOT_FOLDER


class Test_Tft_Packer(unittest.TestCase):
    def setUp(self):
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
        os.symlink(self.cwd + '/tests/data/tfr_packer/gvectors',
                   'gvectors')
        config_file = self.cwd + '/tests/data/tfr_packer/tfr.ini'

        # Run the tfr_packer, suppressing output
        suppress_text = io.StringIO()
        sys.stdout = suppress_text 
        with unittest.mock.patch('sys.argv', ['program_name', '-c', config_file]):
            tfr_packer()
        sys.stdout = sys.__stdout__

        # Just checking for existence of the file
        out = os.listdir('./tfr')
        np.testing.assert_string_equal(out[0], 'test-1-1.tfrecord')


if __name__ == '__main__':
    unittest.main()
