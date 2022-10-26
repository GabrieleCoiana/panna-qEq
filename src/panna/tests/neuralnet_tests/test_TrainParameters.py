###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import configparser
import logging
import unittest
import unittest.mock as mock
from functools import partial

import numpy as np

from lib.errors import NetworkNotAvailableError, ShapeError
from neuralnet.trainparameters import parameter_file_parser
from tests.utils import section_mocker

# override all loggers output to only critical
# INFO and WARNING are full of useless quantity in testing
logging.disable(logging.CRITICAL)


class Test_TrainParams(unittest.TestCase):
    def setUp(self):
        # create a default mocker
        io_infomration = {
            'train_dir': 'checkpoint',
            'data_dir': 'bin',
            'input_format': 'TFR',
            'save_checkpoint_steps': '1',
            'max_ckpt_to_keep': '42',
            'log_frequency': '10'
        }

        training_parameters = {
            'batch_size': '100',
            'learning_rate': '.3',
            'max_steps': '1000'
        }
        data_information = {
            'n_atoms': '100',
            'atomic_sequence': 'H, C, N, O',
            'output_offset': '0.0, 1.7, 42, 9',
        }
        default_network = {
            'g_size': '32',
            'nn_type': 'a2aff',
            'architecture': '16:8:4:1',
            'trainable': '1:0:1:1'
        }

        self.io_information = io_infomration
        self.training_parameters = training_parameters
        self.data_information = data_information
        self.default_network = default_network

        # fast helper to mock entire sections
        self.mock_io_information = partial(
            section_mocker, section='IO_INFORMATION', options=io_infomration)
        self.mock_training_parameters = partial(
            section_mocker,
            section='TRAINING_PARAMETERS',
            options=training_parameters)
        self.mock_data_information = partial(
            section_mocker,
            section='DATA_INFORMATION',
            options=data_information)
        self.mock_default_network = partial(
            section_mocker, section='DEFAULT_NETWORK', options=default_network)

    def test_basic_mocking(self):
        # default personalizations
        # this change the variable declared in setUP but
        # that routine should be executed before each test
        # so there is no interference among tests
        # eg: self.io_information['train_dir'] = 'pippo'

        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information
        mock_default_network = self.mock_default_network

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                mock_default_network(self)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            *dummy, system_scaffold = parameter_file_parser('pippo')

    def test_extraction_of_a_network(self):
        """

        Extraction of a network with default, and change the offset
        """
        c = {
            'architecture': '16:8:4:1',
            'trainable': '1:1:1:1',
            'output_offset': '7'
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information
        mock_default_network = self.mock_default_network

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                mock_default_network(self)
                section_mocker(self, 'C', c)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            *dummy, system_scaffold = parameter_file_parser('pippo')
        self.assertEqual(system_scaffold['C'].offset, 7)
        self.assertEqual(system_scaffold['C'].feature_size,
                         int(self.default_network['g_size']))
        self.assertEqual(system_scaffold['H'].offset, 0)
        self.assertEqual(system_scaffold['H'].feature_size,
                         int(self.default_network['g_size']))

    def test_extraction_of_a_network_2(self):
        """

        Extraction of a network with default
         - change offset
         - change feature size
        this should trigger new network
        """
        c = {
            'g_size': '56',
            'architecture': '16:8:4:1',
            'trainable': '1:1:1:1',
            'output_offset': '7'
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information
        mock_default_network = self.mock_default_network

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                mock_default_network(self)
                section_mocker(self, 'C', c)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            *dummy, system_scaffold = parameter_file_parser('pippo')
        self.assertEqual(system_scaffold['C'].offset, 7)
        self.assertEqual(system_scaffold['C'].feature_size, int(c['g_size']))
        self.assertEqual(system_scaffold['H'].offset, 0)
        self.assertEqual(system_scaffold['H'].feature_size,
                         int(self.default_network['g_size']))
        for i, j in zip(system_scaffold['C'].layers_shaping[0],
                        [int(c['g_size']), 16]):
            self.assertEqual(i, j)

    def test_extraction_of_a_network_3(self):
        """

        Extraction of a network with default
         - change network shape
        this should trigger new network
        """
        c = {
            'architecture': '16:8:7:4:1',
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information
        mock_default_network = self.mock_default_network

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                mock_default_network(self)
                section_mocker(self, 'C', c)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            *dummy, system_scaffold = parameter_file_parser('pippo')
        # test layer 0
        for i, j in zip(system_scaffold['C'].layers_shaping[0],
                        [int(self.default_network['g_size']), 16]):
            self.assertEqual(i, j)
        # test layer 2
        for i, j in zip(system_scaffold['C'].layers_shaping[2], [8, 7]):
            self.assertEqual(i, j)
        # test layer 3
        for i, j in zip(system_scaffold['C'].layers_shaping[3], [7, 4]):
            self.assertEqual(i, j)
        # test for trainable
        for i in system_scaffold['C'].layers_trainable:
            self.assertEqual(i, True)

    def test_definition_of_a_network(self):
        """

        Putting in place a network without default
        """
        c = {
            'g_size': '32',
            'architecture': '16:8:4:1',
            'trainable': '1:1:1:1',
            'output_offset': '7'
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                section_mocker(self, 'C', c)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            *dummy, system_scaffold = parameter_file_parser('pippo')
        self.assertAlmostEqual(system_scaffold['C'].offset, 7)
        with self.assertRaises(NetworkNotAvailableError):
            dummy = system_scaffold['H']

    def test_malformed_definition_of_a_network(self):
        """

        Putting in place a network without architecture and no default
        """
        c = {'trainable': '1:1:1:1', 'output_offset': '7'}
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                section_mocker(self, 'C', c)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            with self.assertRaises(ValueError):
                dummy = parameter_file_parser('pippo')

    def test_malformed_definition_of_a_network_2(self):
        """

        Putting in place a network without g_size
        """
        c = {
            'architecture': '16:8:4:1',
            'trainable': '1:1:1:1',
            'output_offset': '7'
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                section_mocker(self, 'C', c)

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            #with self.assertRaises(ShapeError):
            with self.assertRaises(TypeError):
                dummy = parameter_file_parser('pippo')

    def test_extraction_of_a_network_and_load(self):
        """
        """
        c = {
            'architecture': '16:8:4:1',
            'trainable': '1:1:1:1',
            'output_offset': '7',
            'behavior': 'keep:keep:load:new',
            'layer2_w_file': 'w',
            'layer2_b_file': 'b'
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information
        mock_default_network = self.mock_default_network

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                mock_default_network(self)
                section_mocker(self, 'C', c)

        def mock_numpy_load(filename):
            if filename == 'w':
                return np.arange(32).reshape(8, 4)
            elif filename == 'b':
                return np.arange(4)
            else:
                raise ValueError('mock not implemented')

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            with mock.patch('numpy.load', mock_numpy_load):
                *dummy, system_scaffold = parameter_file_parser('pippo')
        self.assertEqual(system_scaffold['C'].offset, 7)
        self.assertEqual(system_scaffold['H'].offset, 0)

    def test_definition_of_a_network_and_load(self):
        """
        """
        c = {
            'g_size': '32',
            'architecture': '16:8:4:1',
            'trainable': '1:1:1:1',
            'output_offset': '7',
            'behavior': 'keep:keep:load:new',
            'layer2_w_file': 'w',
            'layer2_b_file': 'b'
        }
        # recover mocker functions
        mock_io_information = self.mock_io_information
        mock_training_parameters = self.mock_training_parameters
        mock_data_information = self.mock_data_information

        # creation of mocking function
        class MockConfigParser(configparser.ConfigParser):
            def __init__(self, *args, **kvargs):
                super().__init__(*args, **kvargs)
                mock_io_information(self)
                mock_training_parameters(self)
                mock_data_information(self)
                section_mocker(self, 'C', c)

        def mock_numpy_load(filename):
            if filename == 'w':
                return np.arange(32).reshape(8, 4)
            elif filename == 'b':
                return np.arange(4)
            else:
                raise ValueError('mock not implemented')

        with mock.patch('configparser.ConfigParser', MockConfigParser):
            with mock.patch('numpy.load', mock_numpy_load):
                *dummy, system_scaffold = parameter_file_parser('pippo')

        self.assertAlmostEqual(system_scaffold['C'].offset, 7)
        with self.assertRaises(NetworkNotAvailableError):
            dummy = system_scaffold['H']


if __name__ == '__main__':
    unittest.main()
