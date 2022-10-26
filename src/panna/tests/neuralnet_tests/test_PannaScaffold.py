###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import configparser
import unittest

import numpy as np

import lib.parser_callable as parser_callable
from lib.errors import NetworkNotAvailableError
from neuralnet.a2affnetwork import A2affNetwork
from neuralnet.example import Example
from neuralnet.panna_scaffold import PannaScaffold
from tests.utils import section_mocker


class Test_PannaScaffold(unittest.TestCase):
    def setUp(self):
        feature_size = 16
        layer_sizes = [8, 4, 2, 1]
        name = 'default'
        self.layer_sizes = layer_sizes
        self.default_network = A2affNetwork(feature_size, layer_sizes, name)

    def test_get_network(self):
        system_scaffold = PannaScaffold()
        system_scaffold.default_network = self.default_network
        c_network = system_scaffold['C']
        self.assertEqual(self.default_network.layers_shaping,
                         c_network.layers_shaping)
        self.assertEqual(self.default_network.layers_trainable,
                         c_network.layers_trainable)
        self.assertEqual(self.default_network.layers_activation,
                         c_network.layers_activation)
        for ref, test in zip(self.default_network.network_wb,
                             c_network.network_wb):
            np.testing.assert_array_equal(ref, test)
        self.assertEqual(self.default_network.offset, c_network.offset)

        system_scaffold = PannaScaffold()
        with self.assertRaises(NetworkNotAvailableError):
            c_network = system_scaffold['C']

    def test_creation_atomic_sequence(self):
        system_scaffold = PannaScaffold()
        system_scaffold.default_network = self.default_network
        sequence = ('C', 'H', 'N')
        [system_scaffold[x] for x in sequence]
        self.assertEqual(system_scaffold.atomic_sequence, sequence)
        self.assertEqual(system_scaffold.n_species, len(sequence))
        with self.assertRaises(ValueError):
            system_scaffold[sequence[0]] = A2affNetwork(24, [1, 2, 4], 'C')
        with self.assertRaises(ValueError):
            system_scaffold['K'] = A2affNetwork(24, [1, 2, 4], 'C')

        system_scaffold['K'] = A2affNetwork(24, [1, 2, 4], 'K')
        new_sequence = tuple(list(sequence) + ['K'])
        self.assertEqual(system_scaffold.atomic_sequence, new_sequence)

    def test_creation_given_atomic_sequence_w_zeros(self):
        sequence = ('C', 'H', 'N')
        zeros = (1.0, 2.0, 3.0)

        # creation of mocking function
        config = configparser.ConfigParser(
            converters={
                '_comma_list':
                parser_callable.get_list_from_comma_sep_strings,
                '_comma_list_floats':
                parser_callable.get_list_floats_from_comma_sep_strings,
            })
        section_mocker(
            config, 'DATA_INFORMATION', {
                'atomic_sequence': ','.join(sequence),
                'output_offset': ','.join([str(zero) for zero in zeros])
            })

        system_scaffold = PannaScaffold(config)
        system_scaffold.default_network = self.default_network

        for species, zero in zip(sequence, zeros):
            self.assertEqual(system_scaffold[species].offset, zero)

    def test_creation_given_atomic_sequence_wo_zeros(self):
        sequence = ('C', 'H', 'N')

        # creation of mocking function
        config = configparser.ConfigParser(
            converters={
                '_comma_list':
                parser_callable.get_list_from_comma_sep_strings,
                '_comma_list_floats':
                parser_callable.get_list_floats_from_comma_sep_strings,
            })
        section_mocker(config, 'DATA_INFORMATION', {
            'atomic_sequence': ','.join(sequence),
        })

        system_scaffold = PannaScaffold(config)
        system_scaffold.default_network = self.default_network

        for species in sequence:
            self.assertEqual(system_scaffold[species].offset, 0.0)

    def test_creation_given_atomic_sequence_error1(self):
        zeros = (1.0, 2.0, 3.0)

        # creation of mocking function
        config = configparser.ConfigParser(
            converters={
                '_comma_list':
                parser_callable.get_list_from_comma_sep_strings,
                '_comma_list_floats':
                parser_callable.get_list_floats_from_comma_sep_strings,
            })
        section_mocker(
            config, 'DATA_INFORMATION',
            {'output_offset': ','.join([str(zero) for zero in zeros])})

        with self.assertRaises(ValueError):
            PannaScaffold(config)

    def test_creation_given_atomic_sequence_error2(self):
        sequence = ('C', 'H', 'N')
        zeros = (2.0, 3.0)

        # creation of mocking function
        config = configparser.ConfigParser(
            converters={
                '_comma_list':
                parser_callable.get_list_from_comma_sep_strings,
                '_comma_list_floats':
                parser_callable.get_list_floats_from_comma_sep_strings,
            })
        section_mocker(
            config, 'DATA_INFORMATION', {
                'output_offset': ','.join([str(zero) for zero in zeros]),
                'atomic_sequence': ','.join(sequence)
            })

        with self.assertRaises(ValueError):
            PannaScaffold(config)

    def test_evaluation_of_a_system(self):
        # taking the default network
        network = self.default_network

        # give value to wb
        weights_biases = [(np.arange(x * y).reshape(x, y),
                           np.arange(y).reshape(y))
                          for x, y in network.layers_shaping]
        behaviors = ['load' for x in self.layer_sizes]
        network.customize_network(behaviors=behaviors,
                                  override_wb=weights_biases)

        # extract the scaffold
        sequence = ('C', 'H', 'N')
        zeros = (1.0, 2.0, 3.0)

        # creation of mocking function
        config = configparser.ConfigParser(
            converters={
                '_comma_list':
                parser_callable.get_list_from_comma_sep_strings,
                '_comma_list_floats':
                parser_callable.get_list_floats_from_comma_sep_strings,
            })
        section_mocker(
            config, 'DATA_INFORMATION', {
                'atomic_sequence': ','.join(sequence),
                'output_offset': ','.join([str(zero) for zero in zeros])
            })

        system_scaffold = PannaScaffold(config)
        system_scaffold.default_network = network

        # evaluate
        example = Example(
            np.arange(48).reshape(3, 16), np.arange(3), 0,
            np.arange(432).reshape(3, 16, 3 * 3),
            np.arange(9).reshape(3, 3))

        energy, forces = system_scaffold.evaluate(example, True)

        self.assertEqual(energy.shape, ())
        self.assertEqual(forces.shape, (9, ))

    def test_failed_evaluation_of_a_system(self):
        """
        if not weight and biases error must be raised
        """
        # extract the scaffold
        sequence = ('C', 'H', 'N')
        zeros = (1.0, 2.0, 3.0)

        # creation of mocking function
        config = configparser.ConfigParser(
            converters={
                '_comma_list':
                parser_callable.get_list_from_comma_sep_strings,
                '_comma_list_floats':
                parser_callable.get_list_floats_from_comma_sep_strings,
            })
        section_mocker(
            config, 'DATA_INFORMATION', {
                'atomic_sequence': ','.join(sequence),
                'output_offset': ','.join([str(zero) for zero in zeros])
            })

        system_scaffold = PannaScaffold(config)
        system_scaffold.default_network = self.default_network

        # evaluate
        example = Example(
            np.arange(48).reshape(3, 16), np.arange(3), 0,
            np.arange(432).reshape(3, 16, 3 * 3),
            np.arange(9).reshape(3, 3))

        with self.assertRaises(ValueError):
            system_scaffold.evaluate(example, True)


if __name__ == '__main__':
    unittest.main()
