###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import unittest
from copy import copy, deepcopy

import numpy as np

from lib.errors import ShapeError
from neuralnet.a2affnetwork import A2affNetwork
from neuralnet.layers import Layer


class Test_A2affNetwork(unittest.TestCase):
    def setUp(self):
        self.feature_size = 16
        self.layer_size = [8, 4, 1]
        self.layers_shaping = [(16, 8), (8, 4), (4, 1)]
        self.name = 'C'
        self.trainables = (1, 0, 1)
        self.activations = (1, 0, 0)
        self.offset = 42
        # building auto quantity
        self.network_wb = list(
            zip([
                np.arange(x * y).reshape(x, y) for x, y in self.layers_shaping
            ], [np.arange(x) for x in self.layer_size]))
        self.n_layers = len(self.layers_shaping)

        self.default_network = A2affNetwork(self.feature_size, self.layer_size,
                                            self.name)
        self.test_network = A2affNetwork(
            self.feature_size, self.layer_size, self.name, self.network_wb,
            self.trainables, self.activations, self.offset)

    def test_verify_default_network(self):
        self.assertEqual(self.default_network.layers_shaping,
                         self.layers_shaping)
        self.assertEqual(self.default_network.layers_trainable,
                         tuple([True for x in range(self.n_layers)]))
        self.assertEqual(self.default_network.layers_activation,
                         tuple([1 for x in range(self.n_layers - 1)] + [0]))
        self.assertEqual(self.default_network.offset, 0.0)
        for test, reference in zip(self.default_network.network_wb,
                                   [(np.empty(0), np.empty(0))
                                    for x in range(self.n_layers)]):
            np.testing.assert_array_equal(test, reference)

    def test_verify_test_network(self):
        self.assertEqual(self.test_network.layers_shaping, self.layers_shaping)
        self.assertEqual(self.test_network.layers_trainable, self.trainables)
        self.assertEqual(self.test_network.layers_activation, self.activations)
        for (w, b), (tw, tb) in zip(self.test_network.network_wb,
                                    self.network_wb):
            np.testing.assert_array_equal(w, tw)
            np.testing.assert_array_equal(b, tb)
        self.assertEqual(self.test_network.offset, self.offset)

    def test_initialize_with_incompatible_parameters(self):
        network_wb = self.network_wb + [(np.asarray([4, 4]))]
        trainables = list(self.trainables) + [True]
        activations = list(self.activations) + [0]
        with self.assertRaises(ValueError):
            A2affNetwork(self.feature_size, self.layer_size, self.name,
                         network_wb)
        with self.assertRaises(ValueError):
            A2affNetwork(self.feature_size, self.layer_size, self.name, None,
                         trainables)
        with self.assertRaises(ValueError):
            A2affNetwork(self.feature_size, self.layer_size, self.name, None,
                         None, activations)
        # this test failure is inherited by construction form the
        # layer.
        network_wb = copy(self.network_wb)
        x, y = self.layers_shaping[0]
        x += 1
        y += 1
        network_wb[0] = (np.arange(x * y).reshape(x, y), np.arange(y))
        with self.assertRaises(ShapeError):
            A2affNetwork(self.feature_size, self.layer_size, self.name,
                         network_wb)

    def test_override_activation(self):
        new_activations = list(self.activations)
        new_activations[0] = 0
        network = self.test_network
        network.layers_activation = new_activations
        self.assertEqual(network.layers_activation, tuple(new_activations))
        with self.assertRaises(ValueError):
            network.layers_activation = (new_activations + [1, 7, 8])
        with self.assertRaises(ValueError):
            network.layers_activation = new_activations[:-1]

    def test_override_trainable(self):
        new_trainables = list(self.trainables)
        new_trainables[0] = 0
        network = self.test_network
        network.layers_trainable = new_trainables
        self.assertEqual(network.layers_trainable, tuple(new_trainables))
        with self.assertRaises(ValueError):
            network.layers_trainable = (new_trainables + [1, 7, 8])
        with self.assertRaises(ValueError):
            network.layers_trainable = new_trainables[:-1]

    def test_getter(self):
        a = self.default_network[0]
        b = self.default_network._layers[0]
        self.assertNotEqual(id(a), id(b))

    def test_compatible_setter(self):
        network = self.test_network
        new_first_layer = Layer(self.layers_shaping[0], False, 0)
        network[0] = new_first_layer
        np.testing.assert_array_equal(network[0].wb_value,
                                      (np.empty(0), np.empty(0)))

    def test_incompatible_setter(self):
        network = self.test_network
        correct_shape = self.layers_shaping[0]
        wrong_shape = (correct_shape[0] + 1, correct_shape[1] + 1)
        new_first_layer = Layer(wrong_shape, False, 0)
        with self.assertRaises(ShapeError):
            network[0] = new_first_layer

    # test customize

    def test_customize1(self):
        """Do nothing
        test the ability of the customize of doing nothing
        """
        new_network = deepcopy(self.test_network)
        new_network.customize_network()
        self.assertEqual(self.test_network.layers_shaping,
                         new_network.layers_shaping)
        self.assertEqual(self.test_network.layers_trainable,
                         new_network.layers_trainable)
        self.assertEqual(self.test_network.layers_activation,
                         new_network.layers_activation)
        for (w, b), (tw, tb) in zip(self.test_network.network_wb,
                                    new_network.network_wb):
            np.testing.assert_array_equal(w, tw)
            np.testing.assert_array_equal(b, tb)
        self.assertEqual(self.test_network.offset, new_network.offset)

    def test_customize2(self):
        """Change only zero
        """
        zero = 27.0
        new_network = deepcopy(self.test_network)
        new_network.customize_network(offset=zero)
        self.assertEqual(new_network.offset, zero)

    def test_customize3(self):
        """Same structure, change trainable
        """
        new_network = deepcopy(self.test_network)
        new_trainables = (0, 1, 0)
        new_network.customize_network(trainables=new_trainables)
        self.assertEqual(new_network.layers_trainable, new_trainables)

    def test_customize4(self):
        """Same structure, change activations
        """
        new_network = deepcopy(self.test_network)
        new_activations = (0, 1, 0)
        new_network.customize_network(activations=new_activations)
        self.assertEqual(new_network.layers_activation, new_activations)

    def test_customize5(self):
        """Same stucture, try to use a wrong layer_behavior
        """
        behaviors = ['new' for x in range(self.n_layers)] + ['new']
        new_network = deepcopy(self.test_network)
        with self.assertRaises(ValueError):
            new_network.customize_network(behaviors=behaviors)

    def test_customize6(self):
        """Same structure, try to keep a layer, load a layer, generate a new layer
        """
        new_network = deepcopy(self.test_network)
        behaviors = ['new' for x in range(self.n_layers)]
        behaviors[0] = 'keep'
        behaviors[1] = 'load'
        behaviors[2] = 'new'
        x, y = self.layers_shaping[1]
        override_wb = {
            1: (np.arange(x * y, 2 * x * y).reshape(x, y), np.arange(y, 2 * y))
        }
        new_network.customize_network(
            behaviors=behaviors, override_wb=override_wb)
        # check keep:
        # trainable and activation do not need to be conserved
        # except in this, particular case so are not checked
        layer = new_network[0]
        ref_layer = self.test_network[0]
        self.assertEqual(layer.wb_shape, ref_layer.wb_shape)
        self.assertEqual(layer.b_shape, ref_layer.b_shape)
        np.testing.assert_array_equal(layer.w_value, ref_layer.w_value)
        np.testing.assert_array_equal(layer.b_value, ref_layer.b_value)
        # check load:
        # loaded tensor can not have a wrong shape
        # because it is forced by the layer itself
        layer = new_network[1]
        ref_layer = self.test_network[1]
        self.assertEqual(layer.wb_shape, ref_layer.wb_shape)
        self.assertEqual(layer.b_shape, ref_layer.b_shape)
        np.testing.assert_array_equal(layer.w_value, override_wb[1][0])
        np.testing.assert_array_equal(layer.b_value, override_wb[1][1])
        # check new:
        # loaded tensor can not have a wrong shape
        # because it is forced by the layer itself
        layer = new_network[2]
        ref_layer = self.test_network[2]
        self.assertEqual(layer.wb_shape, ref_layer.wb_shape)
        self.assertEqual(layer.b_shape, ref_layer.b_shape)
        np.testing.assert_array_equal(layer.wb_value,
                                      (np.empty(0), np.empty(0)))

    def test_customize7(self):
        """Different structure

        Ask for a new structure, without other information,
        expected behavior is a new PANNA default network
        that keeps name and offset
        """
        new_network = deepcopy(self.test_network)
        new_strucutre = [8, 4, 3, 2, 1]
        new_network.customize_network(layers_size=new_strucutre)
        ref_network = A2affNetwork(
            self.feature_size, new_strucutre, self.name, offset=self.offset)

        self.assertEqual(new_network.layers_shaping,
                         ref_network.layers_shaping)
        self.assertEqual(new_network.layers_trainable,
                         ref_network.layers_trainable)
        self.assertEqual(new_network.layers_activation,
                         ref_network.layers_activation)

        for (w, b), (tw, tb) in zip(new_network.network_wb,
                                    ref_network.network_wb):
            np.testing.assert_array_equal(w, tw)
            np.testing.assert_array_equal(b, tb)
        self.assertEqual(ref_network.offset, new_network.offset)
        self.assertEqual(ref_network.name, new_network.name)

    def test_customize8(self):
        """Different structure

        same as 7, just different kind of structure
        """
        new_network = deepcopy(self.test_network)
        new_strucutre = [8, 5, 2, 1]
        new_network.customize_network(layers_size=new_strucutre)
        ref_network = A2affNetwork(
            self.feature_size, new_strucutre, self.name, offset=self.offset)

        self.assertEqual(new_network.layers_shaping,
                         ref_network.layers_shaping)
        self.assertEqual(new_network.layers_trainable,
                         ref_network.layers_trainable)
        self.assertEqual(new_network.layers_activation,
                         ref_network.layers_activation)

        for (w, b), (tw, tb) in zip(new_network.network_wb,
                                    ref_network.network_wb):
            np.testing.assert_array_equal(w, tw)
            np.testing.assert_array_equal(b, tb)
        self.assertEqual(ref_network.offset, new_network.offset)
        self.assertEqual(ref_network.name, new_network.name)

    def test_customize9(self):
        """Different structure, change activations and trainables
        """
        new_network = deepcopy(self.test_network)
        new_strucutre = [8, 5, 2, 1]
        new_activations = (1, 1, 1, 1)
        new_trainables = (1, 0, 0, 1)
        new_network.customize_network(
            layers_size=new_strucutre,
            trainables=new_trainables,
            activations=new_activations)

        self.assertEqual(new_network.layers_trainable, new_trainables)
        self.assertEqual(new_network.layers_activation, new_activations)

        for w, b in new_network.network_wb:
            np.testing.assert_array_equal(w, np.empty(0))
            np.testing.assert_array_equal(b, np.empty(0))
        self.assertEqual(new_network.offset, self.test_network.offset)
        self.assertEqual(new_network.name, self.test_network.name)

    def test_customize10(self):
        """Different structure, test, keep_wb, load, new
        """
        new_network = deepcopy(self.test_network)
        new_strucutre = [8, 5, 2, 1]
        new_activations = (1, 1, 1, 1)
        new_trainables = (1, 0, 0, 1)
        behaviors = ['new' for x in range(len(new_strucutre))]
        behaviors[0] = 'keep'
        behaviors[1] = 'new'
        behaviors[2] = 'load'
        x, y = 5, 2
        override_wb = {
            2: (np.arange(x * y, 2 * x * y).reshape(x, y), np.arange(y, 2 * y))
        }
        new_network.customize_network(
            layers_size=new_strucutre,
            behaviors=behaviors,
            trainables=new_trainables,
            activations=new_activations,
            override_wb=override_wb)

        self.assertEqual(new_network.layers_trainable, new_trainables)
        self.assertEqual(new_network.layers_activation, new_activations)
        self.assertEqual(new_network.offset, self.test_network.offset)
        self.assertEqual(new_network.name, self.test_network.name)

        # check keep:
        # trainable and activation do not need to be conserved
        # except in this, particular case so are not checked
        layer = new_network[0]
        ref_layer = self.test_network[0]
        self.assertEqual(layer.wb_shape, ref_layer.wb_shape)
        self.assertEqual(layer.b_shape, ref_layer.b_shape)
        np.testing.assert_array_equal(layer.w_value, ref_layer.w_value)
        np.testing.assert_array_equal(layer.b_value, ref_layer.b_value)
        # check new:
        # loaded tensor can not have a wrong shape
        # because it is forced by the layer itself
        layer = new_network[1]
        np.testing.assert_array_equal(layer.w_value, np.empty(0))
        np.testing.assert_array_equal(layer.b_value, np.empty(0))
        # check load:
        # loaded tensor can not have a wrong shape
        # because it is forced by the layer itself
        layer = new_network[2]
        np.testing.assert_array_equal(layer.w_value, override_wb[2][0])
        np.testing.assert_array_equal(layer.b_value, override_wb[2][1])

    def test_evalueate1(self):
        network = self.test_network
        features_vector = np.arange(32).reshape(2, 16)
        out = network.evaluate(features_vector)
        self.assertEqual(out.shape, (2, 1))

        dfeatures_vector = np.arange(384).reshape(2, 16, 3 * 4)
        out, dout = network.evaluate(features_vector, dfeatures_vector)
        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(dout.shape, (2, 3 * 4))


if __name__ == '__main__':
    unittest.main()
