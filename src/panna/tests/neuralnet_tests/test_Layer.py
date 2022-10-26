###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import unittest

import numpy as np
import tensorflow as tf

from neuralnet.layers import Layer
from lib.errors import ShapeError


class Test_Layer(unittest.TestCase):
    def setUp(self):
        self.shape = (4, 5)
        self.wrong_shape = (4, 6)
        self.trainable = True
        self.activation = 0

        self.w_c = np.arange(self.shape[0] * self.shape[1],
                             dtype=np.float32).reshape(*self.shape)
        self.b_c = np.arange(self.shape[1], dtype=np.float32)
        self.w_w = np.arange(self.wrong_shape[0] * self.wrong_shape[1],
                             dtype=np.float32).reshape(*self.wrong_shape)
        self.b_w = np.arange(self.wrong_shape[1], dtype=np.float32)
        self.empty_layer = Layer(self.shape, self.trainable, self.activation)
        self.loaded_layer = Layer(self.shape, self.trainable, self.activation,
                                  self.w_c, self.b_c)

    def test_layer_empty(self):
        # Testing basic Layer creation
        self.assertEqual(self.empty_layer.wb_shape, self.shape)
        self.assertEqual(self.empty_layer.b_shape, (self.shape[1], ))
        self.assertEqual(self.empty_layer.trainable, self.trainable)
        self.assertEqual(self.empty_layer.activation, self.activation)
        np.testing.assert_array_equal(self.empty_layer.wb_value,
                                      (np.empty(0), np.empty(0)))

    def test_layer_loaded(self):
        # Testing Layer creation with weights and bias
        np.testing.assert_array_equal(self.w_c, self.loaded_layer.w_value)
        np.testing.assert_array_equal(self.b_c, self.loaded_layer.b_value)

    def test_inconsistent_layer_creation(self):
        # Testing Layer creation with wrong weights and bias
        with self.assertRaises(ShapeError):
            Layer(self.shape, self.trainable, self.activation, self.w_w,
                  self.b_c)
        with self.assertRaises(ShapeError):
            Layer(self.shape, self.trainable, self.activation, self.w_c,
                  self.b_w)

    def test_layer_consistent_wb_setter(self):
        # Testing Layer assignment with weights and bias
        n = self.shape[0] * self.shape[1]
        w_c1 = np.arange(n, 2 * n).reshape(*self.shape)
        b_c1 = np.arange(self.shape[1], 2 * self.shape[1])
        old_w = self.loaded_layer.w_value
        old_b = self.loaded_layer.b_value
        self.loaded_layer.w_value = w_c1
        self.loaded_layer.b_value = b_c1
        np.testing.assert_array_equal(w_c1, self.loaded_layer.w_value)
        np.testing.assert_array_equal(b_c1, self.loaded_layer.b_value)
        self.loaded_layer.w_value = old_w
        self.loaded_layer.b_value = old_b

    def test_layer_inconsistent_wb_setter(self):
        with self.assertRaises(ShapeError):
            self.loaded_layer.w_value = self.w_w
        with self.assertRaises(ShapeError):
            self.loaded_layer.b_value = self.b_w

    def test_layer_wb_emptyer(self):
        self.loaded_layer.w_value = np.empty(0)
        self.loaded_layer.b_value = np.empty(0)
        np.testing.assert_array_equal(self.empty_layer.wb_value,
                                      (np.empty(0), np.empty(0)))

    def test_layer_activation_linear_shape(self):
        in_vectors = np.arange(40).reshape(10, 4)
        din_vectors = np.arange(240).reshape(10, 3 * 2, 4)
        out1, out2 = self.loaded_layer.evaluate(in_vectors, din_vectors)
        self.assertEqual(out1.shape, (in_vectors.shape[0], self.shape[1]))
        self.assertEqual(
            out2.shape,
            (din_vectors.shape[0], din_vectors.shape[1], self.shape[1]))

    def test_layer_activation_gaussian_shape(self):
        in_vectors = np.arange(40).reshape(10, 4)
        din_vectors = np.arange(240).reshape(10, 3 * 2, 4)
        self.loaded_layer.activation = 1
        out1, out2 = self.loaded_layer.evaluate(in_vectors, din_vectors)
        self.assertEqual(out1.shape, (in_vectors.shape[0], self.shape[1]))
        self.assertEqual(
            out2.shape,
            (din_vectors.shape[0], din_vectors.shape[1], self.shape[1]))

    def test_layer_activation_rbf_shape(self):
        in_vectors = np.arange(40).reshape(10, 4)
        # din_vectors = np.arange(240).reshape(10, 3 * 2, 4)
        self.loaded_layer.activation = 2
        # out1, out2 = self.loaded_layer.evaluate(in_vectors, din_vectors)
        out1 = self.loaded_layer.evaluate(in_vectors)
        self.assertEqual(out1.shape, (in_vectors.shape[0], self.shape[1]))
        # self.assertEqual(
        #    out2.shape,
        #    (din_vectors.shape[0], din_vectors.shape[1], self.shape[1]))

    def test_layer_activation_relu_shape(self):
        in_vectors = np.arange(40).reshape(10, 4)
        # din_vectors = np.arange(240).reshape(10, 3 * 2, 4)
        self.loaded_layer.activation = 3
        # out1, out2 = self.loaded_layer.evaluate(in_vectors, din_vectors)
        out1 = self.loaded_layer.evaluate(in_vectors)
        self.assertEqual(out1.shape, (in_vectors.shape[0], self.shape[1]))
        # self.assertEqual(
        #    out2.shape,
        #    (din_vectors.shape[0], din_vectors.shape[1], self.shape[1]))

    def test_consistency(self):
        in_vectors = np.arange(40, dtype=np.float32).reshape(10, 4)
        for act in range(4):
            graph = tf.Graph()
            with graph.as_default():
                self.loaded_layer.activation = act
                out1 = self.loaded_layer.evaluate(in_vectors)
                out2 = self.loaded_layer.tf_evaluate(in_vectors)
                init = tf.compat.v1.global_variables_initializer()
                with tf.compat.v1.Session() as sess:
                    sess.run(init)
                    out2 = sess.run(out2)
                np.testing.assert_almost_equal(out1, out2)

    def test_layer_activation_noactivation_shape(self):
        # yes, I won't let you add activation without testing
        # pls also update test consistency routine :)
        in_vectors = np.arange(40).reshape(10, 4)
        self.loaded_layer.activation = 4
        with self.assertRaises(ValueError):
            self.loaded_layer.evaluate(in_vectors)


if __name__ == '__main__':
    unittest.main()
