###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from lib.errors import ShapeError

from .variable_helpers import _variable_on_cpu, _variable_random_uniform

logger = logging.getLogger(__name__)

class Layer():
    """All to all connected layer to be used to create networks

    A layer is a a generic all to all connected layer with several
    activation functions.

    Parameters
    ----------
    wb_shape: sequence of 2 numbers
        First number is interpreted as the number of weights per node
        Second number is interpreted as the number of nodes
    trainable: integer of bool
        Specify if the layer is trainable or not
    activation: integer
        Specify the activation type
            - 0:: linear
            - 1:: Gaussian
            - 2:: rbf
                  o_i = \\exp(-\\sum_{j=0}^{in_size}(in_j - w_i)^2)
                  basically each neuron represent a basis element
                  (Gaussian centered around w)
            - 3:: relu
            - 4:: tanh
    name: a layer name like its number
    w_value: numpy array, optional
        Tensor with weights value, shape must be
        (number_of_weights, number_of_biases)
        if not provided the layer can not be evaluated and it will be
        randomly initialized if required in TF
    b_value: numpy array, optional
        Tensor with biases value, shape must be
        if not provided the layer can not be evaluated and it will be
        randomly initialized if required in TF
    """

    _wb_file_name = 'layer_{}'  #: string used when dumping/loading
    _tf_regexp = r'layer_(\d+)'

    def __init__(self, wb_shape, trainable, activation, name):
        super().__init__()

        for i in wb_shape:
            if not i:
                raise ShapeError('wb shape can not be undefined nor zero')

        self.wb_shape = tuple(wb_shape)
        self.b_shape = (wb_shape[1], )
        self.trainable = trainable
        self.activation = activation
        self.name = name

        self._w_value = np.empty(0)
        self._b_value = np.empty(0)
        self._w_distribution = None

    def __str__(self):
        s = f'layer: {self.name}\n'
        s += f'shape: {self.wb_shape}\n'
        s += f'w: {self._w_value.size != 0},b: {self._b_value.size != 0}'
        return s

    @property
    def wb_value(self):
        """ tuple(numpy_array): value of weights and biases
        """
        return self.w_value, self.b_value

    @property
    def w_value(self):
        """ numpy_array: value of weights

        The setter also check for shape consistency with what
        stored in wb_shape that is the leading quantity

        Raises
        ------
        ShapeError:
            if the shape is not consistent with what is stored in Wb_shape
        """
        return self._w_value

    @w_value.setter
    def w_value(self, value):
        if value.size == 0:
            self._w_value = value
            return

        if value.shape != self.wb_shape:
            raise ShapeError('request to setup a leayer of shape '
                             '{} with parameters of shape {}'.format(
                                 self.wb_shape, value.shape))
        self._w_value = value

    @property
    def b_value(self):
        """ numpy_array: value of biases

        The setter also check for shape consistency with what
        stored in b_shape that is the leading quantity

        Raises
        ------
        ShapeError:
            if the shape is not consistent with what is stored in b_shape
        """
        return self._b_value

    @b_value.setter
    def b_value(self, value):
        if value.size == 0:
            self._b_value = value
            return

        if value.shape != self.b_shape:
            raise ShapeError('request to setup a leayer of shape '
                             '{} with parameters of shape {}'.format(
                                 self.b_shape, value.shape))
        self._b_value = value

    def evaluate(self, in_vectors, din_vectors=np.empty(0)):
        """ Evaluate the layer for a given set of inputs

        Note
        ----
        activation can be
          - 0:: linear
          - 1:: gaussian
          - 2:: rbf o_i = .. math:: \exp(-\sum_{j=0}^{in_size}(in_j - w_i)^2)
          - 3:: relu
          - 4:: tanh

        Parameters
        ----------
        in_vectors: numpy_array (?, wb_shape[0])
            First dimension is the number of vector that must be evaluate by
            the layer
        din_vectors: numpy_array(?, wb_shape[0]), optional
            Derivatives of the layer to be computed, used in to evaluate forces

            First dimension is the number of vector that must be evaluate by
            the layer

        Returns
        -------
        numpy_array(?, wb_shape[1])
            Result of the activation

        numpy_array(?, wb_shape[1])
            Derivative of the activation.
            this is returned ONLY if din_vectors.size > 0

        Raises
        ------
        ValueError:
            if weights or biases are not available and the computation
            can not be performed

        ValueError:
            For some non yet implemented derivatives
        """
        if (self._w_value.size == 0 or self._b_value.size == 0) and \
           self.activation != 0:
            raise ValueError('weights and bias not availables')
        elif self._w_value.size == 0 and self.activation == 0:
            raise ValueError('weights not availables')

        if self.activation == 0:
            tmp = np.matmul(in_vectors, self._w_value)
            act = tmp + self._b_value
            if din_vectors.size > 0:
                d_act = np.matmul(din_vectors, self._w_value)
        elif self.activation == 1:
            tmp = np.matmul(in_vectors, self._w_value) + self._b_value
            act = np.exp(np.negative(np.square(tmp)))
            if din_vectors.size > 0:
                tmp2 = np.matmul(din_vectors, self._w_value)
                # vulgar shaping:
                # in_vectors.shape[0]: number of atoms of this species
                # wb_shape[1]: out size
                # din_vectors.shape[-1]: N_atoms * 3
                d_act = -2.0 * np.reshape(
                    tmp * act, (in_vectors.shape[0], 1, self.wb_shape[1])) * tmp2

        elif self.activation == 2:
            tmp = np.tile(np.expand_dims(in_vectors, -1), self.wb_shape[1])
            tmp = tmp - self._w_value
            act = np.exp(-np.sum(tmp**2, 1))
            if din_vectors.size > 0:
                d_act = - 2.0 * \
                    np.reshape(act, (in_vectors.shape[0], 1, self.wb_shape[1])) * \
                    np.matmul(din_vectors, tmp)
        elif self.activation == 3:
            tmp = np.matmul(in_vectors, self._w_value) + self._b_value
            act = np.maximum(0, tmp)
            if din_vectors.size > 0:
                tmp2 = np.matmul(din_vectors, self._w_value)
                d_act = np.reshape(np.where(tmp > 0.0, 1.0, 0.0),
                                   (in_vectors.shape[0], 1, self.wb_shape[1])) * tmp2
        elif self.activation == 4:  # tanh
            tmp = np.matmul(in_vectors, self._w_value) + self._b_value
            act = np.tanh(tmp)
            if din_vectors.size > 0:
                tmp2 = np.matmul(din_vectors, self._w_value)
                d_act = np.reshape(1 - act**2,
                                   (in_vectors.shape[0], 1, self.wb_shape[1])) * tmp2
        else:
            raise ValueError('activation function not implemented')
        if din_vectors.size > 0:
            return act, d_act
        return act

    def tf_evaluate(self, in_tensor):
        """Define an all to all connected layer with species division.

        Parameters
        ----------
        in_tensor: input to be computed, (n_atoms of this species x in_size)

        Returns
        -------
        Output of the layer
        """
        dist_parameter = 1.0 / self.b_shape[0]

        if self.w_value.size > 0:
            weights = _variable_on_cpu('weights',
                                       self.wb_shape,
                                       tf.constant_initializer(self.w_value),
                                       trainable=self.trainable)
        else:
            weights = _variable_random_uniform('weights',
                                               self.wb_shape,
                                               dist_parameter,
                                               trainable=self.trainable)

        if self.b_value.size > 0:
            biases_init = tf.constant_initializer(self.b_value)
        else:
            biases_init = tf.constant_initializer(0.0)

        biases = _variable_on_cpu('biases',
                                  self.b_shape,
                                  biases_init,
                                  trainable=self.trainable)

        tf.add_to_collection(ops.GraphKeys.WEIGHTS, weights)
        tf.add_to_collection(ops.GraphKeys.BIASES, biases)

        if self.activation == 1:  # gaussian
            w_contrib = tf.matmul(in_tensor, weights, name='w_contrib') + biases
            exp_arg = tf.square(w_contrib, name='exp_arg')
            output = tf.exp(tf.negative(exp_arg), name='activation')
        elif self.activation == 2:  # rbf
            reshaped_in = tf.tile(tf.expand_dims(in_tensor, -1),
                                  [1, 1, self.b_shape[0]])
            w_contrib = tf.subtract(reshaped_in, weights, name='w_contrib')
            exp_arg = tf.reduce_sum(tf.square(w_contrib, name='exp_arg'), 1)
            output = tf.exp(tf.negative(exp_arg), name='activation')
        elif self.activation == 3:  # relu
            w_contrib = tf.matmul(in_tensor, weights, name='w_contrib') + biases
            output = tf.nn.relu(w_contrib, name='my_relu')
        elif self.activation == 4:  # tanh
            w_contrib = tf.matmul(in_tensor, weights, name='w_contrib') + biases
            output = tf.nn.tanh(w_contrib, name='activation')
        elif self.activation == 0:  # linear
            w_contrib = tf.matmul(in_tensor, weights, name='w_contrib')
            output = tf.add(w_contrib, biases, name='activation')
        else:
            raise ValueError('activation not implemented', self.activation)

        return output

    @property
    def tf_ckpt_elements(self):
        """ Regexps for TF checkpoint, this call is meant to be used in a
        recursive script.

        Returns
        ------
        names: list of str
            each string is a regexp that need to be matched in the tf namespace
            tree to access the subordinate object data.
        objects: None
            This elements does not have sub-object so the node in the tf
            namespace contains only the information needed to self-initialize.
        """
        names = ['weights', 'biases']
        objects = None
        return names, objects

    @tf_ckpt_elements.setter
    def tf_ckpt_elements(self, value):
        """ Self initialization function

        Prameters
        ---------
        value: tuple of 2 np.array
            This values are obtained by matching the 2 names in tf_ckpt_elements

        """
        self.w_value, self.b_value = value

    def to_json(self):
        """ serialize obj to json
        Return
        ------
           json data, {name:variable, ..}
        """
        json_data = {
            'wb_shape': self.wb_shape,
            'activation': self.activation,
            'trainable': self.trainable,
            'layer_name': self.name
        }
        base = self.__class__._wb_file_name.format(self.name)
        wb = {
            base + '_weights_{}x{}.npy'.format(*self.wb_shape): self.w_value,
            base + '_biases_{}.npy'.format(self.b_shape[0]): self.b_value
        }

        return json_data, wb

    @classmethod
    def from_json(cls, json_data, base_path=None):
        """
        Parameters
        ----------
          json_data: dumped json
          base_path: a path to search fro the weights and the biases
        """
        layer_name = json_data['layer_name']

        if base_path:
            base_path = os.path.join(base_path, cls._wb_file_name.format(layer_name))
        else:
            base_path = cls._wb_file_name.format(layer_name)

        wb_shape = json_data['wb_shape']
        layer = cls(wb_shape=wb_shape,
                    trainable=json_data['trainable'],
                    activation=json_data['activation'],
                    name=layer_name)
        path = base_path + '_weights_{}x{}.npy'.format(*wb_shape)
        # np load does not support path objects
        if os.path.isfile(path):
            logger.info(f'loading {path}')
            layer.w_value = np.load(str(path))
        path = base_path + '_biases_{}.npy'.format(wb_shape[1])
        if os.path.isfile(path):
            logger.info(f'loading {path}')
            layer.b_value = np.load(str(path))
        return layer
