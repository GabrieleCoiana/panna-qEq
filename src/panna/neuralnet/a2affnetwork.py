###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from collections import namedtuple
from copy import deepcopy
import logging
import os
import sys

import numpy as np
import tensorflow as tf

if __package__ != 'panna.neuralnet':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib.errors import ShapeError
else:
    from ..lib.errors import ShapeError

from .layers import Layer

logger = logging.getLogger(__name__)

def normalize_network_type(nn_type):
    if nn_type in ['a2aff', 'A2AFF', 'ff', 'a2a', 'FF', 'A2A']:
        nn_type = 'a2ff'
    else:
        raise ValueError(
            '{} != a2aff : '
            'Only all-to-all feed forward networks supported'.format(nn_type))
    return nn_type


class A2affNetwork():
    """ Implementation of a all to all connected feed forward network
    A2affNetwork (feature_size, layers_size, name, network_wb=None,
                  trainables=None, activations=None, offset=None)

    Returns an all to all connected feed forward network that is a
    basic  **general purpose** neural network

    Parameters
    ----------
    feature_size: int
        Size of the feature vector, or Input vector, or Gvector
        depending on notations
    layers_size: list of integers
        List of integers that describe the layers shape:
        for example a list like [128, 64, ..., 4, 1] will create a network
        like:
        - in layer: feature_size: 128
        - first layer: 128:64
        - second layer: 64:...
        - .....
        - last layer: ...:4
        - out layer:  4:1
    name: str
        name of the network, usually the species name
    network_wb: list of tuples, optional
        one tuple for each layer, a tuple is composed by 2 numpy_array
        [(w_values, b_values),......, (w_values, b_values)]
        an empty value is passed as np.empty(0)
        default is: creates all layer with Gaussian distribution
    trainables: list of bools, optional
        one for each layer to set a layer as trainable or not
        default is all layer trainables
    activations: list of integer, optional
        one for each layer to set the activation kind of the layer
        default is: gaussian:gaussian:....:linear
    offset: float, optional
        network offset value
        default is: 0.0
    norm_layer: Layer
        a layer applied before anything else for data normalization

    Raises
    ------
    ValueError if:
        - network_wb is too small
        - trainables is too small
        - activations is too small
    """
    network_type = 'a2ff'
    _wb_prefix = 'species_{}'
    # quantity used to recover tensors from TF checkpoint
    _reg_exps = {
        'network': r'species_(\d+)',
    }

    def __init__(self,
                 feature_size,
                 layers_size,
                 name,
                 network_wb=None,
                 trainables=None,
                 activations=None,
                 offset=None,
                 norm_layer=None):
        super().__init__()

        if not trainables:
            # default behavior: all trainable
            trainables = [True for i in range(len(layers_size))]
        if not activations:
            # default behavior: gaussian:gaussian:....:linear
            activations = [1 for i in range(len(layers_size) - 1)] + [0]

        if network_wb and (len(layers_size) != len(network_wb)):
            raise ValueError('network wb parameters are not enough')
        if len(layers_size) != len(trainables):
            raise ValueError('trainables parameters are not enough')
        if len(layers_size) != len(activations):
            raise ValueError('activations parameters are not enough')

        self.feature_size = int(feature_size)
        self.offset = offset or 0.0
        self.name = name
        self._norm_layer = norm_layer

        input_size = norm_layer.wb_shape[1] if norm_layer else self.feature_size
        layers_shaping = [(input_size, layers_size[0])] +\
                         [(layers_size[i], layers_size[i + 1]) for i
                          in range(len(layers_size) - 1)]
        layers = []
        for i in range(len(layers_size)):
            layer = Layer(layers_shaping[i], trainables[i], activations[i], f'{i}')
            if network_wb:
                layer.w_value, layer.b_value = network_wb[i]
            layers.append(layer)

        self._layers = layers

    def __str__(self):
        s = f'= {self.name} = network\n'
        s += f'feature size: {self.feature_size}\n'
        s += f'layers: {self.layers_size}\n'
        s += f'trainables: {self.layers_trainable}\n'
        s += f'activations: {self.layers_activation}\n'
        s += f'offset: {self.offset}'
        return s

    @property
    def layers_size(self):
        return [int(x.b_shape[0]) for x in self._layers]

    @property
    def norm_layer(self):
        return self._norm_layer

    @norm_layer.setter
    def norm_layer(self, value):
        if value.b_shape[0] == self[0].wb_shape[0]:
            self._norm_layer = value
        else:
            raise NotImplementedError('operation not yet supported')

    @property
    def layers_shaping(self):
        """
        Returns
        -------
        list of tuples, two elements for each layer e.g. [(shape_w, shape_b),.....]
        """
        return [x.wb_shape for x in self]

    @property
    def layers_trainable(self):
        """
        Returns
        -------
        Tuple of Boolean, one elements for each layer

        Raises
        ------
        ValueError
            if number of elements differ from number of layer
        """
        trainability = []
        for layer in self._layers:
            trainability.append(layer.trainable)
        return tuple(trainability)

    @layers_trainable.setter
    def layers_trainable(self, value):
        if len(value) != len(self.layers_size):
            raise ValueError('passed trainable vector is '
                             'not compatible with current layer size')
        for layer, trainable in zip(self._layers, value):
            layer.trainable = trainable

    @property
    def layers_activation(self):
        """ tuple of int
        one elements for each layer

        Raises
        ------
        ValueError
            if number of elements differ from number of layer
        """
        activations = []
        for layer in self._layers:
            activations.append(layer.activation)
        return tuple(activations)

    @layers_activation.setter
    def layers_activation(self, value):
        if len(value) != len(self.layers_size):
            raise ValueError('passed act vector is '
                             'not compatible with current layer size')
        for layer, activation in zip(self._layers, value):
            layer.activation = activation

    @property
    def network_wb(self):
        """ list of tuple
        two numpy_array for each layer, weights and biases values
        """
        wb = []
        for layer in self._layers:
            wb.append(layer.wb_value)
        return tuple(wb)

    def __getitem__(self, index):
        result = self._layers[index]
        return deepcopy(result)

    def __setitem__(self, index, value):
        old_layer = self._layers[index]
        if value.wb_shape != old_layer.wb_shape:
            raise ShapeError('request to substitute a leayer of shape '
                             '{} with one of shape {}'.format(
                                 old_layer.wb_shape, value.wb_shape))
        self._layers[index] = value

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        try:
            result = self[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return result

    def __len__(self):
        """number of layers"""
        return len(self._layers)

    def evaluate(self,
                 features_vector,
                 dfeatures_vectors=np.empty(0),
                 add_offset=True,
                 whole_output=False):
        """ Evaluate the layer for a given set of inputs

        Parameters
        ----------
        features_vector: numpy_array(?, feature_size)
            ? is the number of features that the network has to estimate
              can be read as the number of atoms of the kind the
              network should process
        dfeatures_vectors: numpy_array(?, 3 * total ?, features_size), optional
            TODO: check the sizes, maybe 1,2 are inverted
            3 * total ? is the total number of atoms in the system
            if not passed forces are not evaluated
        add_offset: Boolean, optional
            if adding the offset or not
        whole_output: Boolean, optional
            if returning activations for all layers besides the output

        Returns
        -------
        numpy_array (?, last_layer_shape)
             ? is the number of elements that have been be processed

        numpy_array(?, 3 * total ?)
            Derivatives, aka forces.
            this is returned ONLY if dfeatures_vectors.size > 0
        """
        Out = namedtuple("Out", ["e", "d_g_d_e", "stack"])
        in_vectors = features_vector

        activation_stack = []

        if dfeatures_vectors.size > 0:
            din_vectors = np.transpose(dfeatures_vectors, (0, 2, 1))

            if self.norm_layer:
                in_vectors, din_vectors = self.norm_layer.evaluate(
                    in_vectors, din_vectors)
                if whole_output:
                    activation_stack.apend(in_vectors)
            for layer in self._layers:
                out_vectors, dout_vectors = layer.evaluate(in_vectors, din_vectors)
                in_vectors, din_vectors = out_vectors, dout_vectors
                if whole_output:
                    activation_stack.append(in_vectors)

            dout_vectors = np.squeeze(dout_vectors, axis=(2))
            if add_offset:
                out_vectors += self.offset
            if whole_output:
                return Out(out_vectors, dout_vectors, activation_stack)
            return Out(out_vectors, dout_vectors, None)

        if self.norm_layer:
            in_vectors = self.norm_layer.evaluate(in_vectors)
            if whole_output:
                activation_stack.append(in_vectors)

        for layer in self._layers:
            out_vectors = layer.evaluate(in_vectors)
            in_vectors = out_vectors
            if whole_output:
                activation_stack.append(in_vectors)

        if add_offset:
            out_vectors += self.offset
        if whole_output:
            return Out(out_vectors, None, activation_stack)
        return Out(out_vectors, None, None)

    def tf_evaluate(self, features_vector, compute_gradients=False):
        in_vectors = features_vector

        if self.norm_layer:
            with tf.variable_scope("species_{}_layer_{}".format(
                    self.name, self.norm_layer.name)):
                logger.debug('inserting layer - normalization')
                in_vectors = self.norm_layer.tf_evaluate(in_vectors)

        for l_idx, layer in enumerate(self._layers):
            logger.debug('inserting layer - %d', l_idx)
            with tf.variable_scope("species_{}_layer_{}".format(self.name, layer.name)):
                out_vectors = layer.tf_evaluate(in_vectors)
                in_vectors = out_vectors

        if compute_gradients:
            return out_vectors, tf.gradients(out_vectors, features_vector)
        return out_vectors

    def customize_network(self,
                          *,
                          feature_size=None,
                          layers_size=None,
                          trainables=None,
                          behaviors=None,
                          activations=None,
                          offset=None,
                          override_wb=None,
                          norm_layer=None):
        """Inplace customization of an already defined network.
        This function must be used to consistently update an already
        existent neural network with new parameters.
        This function accept only named parameters.

        Parameters
        ----------
        feature_size: int, optional
            Size of the feature vector, or Input vector, or Gvector
            depending on notations.
            This parameter can be used to change the feature size of
            an already defined network.
            For now different network with different feature size
            are not supported by the code so this feature has no
            practical examples.
        layers_size: list of integers, optional
            List of integers that describe the layers shape:
            For example a list like [128, 64, ..., 4, 1] will create a network
            like:
            - in layer: feature_size: 128
            - first layer: 128:64
            - second layer: 64:...
            - ...
            - last layer: ...:4
            - out layer:  4:1
            This function can used to change the network shape. For example in
            the case were we want to make a network for a given specie
            different.
            In combination with the *behaviors* keyword it can be used also to
            perform some limited on the fly brain surgery to the neural network.

            ..alert:
              Changing the layer depth *always* require the specification of the
              parameters to describe the new layers:
              - trainables parameter must be specified
              - activations parameter must be specified
              *If this isn't done* default values are loaded. For default values
              refer to class documentations.
              If a new layers_size is passed but the depth is kept equal old values
              will be preserved.
        trainables: list of bools, optional
            One for each layer to set a layer as trainable or not.
            Used to update values or define values for new layers
        activations: list of integer, optional
            One for each layer to set the activation kind of the layer
            Used to update values or define values for new layers
        Offset: float, optional
            Network offset values
        behaviors: list of string, optional
            This parameter define what to do on weight an biases.
            If used it must be *specified for every* layer in any case.
            accepted values are
            - new :: empty the weights and the biases of the layer.
                     This must be used when a network architecture is deepened
                     to initialize new layers with zeros.
                     If no behavior is provided the default zero value is
                     applied to all the layers.
            - keep :: keep the weights and biases that are already present.
                      This is practical in two cases:
                      - when we want to reset a set of layers while restarting
                        a training
                      - when we change the architecture and we want to keep
                        the layers closed to the input
            - load :: Load a layer, the weight and biases values are loaded from
                      the provided override_wb parameter.
            .. alert::
                If the network depth is changed the default behaviors for every
                layer is 'new'
        override_wb: dict, optional
            Used when load is specified in behaviors.
            format: {layer_key: (w, b)}
            - layer_key = number, starting from 0
            - w, b = weights and biases with a correct shape for the target
                     layer
        norm_layer: Layer, optional
            update the normalization layer
        """
        # TODO: substitute behaviors with enum

        # is easier to recreate a network, this will automatically apply all the
        # consistency check
        # trainable and activation falgs are layer properties
        if (layers_size and (len(layers_size) == len(self._layers))) or not layers_size:
            if not trainables:
                trainables = self.layers_trainable
            if not activations:
                activations = self.layers_activation
            if not behaviors:
                for s_a, s_b in zip(self.layers_size,
                                    layers_size if layers_size else self.layers_size):
                    if s_a != s_b:
                        behaviors = ['new'] * len(layers_size)
                else:
                    behaviors = ['keep'] * len(self._layers)
        else:
            if not behaviors:
                behaviors = ['new'] * len(layers_size)

        new_net = A2affNetwork(feature_size if feature_size else self.feature_size,
                               layers_size if layers_size else self.layers_size,
                               'dummy',
                               trainables=trainables,
                               activations=activations,
                               norm_layer=norm_layer if norm_layer else self.norm_layer)
        old_layers = self._layers

        # trainable and activations are layer properties
        self.feature_size = new_net.feature_size
        self._layers = new_net._layers
        if norm_layer:
            self.norm_layer = norm_layer  # checkmate :)

        if len(behaviors) != len(self.layers_size):
            raise ValueError(f'{self.name} layer_behavior != layer sizes')

        for i, behavior in enumerate(behaviors):
            if behavior == 'keep':
                if i >= len(old_layers):
                    raise ValueError('requested to default a not available layer')
                self._layers[i].w_value = old_layers[i].w_value
                self._layers[i].b_value = old_layers[i].b_value
            elif behavior == 'load':
                tmp_layer = self._layers[i]
                tmp_layer.w_value, tmp_layer.b_value = override_wb[i]
            elif behavior == 'new':
                # nothing to do, the layer is already the new version
                pass
            else:
                raise ValueError('undefined behavior for layer')

        if offset:
            self.offset = offset

    @property
    def tf_ckpt_elements(self):
        """ Regexps for TF checkpoint, this call is meant to be used in a
        recursive script.

        Returns
        ------
        names: list of str
            each string is a regexp that need to be matched in the tf namespace
            tree to access the subordinate object data.
        objects: list of subordinate object
            Subordinate object that can be initialized with the information
            contained in the tf tree.
            Only Layer objects can be network subordinate objects.
        """
        names = []
        objects = []

        for l_idx, layer in enumerate(self._layers):
            names.append("species_{}_layer_{}".format(self.name, layer.name))
            objects.append(layer)
        if self.norm_layer:
            names.append("species_{}_layer_{}".format(self.name, self.norm_layer.name))
            objects.append(self.norm_layer)
        return names, objects

    def to_json(self):
        """Serialize the a2ff neural network.
        Returns
        -------
        json_data: dict
            dictionary data describing the network object
        wbs: dict
            dictionary data with the weight and biases
             {name:variable, ..}
        """
        layers_data = []
        wbs = {}
        prefix = self.__class__._wb_prefix.format(self.name) + '_'
        for layer in self._layers:
            layer_data, wb = layer.to_json()
            layers_data.append(layer_data)
            wbs.update({prefix + name: value for name, value in wb.items()})

        json_data = {
            'feature_size': self.feature_size,
            'layers': layers_data,
            'name': self.name,
            'offset': self.offset
        }

        if self.norm_layer:
            norm_layer_data, wb = self.norm_layer.to_json()
            wbs.update({prefix + name: value for name, value in wb.items()})
            json_data['norm_layer'] = norm_layer_data

        return json_data, wbs

    @classmethod
    def from_json(cls, json_data, base_path=''):
        """ Load the network from a dump.

        Parameters
        ----------
        json_data: dict
            json serialization of the class.
        base_path: string or path object
            a path to search for weights and biases numpy files

        Returns
        -------
        network:
            A new network object
        """
        feature_size = json_data['feature_size']
        layers_json = json_data['layers']
        name = json_data['name']
        base_path = os.path.join(base_path, cls._wb_prefix.format(name))

        if 'norm_layer' in json_data:
            norm_layer = Layer.from_json(json_data['norm_layer'], base_path)
        else:
            norm_layer = None

        # create a fake network with proper initialization of all
        # non layer dependent parameters, architecture is dummy
        network = cls(feature_size, [1, 1],
                      json_data['name'],
                      offset=json_data['offset'],
                      norm_layer=norm_layer)
        # initialize the layers loading the from files
        real_layers = []
        for layer_json in layers_json:
            real_layers.append(Layer.from_json(layer_json, base_path))

        # substitue the network, calling a customize should be better
        # but this op should be safe enough
        network._layers = real_layers
        return network
