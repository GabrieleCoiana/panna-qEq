###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from contextlib import nullcontext
from dataclasses import dataclass

import logging

from typing import Tuple
from typing import Sequence
from typing import Optional
from typing import List
from typing import Dict

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint

from panna.neuralnet.activations import get_activation

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfigBase():
    name: Optional[str] = None
    compute_jacobian: bool = False
    activation_stack: bool = False
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
    bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
    offset: float = 0.0

    @property
    def is_ready(self):
        if self.name is not None:
            return True


@dataclass
class NetworkConfig(NetworkConfigBase):
    feature_size: Optional[int] = None
    layers_size: Optional[Sequence[int]] = None
    layers_trainable: Optional[Sequence[bool]] = None
    layers_activation: Optional[Sequence[str]] = None
    preload_wbs: Optional[Sequence[np.ndarray]] = None
    layers_prunable: Optional[Sequence[bool]] = None
    layers_mask: Optional[Sequence[np.ndarray]] = None
    metadata: Optional[dict] = None
    metadata_dir: Optional[str] = None
    # TODO normalization layer config: TBD

    @property
    def is_ready(self):
        conditions = [
            super().is_ready, self.feature_size is not None, self.layers_size
            is not None
        ]
        return all(conditions)


class MaskConstraint(Constraint):
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, w):
        return w * self.mask

    def get_config(self):
        return {'mask': self.mask}


class A2affNetwork(tf.keras.layers.Layer):
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
    layers_trainable: list of bools, optional
        one for each layer to set a layer as trainable or not
        default is all layer trainables
    layers_activation: list of integer, optional
        one for each layer to set the activation kind of the layer
        default is: gaussian:gaussian:....:linear
    offset: float, optional
        network offset value
        default is: 0.0
    norm_layer: Layer
        a layer applied before anything else for data normalization

   compute_gradients: boolean
       if gradients w.r.t. descriptor need to be computed
       (memory intensive)
   compute_gradients: boolean
       if activations needs to be recovered for further analysis
    Raises
    ------
    ValueError if:
        - network_wb is too small
        - trainables is too small
        - activations is too small
    """

    _network_type = 'a2aff'

    def __init__(
        self,
        feature_size: int,
        layers_size: Sequence[int],
        name: str,
        preload_wbs: Optional[Sequence[np.ndarray]] = None,
        layers_trainable: Optional[Sequence[bool]] = None,
        layers_activation: Optional[Sequence[str]] = None,
        offset: Optional[float] = None,
        compute_jacobian: bool = False,
        activation_stack: bool = False,
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        layers_prunable: Optional[Sequence[bool]] = None,
        layers_mask: Optional[Sequence[np.ndarray]] = None,
        metadata: Optional[dict] = None,
        metadata_dir: Optional[str] = None,
        constrain_even_terms: bool = False,
        min_eigenvalue: float = 0.0
    ):
        # norm_layer Optional[tf.Module]=None):
        super().__init__(name=name)

        if not layers_trainable:
            # default behavior: all trainable
            layers_trainable = [True for i in range(len(layers_size))]
        if not layers_activation:
            # default behavior: gaussian:gaussian:....:linear
            layers_activation = [1 for i in range(len(layers_size) - 1)] + [0]

        if preload_wbs and (len(layers_size) != len(preload_wbs)):
            raise ValueError('network wb parameters are not enough')
        if len(layers_size) != len(layers_trainable):
            raise ValueError('trainables parameters are not enough')
        if len(layers_size) != len(layers_activation):
            raise ValueError('activations parameters are not enough')

        self.feature_size = feature_size
        self._compute_jacobian = compute_jacobian
        self._activation_stack = activation_stack
        self.offset = offset or 0.0

        self._layers_trainable = layers_trainable
        self._layers_size = layers_size
        self._layers_activation = layers_activation
        self._preload_wbs = preload_wbs
        self._layers_prunable = layers_prunable
        self._layers_mask = layers_mask

        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._metadata = metadata
        self._metadata_dir = metadata_dir

        self._layers = []
        self._order = self._layers_size[-1] - 1 
        self._min_eigenvalue = min_eigenvalue
        self._constrain_even_terms = constrain_even_terms

        # self._norm_layer = norm_layer
        # input_size = norm_layer.wb_shape[1] if norm_layer else self.feature_size

    def build(self, input_shape):
        for i, output_size in enumerate(self._layers_size):
            _, activation = get_activation(self._layers_activation[i])
            if i>0:
                input_size = self._layers_size[i-1]
            else:
                input_size = self.feature_size
            if self._preload_wbs and self._preload_wbs[i]:
                w_values, b_values = self._preload_wbs[i]
                kernel_initializer = tf.constant_initializer(w_values)
                bias_initializer = tf.constant_initializer(b_values)
            else:
                # Glorot initialization
                stddev = np.sqrt(2.0 / (output_size + input_size))
                # kernel_initializer, must be a callable class
                # next time, this is called by the make_variable
                # getter
                kernel_initializer = tf.random_normal_initializer(mean=0.0,
                                                                  stddev=stddev,
                                                                  seed=None)
                bias_initializer = tf.zeros
            if self._layers_prunable and self._layers_prunable[i]:
                # onemask = np.ones(self.layers_shaping[i])
                constraint = MaskConstraint(self._layers_mask[i])
            else:
                constraint = None
            with tf.name_scope(f'layer_{i}'):
                layer = Dense(
                    units=output_size,
                    activation=activation,
                    use_bias=True,  # TODO False if we reimplement rbf
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer,
                    kernel_constraint=constraint,
                    trainable=self._layers_trainable[i],
                    name=f'layer_{i}')
                layer.build(input_shape)
            input_shape = output_size
            self._layers.append(layer)
        self.built = True

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
        return self._layers_size

    # @property
    # def norm_layer(self):
    #     return self._norm_layer

    # @norm_layer.setter
    # def norm_layer(self, value):
    #     if value.b_shape[0] == self[0].wb_shape[0]:
    #         self._norm_layer = value
    #     else:
    #         raise NotImplementedError('operation not yet supported')

    @property
    def layers_shaping(self) -> List[Tuple[float, float]]:
        """
        two elements for each layer e.g. [(shape_w, shape_b),.....]
        """
        l_s = self.layers_size
        layers_shaping = [(self.feature_size, l_s[0])] +\
                         [(l_s[i], l_s[i + 1]) for i
                          in range(len(l_s) - 1)]
        return layers_shaping

    @property
    def layers_trainable(self) -> Tuple[bool, ...]:
        """ If a layer is trainable or not
        """
        return self._layers_trainable

    @layers_trainable.setter
    def layers_trainable(self, value: Tuple[bool, ...]):
        if self.built:
            raise ValueError('After building setter is disabled')
        if len(value) != len(self.layers_size):
            raise ValueError('passed trainable vector is '
                             'not compatible with current layer size')
        self._layers_trainable = value

    @property
    def layers_activation(self) -> Tuple[int, ...]:
        """ activation for each layer
        """
        return self._layers_activation

    @layers_activation.setter
    def layers_activation(self, value: Tuple[int, ...]):
        if self.built:
            raise ValueError('After building setter is disabled')
        if len(value) != len(self.layers_size):
            raise ValueError('passed act vector is '
                             'not compatible with current layer size')
        self._layers_activation = value

    @property
    def wbs_tensors(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ list of tuple
        two numpy_array for each layer, weights and biases values
        """

        if self.built:
            wb = []
            for layer in self._layers:
                wb.append((layer.kernel.numpy(), layer.bias.numpy()))
            return wb
        else:
            if self._preload_wbs is not None:
                return self._preload_wbs
        raise ValueError('WB not available, model must be built')

    def to_json(self):
        """Serialize the a2ff neural network.
        Returns
        -------
        json_data: dict
            dictionary data describing the network object
        """

        json_data = {
            'type': self._network_type,
            'name': self.name,
            'feature_size': self.feature_size,
            'layers': self._layers_size,
            'activations': self.layers_activation,
            'offset': self.offset
        }

        return json_data

    def __getitem__(self, index):
        if self.built:
            return self._layers[index]
        raise ValueError(f'network {self.name} not built')

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

    def call(self, inputs: tf.Tensor):
        """ Network evaluation

        Paramenters
        -----------
        features_vector: Tensor
            standard input vector

        Return
        ------
        jacobian: Tensor
            batch_size, output_size, input size
        """
        in_tensor = inputs
        # if self.norm_layer:
        #     with tf.variable_scope("species_{}_layer_{}".format(
        #             self.name, self.norm_layer.name)):
        #         logger.debug('inserting layer - normalization')
        #         in_vectors = self.norm_layer.tf_evaluate(in_vectors)
        if self._activation_stack:
            stack = []

        if self._compute_jacobian:
            cm = tf.GradientTape(persistent=True)
        else:
            cm = nullcontext()

        with cm as g:
            if g is not None:
                g.watch(inputs)
            
            for l_idx, layer in enumerate(self._layers):
                out_tensor = layer(in_tensor)
                # if self._activation_stack:
                #     stack.append(out_tensor.numpy())
                in_tensor = out_tensor


            if self._order>=1:
                #In this case, the out_tensors returns (order + 1) elements.
                E_collection = []
                if self._compute_jacobian:
                    dE_collection = [] 
                for k in range(self._order+1):
                    Ek = out_tensor[:, k]
                    if k==0:
                        Ek += self.offset
                    if self._constrain_even_terms and self._order==2:
                        if k != 0 and k % 2 == 0:
                            #add a regularizer such that Ek is greater than -min_eigenvalue
                            Ek = -self._min_eigenvalue + tf.nn.softplus(Ek) + 1e-3

                    E_collection.append(Ek)
                    if self._compute_jacobian:
                        dEk_dg = g.batch_jacobian(Ek[:,tf.newaxis], inputs)
                        dE_collection.append(dEk_dg)
                if self._compute_jacobian:
                    return E_collection, dE_collection
                else:
                    return E_collection 

            out_tensor += self.offset
        if self._compute_jacobian and self._order==0:
            return out_tensor, g.batch_jacobian(out_tensor, inputs)
        return out_tensor
