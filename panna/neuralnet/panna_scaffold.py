###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from collections import namedtuple
from copy import deepcopy
from distutils import util
from functools import partial
import json
import logging
import os
import struct
import mendeleev
import numpy as np
import tensorflow as tf

from lib.errors import NetworkNotAvailableError

from .a2affnetwork import A2affNetwork, normalize_network_type
from .basic_scaffold import ScaffoldBasic
from .layers import Layer
from .tfr_parsers import parse_fn

logger = logging.getLogger(__name__)

def _parse_network_section(net_params):
    nn_type = net_params.get('nn_type', 'a2aff')
    nn_type = normalize_network_type(nn_type)

    params = {}
    # set architecture
    params['feature_size'] = net_params.getint('g_size')
    params['layers_size'] = net_params.get_network_architecture('architecture')

    # set trainability
    params['trainables'] = net_params.get_network_trainable('trainable', None)

    # set activations
    params['activations'] = net_params.get_network_act('activations', None)

    # output_offset
    params['offset'] = net_params.getfloat('output_offset', None)

    # check for normalizations
    # in the future this operations must be moved in the
    # input pipeline
    norm_tech = net_params.get('normalization_technique', None)

    if norm_tech == 'pca':
        pca_matrix = np.load(net_params.get('pca_matrix'))
        bias_vector = np.load(net_params.get('g_mean'))
        assert pca_matrix.shape[0] == bias_vector.shape[0]
        # creating the normalization layer
        norm_layer = Layer(pca_matrix.shape, False, 0, 'norm_PCA')
        norm_layer.w_value = pca_matrix
        norm_layer.b_value = -bias_vector @ pca_matrix
        params['norm_layer'] = norm_layer
    elif norm_tech == 'std':
        path_to_normalized_gvec = net_params.get('path_to_gvec_normalization')
        with open(path_to_normalized_gvec) as f:
            normalization_json = json.load(f)
            average_g = np.array(normalization_json['average_G'])
            sigma_g = np.array(normalization_json['sigma_G'])

        e_n = np.identity(sigma_g.shape[0])
        norm_layer = Layer(e_n.shape, False, 0, 'norm_std')
        norm_layer.w_value = e_n / sigma_g
        norm_layer.b_value = -average_g / sigma_g

        if net_params.getboolean('masked_gvec', False):
            path_to_gvec_mask = net_params.get('path_to_gvec_mask')
            index_to_use = np.loadtxt(path_to_gvec_mask, dtype='int')
            bias_mask = np.zeros(params['feature_size'])
            bias_mask[index_to_use] = 1
            weight_mask = np.diagflat(bias_mask)
            norm_layer.w_value = norm_layer.w_value * weight_mask
            norm_layer.b_value = norm_layer.b_value * bias_mask

        params['norm_layer'] = norm_layer
    return nn_type, params


def _parse_default_network(default_net_params):
    """ parse a default network for PANNA

    Parameters
    ----------
    default_net_params: default network section from config

    Returns
    -------
    network: network object or None
        A default network if one has been found, None otherwise
    """
    _nn_type, params = _parse_network_section(default_net_params)

    # set architecture
    if params['feature_size'] and params['layers_size']:
        logger.info('Found a default network!')
        logger.info('This network size will be used as default for all species unless '
                    'specified otherwise')
        network = A2affNetwork(name='default', **params)
        logger.info(network)
    else:
        network = None
    return network


def _parse_specie_network(species_name, species_config, system_scaffold):
    """ parse a specie config

    Parameters
    ----------
    species_name:
        a species name
    species_config:
        a species config
    system_scaffold:
        a scaffold where to perform the operation of storing the species network
    """

    nn_type, params = _parse_network_section(species_config)

    # layers behavior flag:
    layers_behavior = species_config.get_network_behavior('behavior', None)

    # network_wb
    override_wb = {}
    if layers_behavior:
        for layer_number, behavior in enumerate(layers_behavior):
            if behavior == 'load':
                w_file_name = species_config.get('layer{}_w_file'.format(layer_number))
                b_file_name = species_config.get('layer{}_b_file'.format(layer_number))
                if not (w_file_name or b_file_name):
                    raise ValueError('not passed file names')
                override_wb[layer_number] = (
                    np.load(w_file_name) if w_file_name else np.empty(0),
                    np.load(b_file_name) if b_file_name else np.empty(0))

    params['network_type'] = nn_type
    params['behaviors'] = layers_behavior
    params['override_wb'] = override_wb

    _load_network_on_scaffold(system_scaffold, species_name, **params)


def _load_network_on_scaffold(system_scaffold, species_name, **kwargs):
    network_type = kwargs.pop('network_type', None)
    layers_size = kwargs.get('layers_size', None)
    try:
        species_network = system_scaffold[species_name]
        if species_network.network_type != network_type:
            raise ValueError('Undefined behavior, customization is not possible')
        species_network.customize_network(**kwargs)

    except NetworkNotAvailableError:
        if not layers_size:
            raise ValueError('not available network structure')
        # triggered only if a default fall back is not present
        network_wb = [(np.empty(0), np.empty(0)) for i in range(len(layers_size))]
        for key, value in kwargs.get('override_wb', {}).items():
            network_wb[key] = value

        if network_type == 'a2ff':
            species_network = A2affNetwork(
                kwargs['feature_size'],
                layers_size,
                species_name,
                network_wb,
                kwargs.get('trainables', None),
                kwargs.get('activations', None),
                kwargs.get('offset', None),
            )
        else:
            raise ValueError('type not supported {}'.format(network_type))

        system_scaffold[species_name] = species_network
    logger.info(system_scaffold[species_name])


class PannaScaffold(ScaffoldBasic):
    """ A recipe for the PANNA basic network

    Parameters
    ----------
    config: config paraser object, optional,
        A config to extract the quantities needed for the
        networks setup.
        If no config is provide scaffold is empty.
    name: str, optional,
        A name for the network.
    """
    _scaffold_type = 'PANNA'
    _version = 'v1'

    def __init__(self, config=None, name='PANNA_scaffold'):
        super().__init__(config, name)

        self._default_network = None
        self._networks = {}
        self._zeros = {}

        # default options
        self._atomic_sequence = []
        self._en_rescale = 1.0
        self._sparse_derivatives = False

        if not config:
            return

        self.configure(config)

    def configure(self, config):
        """ Configure the scaffold starting from a config file

        Parameters
        ----------
        config: config parser object
            for now read the main documentation on the file structure
        """
        # === Data option part ===
        if 'DATA_INFORMATION' in config:
            data_params = config['DATA_INFORMATION']
            self._atomic_sequence = data_params.get_comma_list(
                'atomic_sequence', self._atomic_sequence)
            offsets = data_params.get_comma_list_floats('output_offset', [])
            self._en_rescale = data_params.getfloat('energy_rescale', self._en_rescale)
            self._sparse_derivatives = data_params.getboolean(
                'sparse_derivatives', self._sparse_derivatives)
        else:
            offsets = []
        if offsets and self._atomic_sequence:
            if len(offsets) != len(self._atomic_sequence):
                raise ValueError('output offset must be of '
                                 'the same size as atomic sequence')
            self._zeros = dict(zip(self._atomic_sequence, offsets))
        elif offsets and not self._atomic_sequence:
            raise ValueError('output offset can not be specified '
                             'without atomic_sequence')

        # === NETWORK option part ===
        # default network to be loaded
        default_networks_metadata_folder = None

        if 'DEFAULT_NETWORK' in config:
            default_net_params = config['DEFAULT_NETWORK']
            # search for a default network stored in file
            default_networks_metadata_folder = default_net_params.get(
                'networks_metadata', default_networks_metadata_folder)
            self._default_network = _parse_default_network(default_net_params)
        else:
            logger.info('No default network is found, '
                        'all species must be fully specified.')

        # load species networks form metadata
        if default_networks_metadata_folder:
            logger.warning('Networks in %s will be loaded',
                           default_networks_metadata_folder)
            self._load_dump(default_networks_metadata_folder)

        # parsing single atomic species
        # if a default is specified every species behave as the default
        # if no default is specified than every species must be fully described
        for species in self._atomic_sequence:
            if species in config:
                logger.info('=={}== Found network specifications'.format(species))
                species_config = config[species]
                _parse_specie_network(species, species_config, self)

        # === Training option specific part ===
        self._loss_func = 'quad'
        self._floss_func = 'quad'
        self._forces_cost = 0.0

        if 'TRAINING_PARAMETERS' in config:
            train_params = config['TRAINING_PARAMETERS']

            self._loss_func = train_params.get('loss_function', self._loss_func)
            self._floss_func = train_params.get('floss_function', self._floss_func)
            self._forces_cost = train_params.getfloat('forces_cost', self._forces_cost)
        if self._forces_cost > 0:
            self._tf_require_gradients = True
        else:
            self._tf_require_gradients = False
        self._usable_for_training = True

    def __getitem__(self, value):
        """ Recover a network for a given species.

        Returns
        -------
        Network object
            - The requested network if already present.
            - a copy of the default network,
              If the requested network is not present,
              In this case the atomic sequence gets also updated

        Raises
        ------
        NetworkNotAvailableError
            If the requested network is not available and there
            is no default network
        """
        tmp_network = self._networks.get(value, None)

        if tmp_network:
            return tmp_network

        if self._default_network:
            tmp_network = deepcopy(self._default_network)
            tmp_network.name = value
            tmp_network.offset = self._zeros.get(value, 0.0)
            self._networks[value] = tmp_network
            if value not in self._atomic_sequence:
                self._atomic_sequence.append(value)
            return tmp_network

        raise NetworkNotAvailableError('default network not available')

    def __setitem__(self, index, value):
        """Set a network for a given species.

        If the network for the given species is already present in the scaffold
        it can not be overwritten but must be recovered with the getter and
        changed.

        If the network is not stored in the scaffold atomic sequence then it
        will be added as the last one

        Energetic zeros are update when new networks are stored.

        Note: The getter return the instance of the network!

        Parameters
        ----------
        Index: string
            Name of the network
        Value: Network object
            A network

        Notes
        -----

        """
        if index in self._networks:
            raise ValueError('network already present, '
                             'recover with getter and change in place')
        if index != value.name:
            raise ValueError('assign inconsistent '
                             '{} != {}'.format(index, value.name))
        self._zeros[index] = value.offset
        if index not in self._atomic_sequence:
            self._atomic_sequence.append(index)
        self._networks[index] = value

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        try:
            specie = self.atomic_sequence[self._index]
            result = self[specie]
        except IndexError:
            raise StopIteration
        self._index += 1
        return result

    def dump_scaffold_lammps(self, folder, filename, extradata):
        # TODO
        pass

    def dump_scaffold_metadata(self, folder, extra_data=None):
        """ Create and write a scaffold metadata needed to reconstruct it.

        Stored in json.

        Parameters
        ----------
        folder: str
            Where to save the json file containing the
            folder structure.
        extra_data: dict, optional
            Additional arbitrary fields added to the json.
            They must be serializable by the json package.

        Raises
        ------
        ValueError
            If the extra_data try to override internal keys
        """
        metadata = extra_data if extra_data else {}
        reserved_field = [
            'version', 'scaffold_type', 'networks_type', 'networks_species'
            'networks'
        ]
        if set(reserved_field) in set(metadata):
            raise ValueError('requested key for extra metadata is ' 'already in use')

        metadata['version'] = self._version
        metadata['scaffold_type'] = self.__class__._scaffold_type
        metadata['networks_type'] = [self[x].network_type for x in self.atomic_sequence]
        metadata['networks_species'] = self.atomic_sequence
        metadata['networks'] = [self[x].to_json()[0] for x in self.atomic_sequence]
        with open(os.path.join(folder, 'networks_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def dump_scaffold_data(self, folder):
        """ Create write all the data relevant arrays contained in the
        scaffold.

        Parameters
        ----------
        folder: str
            Where to save the json file containing the
            folder structure.
        """
        # For now only networks array are dumped
        for specie in self.atomic_sequence:
            for name, array in self[specie].to_json()[1].items():
                path = os.path.dirname(os.path.join(folder, name))
                if not os.path.isdir(path):
                    os.makedirs(path)
                # TODO: this can be moved deep to the layer object
                np.save(os.path.join(folder, name), array)

    def _load_dump(self, folder):
        """method to serialize the class
        """
        with open(os.path.join(folder, 'networks_metadata.json')) as f:
            metadata = json.load(f)

        self._usable_for_validation = True

        for n_type, n_specie, n in zip(metadata['networks_type'],
                                       metadata['networks_species'],
                                       metadata['networks']):
            if n_type == 'a2ff':
                network = A2affNetwork.from_json(n, folder + '/')
                self[n_specie] = network
                for layer in network:
                    w, b = layer.wb_value
                    if w.size == 0 or b.size == 0:
                        #logger.info(f'loaded {n_specie} network is not suitable '
                        #            'for validation')
                        self._usable_for_validation = False
            else:
                raise ValueError('Unknown network type')

    @classmethod
    def load_dump(cls, folder, **kwargs):
        new_scaffold = cls(**kwargs)
        new_scaffold._load_dump(folder)
        return new_scaffold

    def evaluate(self, example, forces_flag=False, **kwargs):
        """ evaluate the network with numpy

        Parameters
        ----------
            example:
            force_flag:
            add_offset:
            gvec_norm_vectors: if not empty, a dictionary containing the
                   average and sdtev of input G vectors, used for normalization.
            masked_gvec_vector:
                   if not None, a np.array containing the indexes to use as
                   mask to sparsify the input gvector.

        Return
        ------
            numpy array
        """
        Out = namedtuple("Out", ["e", "fs", "stack"])
        stack = []
        n_atoms = example.n_atoms
        species_vect, g_vects = example.species_vector.copy(), example.gvects.copy()
        energy = 0
        gsize = example.g_size
        g_vects = np.reshape(np.asarray(g_vects), [-1, gsize])

        if forces_flag:
            forces = np.zeros(n_atoms * 3)
            dgs = example.dgvects.copy()

        # TODO verify if this is no more needed
        # if masked_gvec_vector is not None:
        #     mask = np.zeros(gsize)
        #     mask[masked_gvec_vector] = 1
        #     g_vects = g_vects*mask
        #     if forces_flag:
        #         dgs = dgs*mask[None, :, None]

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            network = self[species_symbol]

            # select per species gvects
            species_indices = np.where(species_vect == species_idx)
            if len(species_indices[0]) > 0:
                in_gs = g_vects[species_indices]
                if forces_flag:
                    in_dgs = dgs[species_indices]
                    network_eval = network.evaluate(in_gs, in_dgs, **kwargs)
                    # minus_partial_forces = network_eval.d_g_d_e
                    forces -= np.sum(network_eval.d_g_d_e, axis=(0))
                else:
                    network_eval = network.evaluate(in_gs, **kwargs)

                if network_eval.stack:
                    stack.append(network_eval.stack)

                energy += network_eval.e.sum()

        return Out(energy, forces if forces_flag else None, stack if stack else None)

    def tf_compute_pa_energy(self, batch_of_species, batch_of_gvects, batch_size):
        """ Perfrom the evaluation of a batch predicting the energies
        and computing the energy
        """
        # Species sorted list of lists [specie 1 list ,specie 2 list,..]:
        # atomic energies for all the atoms in the batch, species sorted

        # hack, for now we support only one feature size
        g_size = self[self.atomic_sequence[0]].feature_size

        atomic_energies = []
        # for each atom, the example number in the batch,
        # to keep track of where they came from, species sorted
        max_number_atoms = tf.shape(batch_of_species)[1]
        partitions = tf.cast(tf.reshape(batch_of_species, [-1]), tf.int32)
        array_gvects = tf.reshape(batch_of_gvects, [-1, g_size])

        atomic_index_list = tf.dynamic_partition(
            tf.range(batch_size * max_number_atoms), partitions, self.n_species + 1)
        # List of derivatives wrt Gs, needed for differentiation
        des_dgs = []

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            logger.debug('crating network: %d, %s', species_idx, species_symbol)
            # recover the network
            network = self[species_symbol]

            s_n_atoms = tf.shape(atomic_index_list[species_idx])[0]
            s_input_features = tf.gather(array_gvects, atomic_index_list[species_idx])

            if self._tf_require_gradients:
                s_atomic_energies, s_des_dgs = network.tf_evaluate(
                    s_input_features, True)
                # dEs_dGs has only one element by construction
                s_des_dgs = s_des_dgs[0]
                des_dgs.append(s_des_dgs)
            else:
                s_atomic_energies = network.tf_evaluate(s_input_features)

            s_atomic_energies = tf.reshape(s_atomic_energies, [-1])

            mean, var = tf.nn.moments(s_atomic_energies, axes=[0])
            tf.summary.scalar(
                "3.Mean_energy/S{}.{}".format(species_idx, species_symbol), mean)
            tf.summary.scalar("4.Std_energy/S{}.{}".format(species_idx, species_symbol),
                              tf.sqrt(var))
            tf.summary.scalar("5.N_atoms/S{}.{}".format(species_idx, species_symbol),
                              s_n_atoms)

            atomic_energies.append(s_atomic_energies)

        # recover how many empty slot we have in the species matrix
        # and fill the predicted energy for those with zeros
        n_placeholder_in_species_matrix = tf.shape(atomic_index_list[self.n_species])[0]
        fake_species_e_contrib = tf.zeros(n_placeholder_in_species_matrix)

        atomic_energies.append(fake_species_e_contrib)

        # reconstruct the matrix
        reconstructed_energies_matrix = tf.dynamic_stitch(atomic_index_list,
                                                          atomic_energies)
        reconstructed_energies_matrix = tf.reshape(reconstructed_energies_matrix,
                                                   [batch_size, max_number_atoms])

        if self._tf_require_gradients:
            des_dgs.append(tf.zeros([n_placeholder_in_species_matrix, g_size]))
            reconstructed_des_dgs_matrix = tf.dynamic_stitch(atomic_index_list, des_dgs)
            reconstructed_des_dgs_matrix = tf.reshape(
                reconstructed_des_dgs_matrix, [batch_size, max_number_atoms, g_size])

            # TODO verify if this is no more needed
            # if self._masked_gvec:
            #     reconstructed_des_dgs_matrix = reconstructed_des_dgs_matrix*mask

        # create a hot matrix to count atom per example
        atoms_presence = tf.where(tf.less(batch_of_species, self.n_species),
                                  tf.ones(tf.shape(batch_of_species)),
                                  tf.zeros(tf.shape(batch_of_species)))
        batch_n_atoms = tf.reduce_sum(atoms_presence, axis=1)

        # helper
        _find_idx_from_str = {
            symbol: idx
            for idx, symbol in enumerate(self.atomic_sequence)
        }

        for x in tf.get_collection(tf.GraphKeys.WEIGHTS):
            name = x.op.name.split("/")[0]
            nameparts = name.split("_")
            tf.summary.histogram(
                "S{}.{}/W{}".format(nameparts[1], _find_idx_from_str[nameparts[1]],
                                    nameparts[3]), x)
        for x in tf.get_collection(tf.GraphKeys.BIASES):
            name = x.op.name.split("/")[0]
            nameparts = name.split("_")
            tf.summary.histogram(
                "S{}.{}/b{}".format(nameparts[1], _find_idx_from_str[nameparts[1]],
                                    nameparts[3]), x)

        if self._tf_require_gradients:
            return reconstructed_energies_matrix, batch_n_atoms, \
                reconstructed_des_dgs_matrix

        return reconstructed_energies_matrix, batch_n_atoms

    def tf_energy_loss(self, batch_delta_e, batch_natoms):
        """ Cost function for this scaffold

        Parameters
        ----------
            batch_delta_e: energy differences
            batch_natoms: number of atoms

        Returns
        -------
            the loss value
            tensor with delta_e for each element of the batch
        """

        with tf.name_scope('energy_loss_calculation'):

            # loss function
            batch_delta_e2 = tf.square(batch_delta_e)

            if self._loss_func == "quad":
                tot_loss = tf.reduce_sum(batch_delta_e2, name='1.E_Loss-Delta_E2')
            elif self._loss_func == "exp_quad":
                tot_loss = 0.5 * tf.exp(2.0 * tf.reduce_sum(batch_delta_e2),
                                        name='1.E_Loss-Exp_Delta_E2')
            elif self._loss_func == "quad_atom":
                tot_loss = tf.reduce_sum(tf.div(batch_delta_e2,
                                                tf.square(batch_natoms)),
                                         name='1.E_Loss-Delta_E2_div_Natom2')
            elif self._loss_func == "quad_std_atom":
                tot_loss = tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms),
                                         name='1.E_Loss-Delta_E2_div_Natom')
            elif self._loss_func == "exp_quad_atom":
                tot_loss = 0.5 * tf.exp(2.0 * tf.reduce_sum(
                    tf.div(batch_delta_e2, tf.square(batch_natoms))),
                    name='1.E_Loss-Exp_Delta_E2_div_Natom2')
            elif self._loss_func == "exp_quad_std_atom":
                tot_loss = 0.5 * tf.exp(
                    2.0 * tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms)),
                    name='1.E_Loss-Exp_Delta_E2_div_Natom')
            elif self._loss_func == "quad_exp_tanh_atom":
                const = tf.constant(5.0)
                red_sum = tf.div(
                    tf.reduce_sum(tf.div(batch_delta_e2, tf.square(batch_natoms))),
                    const)
                tot_loss = tf.add(tf.reduce_sum(batch_delta_e2),
                                  tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                                  name='1.E_Loss-quad_exp_tanh_Delta_E2_div_Natom2')
            elif self._loss_func == "quad_exp_tanh_std_atom":
                const = tf.constant(5.0)
                red_sum = tf.div(tf.reduce_sum(tf.div(batch_delta_e2, batch_natoms)),
                                 const)
                tot_loss = tf.add(tf.reduce_sum(batch_delta_e2),
                                  tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                                  name='1.E_Loss-quad_exp_tanh_Delta_E2_div_Natom')
            elif self._loss_func == "quad_exp_tanh":
                const = tf.constant(5.0)
                red_sum = tf.div(tf.reduce_sum(batch_delta_e2), const)
                tot_loss = tf.add(tf.reduce_sum(batch_delta_e2),
                                  tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                                  name='1.E_Loss-quad_exp_tanh_Delta_E2')
            elif self._loss_func == "quad_atom_exp_tanh_atom":
                const = tf.constant(5.0)
                red_sum = tf.div(
                    tf.reduce_sum(
                        tf.div(batch_delta_e2, tf.square(batch_natoms))),
                    const)
                tot_loss = tf.add(
                    tf.reduce_sum(tf.div(batch_delta_e2, tf.square(batch_natoms))),
                    tf.exp(tf.multiply(const, tf.tanh(red_sum))),
                    name='1.LOSS-quad_atom_exp_tanh_Delta_E2_div_Natom2')
            elif self._loss_func == "zero":
                tot_loss = tf.constant(0.0)

            with tf.name_scope('graphs_variable'):
                mean_delta_e = tf.reduce_mean(batch_delta_e)
                mean_delta_e2 = tf.reduce_mean(batch_delta_e2)
                energy_rmse = tf.sqrt((mean_delta_e2 - mean_delta_e**2),
                                      name='Energy_STD')

                mean_delta_e_atom = tf.reduce_mean(tf.div(batch_delta_e, batch_natoms))
                mean_delta_e2_atom = tf.reduce_mean(
                    tf.div(batch_delta_e2, tf.square(batch_natoms)))
                energy_rmse_atom = tf.sqrt((mean_delta_e2_atom - mean_delta_e_atom**2),
                                           name='Energy_STD_atom')

        tf.add_to_collection('losses', tot_loss)
        tf.add_to_collection('training_stats', energy_rmse_atom)
        tf.add_to_collection('training_stats', energy_rmse)

        return tot_loss, batch_delta_e

    def tf_single_force_evaluate_dense(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, gsize, dEdg, dGdx, F_ref)
        Elements are sliced (if padded) and reshaped to compute

        .. math:

           F_k = \sum_{ij} dE/dg_{ij} dg{ij}/dx_k

        Then sum of square differences is computed so we return a number
        """
        Natoms = tf.cast(x[0], tf.int64)
        gsize = x[1]
        dGdx = tf.reshape(x[3][:Natoms**2 * gsize * 3], [Natoms * gsize, Natoms * 3])

        dEdg = tf.reshape(x[2][:Natoms, :], [1, -1])
        NNF = tf.reshape(tf.matmul(dEdg, dGdx), [-1])
        # Version with matvec:
        # dEdg = tf.reshape(x[2][:Natoms, :], [-1])
        # NNF = tf.linalg.matvec(dGdx, dEdg, transpose_a=True)

        # NNF is -force
        Fdiff = NNF + x[4][:Natoms * 3]
        Floss = tf.reduce_sum(tf.square(Fdiff))
        return Floss

    def tf_single_force_evaluate_sparse(self, x):
        """ Function operating on each element of the batch.
        x contains:
        (Natoms, gsize, dEdg,
          dGdx_size, dGdx_values, dGdx_ind1,
          dGdx_ind2, F_ref)
        Elements are sliced (if padded) and sparse matrix is reconstructed
        to compute

        F_k = \sum_{ij} dE/dg_{ij} dg{ij}/dx_k

        Then sum of square differences is computed so we return a number
        """
        Natoms = tf.cast(x[0], tf.int64)
        gsize = x[1]
        dGdxsize = x[3]
        dEdg = tf.reshape(x[2][:Natoms, :], [-1, 1])
        dGdx = tf.SparseTensor(indices=tf.cast(tf.stack(
            [x[5][:dGdxsize], x[6][:dGdxsize]], axis=1),
            dtype=tf.int64), values=x[4][:dGdxsize],
            dense_shape=[Natoms * gsize, Natoms * 3])
        NNF = tf.sparse.matmul(dGdx, dEdg, adjoint_a=True)

        # Alternative code with sparse_to_dense (seems to be slower)
        # dGdxDense = tf.sparse_to_dense(
        #             sparse_indices=tf.cast(
        #                 tf.stack([x[5][:dGdxsize], x[6][:dGdxsize]], axis=1),
        #                          dtype=tf.int64),
        #             output_shape=[Natoms*gsize, Natoms*3],
        #             sparse_values=x[4][:dGdxsize])
        # NNF = tf.matmul(dGdxDense, dEdg, transpose_a=True, a_is_sparse=True)

        # NNF is -force
        Fdiff = NNF[:, 0] + x[7][:Natoms * 3]
        Floss = tf.reduce_sum(tf.square(Fdiff))
        return Floss

    def tf_force_evaluate(self,
                          batch_size,
                          batch_dEdG,
                          batch_forces_ref,
                          batch_natoms,
                          sparse_dgvect=False,
                          batch_dgvect=None,
                          batch_dgvect_size=None,
                          batch_dgvect_values=None,
                          batch_dgvect_indices1=None,
                          batch_dgvect_indices2=None):
        g_size = self[self.atomic_sequence[0]].feature_size
        batch_gsize = g_size * tf.ones(batch_size, dtype=tf.int64)

        # Executing single force calculation for each element of the batch
        if not sparse_dgvect:
            sumFdiff = tf.map_fn(
                self.tf_single_force_evaluate_dense,
                (batch_natoms, batch_gsize, batch_dEdG, batch_dgvect, batch_forces_ref),
                dtype=tf.float32,
                parallel_iterations=batch_size,
                infer_shape=False)
        else:
            sumFdiff = tf.map_fn(
                self.tf_single_force_evaluate_sparse,
                (batch_natoms, batch_gsize, batch_dEdG, batch_dgvect_size,
                 batch_dgvect_values, batch_dgvect_indices1, batch_dgvect_indices2,
                 batch_forces_ref),
                dtype=tf.float32,
                parallel_iterations=batch_size,
                infer_shape=False)

        return sumFdiff

    def tf_force_loss(self, sumFdiff, batch_natoms):
        """this is the cost function contribution from forces

        Parameters
        ----------
            sumFdiff (sum of square difference of forces per example),
            batch_natoms,

        Returns
        -------
            the loss value for the forces term

        """

        with tf.name_scope('force_loss_calculation'):
            if self._floss_func == "quad":
                force_loss = tf.reduce_sum(sumFdiff, name='5.F_Loss_quad')
            elif self._floss_func == "exp_quad":
                force_loss = 0.5 * tf.exp(2.0 * tf.reduce_sum(sumFdiff),
                                          name='5.F_Loss_exp_quad')
            elif self._floss_func == "quad_atom_legacy":
                force_loss = tf.reduce_sum(tf.div(sumFdiff, tf.square(batch_natoms)),
                                           name='5.F_Loss_quad_atom_legacy')
            elif self._floss_func == "quad_atom":
                force_loss = tf.reduce_mean(tf.div(sumFdiff, batch_natoms),
                                            name='5.F_Loss_quad_atom')
            elif self._floss_func == "exp_quad_atom":
                force_loss = 0.5 * tf.exp(
                    2.0 * tf.reduce_sum(tf.div(sumFdiff, tf.square(batch_natoms))),
                    name='5.F_Loss_exp_quad_atom')
            elif self._floss_func == "quad_percomp":
                force_loss = tf.reduce_sum(tf.div(sumFdiff, 3.0 * batch_natoms),
                                           name='5.F_Loss')
        with tf.name_scope('graphs_variable'):
            force_rmse = tf.sqrt(tf.reduce_mean(tf.div(sumFdiff, batch_natoms)),
                                 name='Force_RMSE')
        tf.add_to_collection('losses', force_loss)
        tf.add_to_collection('training_stats', force_rmse)

        return force_loss

    def tf_network(self, batch_size, batch_dict):
        """ helper to combine all the evaluation

        Parameters
        ----------
            batch_size: number of elements in the batch
            batch_dict: containing:
                batch_of_species:
                batch_of_gvects:
                batch_size:
                batch_energy_ref:
                [opt] batch_of_dgvect: (or sparse variant)
                [opt] batch_forces_ref:
        Return
        ------

        """
        # Parse input dictionary
        batch_of_species = batch_dict['species']
        batch_of_gvects = batch_dict['gvects']
        batch_energies_ref = batch_dict['energy']
        if self._tf_require_gradients:
            batch_forces_ref = batch_dict['forces']
            if not self._sparse_derivatives:
                batch_of_dgvect = batch_dict['dgvects']
            else:
                batch_dgvect_size = batch_dict['dgvect_size']
                batch_dgvect_values = batch_dict['dgvect_values']
                batch_dgvect_indices1 = batch_dict['dgvect_indices1']
                batch_dgvect_indices2 = batch_dict['dgvect_indices2']

        if self._tf_require_gradients:
            prediction_matrix, batch_n_atoms, des_dgs = self.tf_compute_pa_energy(
                batch_of_species, batch_of_gvects, batch_size)
        else:
            prediction_matrix, batch_n_atoms = self.tf_compute_pa_energy(
                batch_of_species, batch_of_gvects, batch_size)

        prediction = tf.reduce_sum(prediction_matrix, axis=1)
        with tf.name_scope('energy_loss_calculation'):
            # reshape for convenience[100,1]->[100]
            batch_energies_ref = tf.reshape(batch_energies_ref, [-1],
                                            name='reshape_ref_en')
            batch_delta_e = prediction - batch_energies_ref

        energy_cost, _batch_deltae = self.tf_energy_loss(batch_delta_e, batch_n_atoms)

        if self._tf_require_gradients:
            if not self._sparse_derivatives:
                sumFdiff = self.tf_force_evaluate(batch_size,
                                                  des_dgs,
                                                  batch_forces_ref,
                                                  batch_n_atoms,
                                                  sparse_dgvect=False,
                                                  batch_dgvect=batch_of_dgvect)
            else:
                sumFdiff = self.tf_force_evaluate(
                    batch_size,
                    des_dgs,
                    batch_forces_ref,
                    batch_n_atoms,
                    sparse_dgvect=True,
                    batch_dgvect_size=batch_dgvect_size,
                    batch_dgvect_values=batch_dgvect_values,
                    batch_dgvect_indices1=batch_dgvect_indices1,
                    batch_dgvect_indices2=batch_dgvect_indices2)
            force_cost = self.tf_force_loss(sumFdiff, batch_n_atoms)
            energy_cost += self._forces_cost * force_cost

        return energy_cost, batch_delta_e

    @property
    def tfr_parse_function(self):
        # same trick done in tf_evaluate
        if not self._usable_for_training:
            raise ValueError('this scaffold is missing training parameters')
        g_size = self[self.atomic_sequence[0]].feature_size
        zeros = np.asarray([self._zeros[species] for species in self.atomic_sequence],
                           dtype=np.float32)

        return partial(parse_fn,
                       g_size=g_size,
                       zeros=zeros,
                       n_species=self.n_species,
                       forces=self._tf_require_gradients,
                       sparse_dgvect=self._sparse_derivatives,
                       energy_rescale=self._en_rescale)

    def _networks_file_metadata(self):
        # FIXME, I moved this code here for now, but this can be
        # done in a better way
        networks_files_name = []
        networks_tensors = []

        for species in self.atomic_sequence:
            network_files_name = []
            network_tensors = []
            network = self[species]
            for idx_l, layer in enumerate(network):
                w_size = layer.wb_shape
                b_size = layer.b_shape
                w_file_name = ('species_{}_layer_{}_weights_'
                               '{}x{}.npy'.format(
                                   species,
                                   idx_l,
                                   *w_size,
                               ))
                b_file_name = ('species_{}_layer_{}_biases_'
                               '{}.npy'.format(
                                   species,
                                   idx_l,
                                   *b_size,
                               ))
                network_files_name.append((w_file_name, b_file_name))
                network_tensors.append((layer.w_value, layer.b_value))

            networks_files_name.append(network_files_name)
            networks_tensors.append(network_tensors)
        return networks_files_name, networks_tensors

    @property
    def name(self):
        return self._name

    @property
    def atomic_sequence(self):
        return tuple(self._atomic_sequence)

    @property
    def n_species(self):
        return len(self._atomic_sequence)

    @property
    def default_network(self):
        return deepcopy(self._default_network)

    @default_network.setter
    def default_network(self, value):
        if not self._default_network:
            self._default_network = value
        else:
            raise ValueError('default network can not be changed')

    @property
    def sparse_derivative(self):
        if self._usable_for_training:
            return self._sparse_derivatives
        raise ValueError('Network not usable for training')

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
            Only networks objects can be scaffold subordinate objects.
        """
        names = []
        objects = []
        for species_symbol in self.atomic_sequence:
            names.append('')
            network = self[species_symbol]
            objects.append(network)
        return names, objects

    def dump_scaffold_lammps(self, folder, file_name, *args, **kwargs):
        logger.info(f'GVERSION {kwargs["gversion"]}')
        s = ''
        # lammps parameters string composition
        if int(kwargs['gversion']) == 1:
             #logger.info('MY GVERSION {}'.format(kwargs['gversion']))
             #s += f'!version = {_version}\n'
             s += f'!gversion = {kwargs["gversion"]}\n'
        s += '[GVECT_PARAMETERS]\n'
        s += f'Nspecies = {len(self.atomic_sequence)}\n'
        s += 'species = {}\n'.format(','.join(self.atomic_sequence))
        # Lists get written with a [ .. ]
        # which is ok, we can take care of this on lammps reader end
        # for gversion = 0 make sure that the order is as before?
        mod_data = {}
        mod_data['RsN_rad'] = kwargs['gvect_params']['RsN_rad']
        mod_data['eta_rad'] = kwargs['gvect_params']['eta_rad']
        mod_data['Rc_rad'] = kwargs['gvect_params']['Rc_rad']
        mod_data['Rs0_rad'] = kwargs['gvect_params']['Rs_rad'][0]
        if kwargs["gversion"] == 1:
            mod_data['Rs_rad'] =kwargs['gvect_params']['Rs_rad']
        elif kwargs["gversion"] == 0:
            if kwargs['gvect_params']['Rs_rad'] > 1:
                mod_data['Rsst_rad'] = kwargs['gvect_params']['Rs_rad'][1] \
                    - kwargs['gvect_params']['Rs_rad'][0]
            else:
                mod_data['Rsst_rad'] = 0.0
        #error can be raised if version not implemented

        mod_data['RsN_ang'] = kwargs['gvect_params']['RsN_ang']
        mod_data['eta_ang'] = kwargs['gvect_params']['eta_ang']
        mod_data['Rc_ang'] = kwargs['gvect_params']['Rc_ang']
        mod_data['Rs0_ang'] = kwargs['gvect_params']['Rs_ang'][0]

        if kwargs["gversion"] == 1:
            mod_data['Rs_ang'] = kwargs['gvect_params']['Rs_ang'] 
        elif kwargs["gversion"] == 0:
            if kwargs['gvect_params']['Rs_ang'] > 1:
                mod_data['Rsst_ang'] = kwargs['gvect_params']['Rs_ang'][1] \
                - kwargs['gvect_params']['Rs_ang'][0]
            else:
                mod_data['Rsst_ang'] = 0.0
        mod_data['ThetasN'] = kwargs['gvect_params']['ThetasN']
        mod_data['zeta'] = kwargs['gvect_params']['zeta']
        if kwargs["gversion"] == 1:
            mod_data['Thetas'] = kwargs['gvect_params']['Thetas']

        if 'weights' in kwargs['gvect_params']:
            mod_data['weights'] = ','.join(
                [str(x) for x in kwargs['gvect_params']['weights']])

        for key, value in mod_data.items():
            s += f'{key} = {value}\n'

        for idx_s, species in enumerate(self.atomic_sequence):
            network = self[species]
            sizes = []
            activs = []
            weights = np.array([], dtype=np.float32)

            if network.norm_layer:
                layer = network.norm_layer.wb_value
                weights = np.append(weights, layer.w_value.flatten())
                weights = np.append(weights, layer.b_value.flatten())
                sizes.append(layer.b_shape[0])
                activs.append(layer.activation)

            for idx_l, layer in enumerate(network):
                weights = np.append(weights, layer.w_value.flatten())
                if idx_l == len(network._layers) - 1:
                    biase = layer.b_value + network.offset
                    logger.info(f'network_offset={network.offset}')
                    weights = np.append(weights, biase.flatten())
                else:
                    weights = np.append(weights, layer.b_value.flatten())

                sizes.append(layer.b_shape[0])
                activs.append(layer.activation)
            atomic_number = mendeleev.element(species).atomic_number
            s += f'\n[{species}]\n'
            s += f'Nlayers = {len(sizes)}\n'
            s += 'sizes = {}\n'.format(','.join(map(str, sizes)))
            weights_file_name = f'weights_{species}.dat'
            s += f'file = {weights_file_name}\n'
            s += 'activations = {}\n'.format(','.join(map(str, activs)))
            s += f'atomic_number = {atomic_number}\n'
            binw = struct.pack('f' * len(weights), *weights)
            with open(os.path.join(folder, weights_file_name), 'wb') as weights_file:
                weights_file.write(binw)
        with open(os.path.join(folder, file_name), 'w') as f:
            f.write(s)
