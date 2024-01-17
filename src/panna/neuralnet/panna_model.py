###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import logging
import itertools
import os
import struct

from copy import deepcopy
from typing import Sequence
from typing import Union
from typing import Dict
from dataclasses import asdict
from abc import ABC

import numpy as np
import tensorflow as tf
import json

from tensorflow.python.keras.utils import losses_utils

from panna.lib.errors import NetworkNotAvailableError
from panna.neuralnet.parse_fn import ParseFn
from panna.neuralnet.force_evaluate import ForceEvaluate
from panna.neuralnet.a2affnetwork import A2affNetwork

logger = logging.getLogger(__name__)


def _get_network(network_name):
    if network_name == 'a2aff':
        return A2affNetwork
    raise ValueError(f'{network_name} not found')

@tf.function
def unpack_data(data,
                forces: bool = False,
                sparse_derivatives: bool = False,
                names: bool = False,
                input_format: str = 'tfr',
                preprocess: ABC = None,
                long_range_el: bool = False):
    """Unpack data for training and validation.

    Parameters
    ----------
      data: tensorflow data model
      compute_forces: if forces need to be unpacked
      sparse_derivatives: if the derivatives are sparse
      names: used for validation dumps
      input_format: tfr for precomputed gvectors, example otherwise
    Return
    ------
      The unpacked tuple:
      - batch_inputs,
      - (batch_energies_ref, batch_forces_ref) if forces else (batch_energies_ref, None)
      - batch_of_names if names
    """

    
    if input_format=='tfr':
        batch_of_species = data['species']
        batch_of_gvects = data['gvects']
        batch_energies_ref = data['energy']
        if long_range_el:
            el_energy_kernel = data['el_energy_kernel']
            charges = data['atomic_charges']
            total_charge = data['total_charge']
            el_ref = (charges)
            if forces:
                el_force_kernel = data['el_force_kernel']
                el_inputs = (el_energy_kernel, total_charge, el_force_kernel)
            else:
                el_inputs = (el_energy_kernel, total_charge, None)

        
        output = []
        if forces:
            batch_forces_ref = data['forces']
            if sparse_derivatives:
                batch_dg_dx_v = data['dgvect_values']
                batch_dg_dx_i1 = data['dgvect_indices1']
                batch_dg_dx_i2 = data['dgvect_indices2']
                batch_inputs = (batch_of_species, batch_of_gvects, batch_dg_dx_v,
                                batch_dg_dx_i1, batch_dg_dx_i2) 
            else:
                batch_dg_dx = data['dgvects']
                batch_inputs = (batch_of_species, batch_of_gvects, batch_dg_dx)


            refs = (batch_energies_ref, batch_forces_ref)
        else:
            batch_inputs = (batch_of_species, batch_of_gvects) 
            refs = (batch_energies_ref, None)
        output.append(batch_inputs)
        if long_range_el:
            output.append(el_inputs)
            output.append(refs)
            output.append(el_ref)
        else:
            output.append(refs)

        if names:
            batch_of_names = data['name']
            output.append(batch_of_names)
    else:
        #this is available only for ase interface.
        #TODO:
        #implement on the fly training.

        if long_range_el:
            el_energy_kernel = data['el_energy_kernel']
            charges = data['atomic_charges']
            total_charge = data['total_charge']
            el_ref = [charges]
            if forces:
                el_force_kernel = data['el_force_kernel']
                el_inputs = [el_energy_kernel, total_charge, el_force_kernel]
            else:
                el_inputs = [el_energy_kernel, total_charge, None]

        sizes = tf.shape(data['species'])
        input_stack = [data['nats'],data['species'],data['positions'],\
                       data['nn_inds'],data['nn_num'],data['nn_vecs'],\
                       data['nn_r'],data['mask1'],data['mask2']]
        if forces:
            output_signature = [tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                tf.TensorSpec(shape=[None], dtype=tf.float32)]
            gs, dgs = tf.map_fn(preprocess.tf_g, input_stack, back_prop=False,
                                fn_output_signature=output_signature)

            output = [[data['species'],gs,dgs]]
            if long_range_el:
                output.append(el_inputs)
                output.append([data['energy'],tf.reshape(data['forces'],[sizes[0],-1])])
                output.append(el_ref)
            else:
                output.append([data['energy'],tf.reshape(data['forces'],[sizes[0],-1])])

        else:
            output_signature = [tf.TensorSpec(shape=[None, None], dtype=tf.float32)]
            gs = tf.map_fn(preprocess.tf_g, input_stack, back_prop=False,
                            fn_output_signature=output_signature)
            output = [(data['species'],gs)]
            if long_range_el:
                output.append(el_inputs)
                output.append((data['energy'], None))
                output.append(el_ref)
            else:
                output.append((data['energy'], None))


        if names:
            batch_of_names = data['name']
            output.append(batch_of_names)
        
    return tuple(output)

class PannaModel(tf.keras.Model):
    """A recipe for the PANNA basic network.

    Parameters
    ----------
    config: config parser object, optional,
        A config to extract the quantities needed for the
        networks setup.
        If no config is provide, scaffold is empty.
    name: str, optional,
        A name for the network.
    """
    _scaffold_type = 'PANNA'
    _version = 'v1'

    def __init__(self,
                 g_size: int,
                 default_nn_config=None,
                 compute_forces: bool = False,
                 sparse_derivatives: bool = False,
                 examples_name: bool = False,
                 input_format: str = 'tfr',
                 val_input_format: str = 'tfr',
                 preprocess: ABC = None,
                 name: str = 'PANNA_model',
                 max_atoms: int = -1,
                 metrics: Sequence[str] = ['MAE'],
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._default_network_config = default_nn_config

        # internal values
        self._networks = {}
        self._atomic_sequence = []
        self._force_computer = None

        self._examples_name = examples_name
        self._input_format = input_format
        self._val_input_format = val_input_format
        self._preprocess = preprocess
        self._compute_forces = compute_forces
        self._sparse_derivatives = sparse_derivatives
        self.g_size = g_size
        self._max_atoms = max_atoms
        self._printmetrics = metrics

    @property
    def atomic_sequence(self):
        return tuple(self._atomic_sequence)

    @property
    def n_species(self):
        return len(self._atomic_sequence)

    @property
    def default_network_config(self):
        return deepcopy(self._default_network_config)

    @default_network_config.setter
    def default_network_config(self, value):
        if self._default_network_config is None:
            self._default_network_config = value
        else:
            raise ValueError('default network can not be changed')

    @property
    def force_computer(self):
        if self._force_computer is None:
            if self._sparse_derivatives:
                # Allow dense here if selected from input?
                opv = 'sparse'
            else:
                opv = 'dense'
            self._force_computer = ForceEvaluate(
                g_size=self.g_size,
                dg_dx_format_sparse=self._sparse_derivatives,
                op_version=opv)
            # op_version='dense')
            return self._force_computer
        return self._force_computer

    def __getitem__(self, value):
        """ Recover a network for a given species.

        Returns
        -------
        Network object
            - The requested network if already present.
            - a default network,
              If the requested network is not present,
              In this case the atomic sequence gets also updated

        Raises
        ------
        NetworkNotAvailableError
            If the requested network is not available and there
            is no default network
        """
        try:
            tmp_network = self._networks[value]
            return tmp_network
        except KeyError as _:
            if self._default_network_config:
                tmp_network = deepcopy(self._default_network_config[1])
                tmp_network.name = value
                tmp_network.compute_jacobian = self._compute_forces
                Network = _get_network(self._default_network_config[0])
                tmp_network = Network(**asdict(tmp_network))
                self._networks[value] = tmp_network
                if value not in self._atomic_sequence:
                    self._atomic_sequence.append(value)
                return self._networks[value]
        raise NetworkNotAvailableError('default network not available')

    def __setitem__(self, index: str, value: tf.keras.layers.Layer):
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
        Value: tf.keras.layers.Layer
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
        
    @tf.function
    def _compute_pa_energy(self, batch_of_species, 
                           batch_of_gvects, 
                           batch_n_atoms):

        """Perform the evaluation of a batch predicting the energies."""

        # (Do a special case for only 1 species?)
        # for each atom, the example number in the batch,
        # to keep track of where they came from, species sorted
        batch_size = tf.shape(batch_of_species)[0]
        max_number_atoms = tf.shape(batch_of_species)[1]
        partitions = tf.cast(tf.reshape(batch_of_species, [-1]), tf.int32)
        array_gvects = tf.reshape(batch_of_gvects, [-1, self.g_size])
        atomic_index_partitions = tf.dynamic_partition(
            tf.range(batch_size * max_number_atoms), partitions, self.n_species + 1)

        # List of derivatives wrt Gs, needed for differentiation
        atomic_energies_partitions = []
        if self._compute_forces:
            des_dgs_partitions = []

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            logger.debug('creating network: %d, %s', species_idx, species_symbol)
            # recover the network
            network = self[species_symbol]
            # get the appropriate gvectors
            species_gvects = tf.gather(array_gvects, atomic_index_partitions[species_idx])

            e_empty_list = tf.zeros(0)
            gvect_size = tf.size(species_gvects)
            if self._compute_forces:

                dg_empty_list = tf.zeros((0,self.g_size))
                s_atomic_energies, s_des_dgs = tf.cond(gvect_size>0,
                                                   lambda: network(species_gvects),
                                                   lambda: (e_empty_list, dg_empty_list))

                # s_des_dgs shape = (None, 1, g_size)
                # None: n of atoms for the given species in the current batch
                # reshape [-1, g_size] to kill the extra [1] dimension
                s_des_dgs = tf.reshape(s_des_dgs, [-1, self.g_size])
                des_dgs_partitions.append(s_des_dgs)
            else:
                s_atomic_energies = tf.cond(gvect_size>0,
                                                   lambda: network(species_gvects),
                                                   lambda: e_empty_list)

            s_atomic_energies = tf.reshape(s_atomic_energies, [-1])
            atomic_energies_partitions.append(s_atomic_energies)

        # recover how many empty slot we have in the species matrix
        # and fill the predicted energy for those with zeros
        n_placeholder_in_species_matrix = tf.shape(atomic_index_partitions[self.n_species])[0]
        fake_species_e_contrib = tf.zeros(n_placeholder_in_species_matrix)
        atomic_energies_partitions.append(fake_species_e_contrib)        
        batch_of_energies = tf.reshape(tf.dynamic_stitch(atomic_index_partitions,
                                                         atomic_energies_partitions),
                                       [batch_size, max_number_atoms])

        if self._compute_forces:            
            des_dgs_partitions.append(tf.zeros([n_placeholder_in_species_matrix, self.g_size]))
            batch_of_des_dgs = tf.reshape(tf.dynamic_stitch(atomic_index_partitions, 
                                                            des_dgs_partitions),
                                          [batch_size, max_number_atoms, self.g_size])
            return batch_of_energies, batch_of_des_dgs
        return batch_of_energies

    def call(self, inputs, training=None):
        """ Model call
        Parameters
        ----------
        Return
        ------
        """

        if not training:
            # inputs can be (a list, a tuple) or a dict of tensors
            # now rewritten to avoid ragged tensors
            def _input_to_tensor(x, dtype=None):
                # Can this be done more efficiently without ragged?
                lens = [len(row) for row in x]
                maxl = tf.reduce_max(lens)
                tens = [tf.concat([row,tf.zeros(maxl-len(row))], axis=0) for row in x]
                tens = tf.stack(tens)
                if dtype:
                    tens = tf.cast(tens, dtype=dtype)
                return tens

            if isinstance(inputs, Sequence):
                # someone feed a sequence by hand
                if isinstance(inputs[0], Sequence):
                    batch_of_species = _input_to_tensor(inputs[0], dtype=tf.int32)
                else:
                    batch_of_species = inputs[0]

                if isinstance(inputs[1], Sequence):
                    batch_of_gvects = _input_to_tensor(inputs[1])
                else:
                    batch_of_gvects = inputs[1]
            elif isinstance(inputs, Dict):
                # This is the most common call to create a new model
                data = unpack_data(inputs, self._compute_forces,
                                   self._sparse_derivatives, self._examples_name,
                                   input_format=self._input_format, preprocess=self._preprocess)
                inputs = data[0]
                batch_of_species = inputs[0]
                batch_of_gvects = inputs[1]
            else:
                raise ValueError('Unclear input format')
        else:
            # We already called unpack in the train_step
            batch_of_species = inputs[0]
            batch_of_gvects = inputs[1]

        atoms_presence = tf.where(batch_of_species<self.n_species,1,0)
        batch_n_atoms = tf.reduce_sum(atoms_presence, axis=1, keepdims=True)

        if self._compute_forces:
            batch_of_energies, batch_of_des_dgs = self._compute_pa_energy(
                batch_of_species, batch_of_gvects, batch_n_atoms)
            forces_pred = self.force_computer(batch_of_des_dgs,
                                              batch_n_atoms,
                                              inputs[2:],
                                              training=training)
        else:
            batch_of_energies = self._compute_pa_energy(batch_of_species,
                                                        batch_of_gvects, batch_n_atoms)
        energies_pred = tf.math.reduce_sum(batch_of_energies, axis=1)

        if self._compute_forces:
            return batch_n_atoms, energies_pred, forces_pred
        return batch_n_atoms, energies_pred

    def compile(self,
                e_loss,
                f_loss=None,
                f_cost=0.0,
                energy_example_weight=None,
                force_example_weight=None,
                optimizer='adam', **kwargs):
        super().compile(optimizer=optimizer)
        self._e_loss = e_loss
        self._energy_example_weight = energy_example_weight
        self._force_example_weight = force_example_weight
        self._f_loss = f_loss
        self._f_cost = f_cost
        
    def _model_inherited_losses(self):
        regularization_losses = losses_utils.cast_losses_to_common_dtype(self.losses)
        reg_loss = tf.add_n(regularization_losses)
        # take also care of distribution mechanism:
        # otherwise each replica add its own contribution
        return losses_utils.scale_loss_for_distribution(reg_loss)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = unpack_data(data, self._compute_forces, self._sparse_derivatives, \
                           input_format=self._input_format, preprocess=self._preprocess)
        batch_inputs = data[0]
        batch_energies_ref, batch_forces_ref = data[1]

        with tf.GradientTape() as tape:
            if self._compute_forces:
                batch_n_atoms, energies_pred, forces_pred = self(batch_inputs, training=True)
            else:
                batch_n_atoms, energies_pred = self(batch_inputs, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            if self._energy_example_weight is not None:
                e_weight = self._energy_example_weight(batch_n_atoms)
            else:
                e_weight = None
            e_loss = self._e_loss(batch_energies_ref,
                                  energies_pred,
                                  sample_weight=e_weight,
                                  n_atoms=batch_n_atoms)
            if self._f_loss:
                if self._force_example_weight is not None:
                    f_weight = self._force_example_weight(batch_n_atoms)
                else:
                    f_weight = None
                f_loss = self._f_loss(batch_forces_ref,
                                      forces_pred,
                                      sample_weight=f_weight)
            else:
                f_loss = 0
            reg_loss = self._model_inherited_losses()
            loss = e_loss + self._f_cost * f_loss + reg_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Adding scalars for tensorboard, _train_counter gives us the step
        # Output quantitites are added once per epoch, these follow the TB callback
        tf.summary.scalar('1. Losses/1. Total',loss,self._train_counter)
        tf.summary.scalar('1. Losses/2. Energy',e_loss,self._train_counter)
        if self._f_loss:
            tf.summary.scalar('1. Losses/3. Forces',f_loss,self._train_counter)
        if self._default_network_config[1].kernel_regularizer != None:
            tf.summary.scalar('1. Losses/4. Regularization',reg_loss,self._train_counter)
        rmseat = tf.sqrt(tf.keras.losses.MeanSquaredError()(batch_energies_ref,energies_pred, \
                                         sample_weight=1/(batch_n_atoms*batch_n_atoms)))
        tf.summary.scalar('2. Metrics/1. RMSE/atom',rmseat,self._train_counter)
        maeat = tf.keras.losses.MeanAbsoluteError()(batch_energies_ref,energies_pred,sample_weight=1/batch_n_atoms)
        tf.summary.scalar('2. Metrics/2. MAE/atom',maeat,self._train_counter)
        if self._f_loss:
            NF = 3.0*tf.cast(tf.reduce_sum(batch_n_atoms),float)
            Frmse = tf.sqrt(tf.reduce_sum((batch_forces_ref-forces_pred)**2)/NF)
            tf.summary.scalar('2. Metrics/3. Force components RMSE',Frmse,self._train_counter)
            Fmae = tf.reduce_sum(tf.abs(batch_forces_ref-forces_pred))/NF
            tf.summary.scalar('2. Metrics/4. Force components MAE',Fmae,self._train_counter)

        metrics = {'tot_st': self._train_counter}
        if 'MAE' in self._printmetrics:
            metrics.update({'MAE/at': maeat})
            if self._f_loss:
                metrics.update({'F_MAE': Fmae})
        if 'RMSE' in self._printmetrics:
            metrics.update({'RMSE/at': rmseat})
            if self._f_loss:
                metrics.update({'F_RMSE': Frmse})
        if 'loss' in self._printmetrics:
            metrics.update({'loss': loss, 'e_loss': e_loss})
            if self._default_network_config[1].kernel_regularizer != None:
                metrics.update({'reg_loss': reg_loss})
            if self._f_loss:
                metrics.update({'f_loss': f_loss})

        # return {'tot_st': self._train_counter, 'loss': loss, 'e_loss': e_loss, 'reg_loss': reg_loss, 'f_loss': f_loss}
        return metrics

    def test_step(self, data):
        data = unpack_data(data, self._compute_forces, self._sparse_derivatives,
                           self._examples_name, input_format=self._val_input_format, 
                           preprocess=self._preprocess)
        batch_inputs = data[0]
        batch_energies_ref, batch_forces_ref = data[1]
        prediction = {}
        if self._examples_name:
            prediction['names'] = data[2]
        if self._compute_forces:
            batch_n_atoms, energies_pred, forces_pred = self(batch_inputs, training=True)
            prediction['forces'] = forces_pred, batch_forces_ref
        else:
            batch_n_atoms, energies_pred = self(batch_inputs, training=True)
        prediction['energies'] = energies_pred, batch_energies_ref

        if self._energy_example_weight is not None:
            e_weight = self._energy_example_weight(batch_n_atoms)
        else:
            e_weight = None
        e_loss = self._e_loss(batch_energies_ref, energies_pred, e_weight, n_atoms=batch_n_atoms)
        if self._f_loss:
            if self._force_example_weight is not None:
                f_weight = self._force_example_weight(batch_n_atoms)
            else:
                f_weight = None
            f_loss = self._f_loss(batch_forces_ref, forces_pred, sample_weight=f_weight)
        else:
            f_loss = 0
        rmseat = tf.sqrt(tf.keras.losses.MeanSquaredError()(batch_energies_ref,energies_pred, \
                                            sample_weight=1/(batch_n_atoms*batch_n_atoms)))
        # Adding scalars for tensorboard, _train_counter gives us the step
        tf.summary.scalar('2. Metrics/1. RMSE/atom',rmseat,self._train_counter)
        maeat = tf.keras.losses.MeanAbsoluteError()(batch_energies_ref,energies_pred,sample_weight=1/batch_n_atoms)
        tf.summary.scalar('2. Metrics/2. MAE/atom',maeat,self._train_counter)
        if self._f_loss:
            NF = 3.0*tf.cast(tf.reduce_sum(batch_n_atoms),float)
            Frmse = tf.sqrt(tf.reduce_sum((batch_forces_ref-forces_pred)**2)/NF)
            tf.summary.scalar('2. Metrics/3. Force components RMSE',Frmse,self._train_counter)
            Fmae = tf.reduce_sum(tf.abs(batch_forces_ref-forces_pred))/NF
            tf.summary.scalar('2. Metrics/4. Force components MAE',Fmae,self._train_counter)
        
        metrics = {}
        if 'MAE' in self._printmetrics:
            metrics.update({'MAE/at': maeat})
            if self._f_loss:
                metrics.update({'F_MAE': Fmae})
        if 'RMSE' in self._printmetrics:
            metrics.update({'RMSE/at': rmseat})
            if self._f_loss:
                metrics.update({'F_RMSE': Frmse})
        if 'loss' in self._printmetrics:
            metrics.update({'e_loss': e_loss})
            if self._f_loss:
                metrics.update({'f_loss': f_loss})

        # return {'e_loss': e_loss, 'f_loss': f_loss}
        return metrics

    def predict_step(self, data):
        """Logic for one inference step.

        Parameters
        ----------
          data: A nested structure of `Tensor`s.

        Returns
        -------
          The result of one inference step, typically the output of calling the
          `Model` on data.
        """
        data = unpack_data(data, self._compute_forces, self._sparse_derivatives,
                           self._examples_name, input_format=self._input_format, 
                           preprocess=self._preprocess)
        batch_inputs = data[0]
        batch_energies_ref, batch_forces_ref = data[1]
        # Padding only for older versions of TF2 where this causes problems
        if self._max_atoms > 0:
            sh = tf.shape(batch_inputs[0])
            batch_forces_ref = tf.concat([batch_forces_ref,
                                  tf.zeros( (sh[0],3*(self._max_atoms-sh[1])) )], 1)


        if self._compute_forces:
            batch_n_atoms, energies_pred, forces_pred = self(batch_inputs, training=False)
            # Padding only for older versions of TF2 where this causes problems
            if self._max_atoms > 0:
                forces_pred = tf.concat([forces_pred,
                                 tf.zeros( (sh[0],3*(self._max_atoms-sh[1])) )], 1)
            output = [energies_pred, forces_pred]
        else:
            batch_n_atoms, output = self(batch_inputs, training=False)

        if self._examples_name:
            batch_names = data[2]
        else:
            batch_names = tf.constant([b'N.A.'])
        if batch_forces_ref is None:
            # all the output must be tensors otherwise
            # multithreading fail to stack the outputs
            batch_forces_ref = tf.constant([0])
        return output, tf.reshape(batch_n_atoms, [-1]), \
            (batch_energies_ref, batch_forces_ref), batch_names

    @property
    def tfr_parse_function(self):
        g_size = self[self.atomic_sequence[0]].feature_size
        return ParseFn(g_size=g_size, n_species=self.n_species)


    def dump_network_lammps(self, folder, file_name, **kwargs):        
        logger.info(f'GVERSION {kwargs["gversion"]}')
        s = ''
        # lammps parameters string composition
        if int(kwargs['gversion']) == 1:
             s += f'!gversion = {kwargs["gversion"]}\n'
        s += '[GVECT_PARAMETERS]\n'
        s += f'Nspecies = {len(self.atomic_sequence)}\n'
        s += 'species = {}\n'.format(','.join(self.atomic_sequence))
        mod_data = {}
        if kwargs["gversion"] == 1:
            mod_data['RsN_rad'] = kwargs['gvect_params']['RsN_rad']
        mod_data['eta_rad'] = kwargs['gvect_params']['eta_rad']
        mod_data['Rc_rad'] = kwargs['gvect_params']['Rc_rad']
        if kwargs["gversion"] == 1:
            mod_data['Rs_rad'] =kwargs['gvect_params']['Rs_rad']
        elif kwargs["gversion"] == 0:
            mod_data['Rs0_rad'] = kwargs['gvect_params']['Rs0_rad']
            mod_data['Rsst_rad'] = kwargs['gvect_params']['Rsst_rad']
            mod_data['RsN_rad'] = kwargs['gvect_params']['RsN_rad']
        #error can be raised if version not implemented

        if kwargs["gversion"] == 1:
            mod_data['RsN_ang'] = kwargs['gvect_params']['RsN_ang']
        mod_data['eta_ang'] = kwargs['gvect_params']['eta_ang']
        mod_data['Rc_ang'] = kwargs['gvect_params']['Rc_ang']
        if kwargs["gversion"] == 1:
            mod_data['Rs_ang'] = kwargs['gvect_params']['Rs_ang']
        elif kwargs["gversion"] == 0:
            mod_data['Rs0_ang'] = kwargs['gvect_params']['Rs0_ang']
            mod_data['Rsst_ang'] = kwargs['gvect_params']['Rsst_ang']
            mod_data['RsN_ang'] = kwargs['gvect_params']['RsN_ang']
            mod_data['zeta'] = kwargs['gvect_params']['zeta']
        mod_data['ThetasN'] = kwargs['gvect_params']['ThetasN']
        if kwargs["gversion"] == 1:
            mod_data['zeta'] = kwargs['gvect_params']['zeta']
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

            for idx_l, (wb_l, act) in enumerate(zip(network.wbs_tensors,\
                                                  network.layers_activation)):
                weights = np.append(weights, wb_l[0].flatten())
                # Adding offset to the last layer
                if idx_l == len(network._layers) - 1:
                    wb_l[1][0] += network.offset
                    logger.info(f'{species} network_offset={network.offset}')
                weights = np.append(weights, wb_l[1].flatten())
                sizes.append(wb_l[1].shape[0])
                activs.append(act)

            binw = struct.pack('f' * len(weights), *weights)
            weights_file_name = f'weights_{species}.dat'
            with open(os.path.join(folder, weights_file_name), 'wb') as weights_file:
                weights_file.write(binw)
            s += f'\n[{species}]\n'
            s += f'Nlayers = {len(sizes)}\n'
            s += 'sizes = {}\n'.format(','.join(map(str, sizes)))
            s += f'file = {weights_file_name}\n'
            s += 'activations = {}\n'.format(','.join(map(str, activs)))
                
        with open(os.path.join(folder, file_name), 'w') as f:
            f.write(s)

    def dump_network_panna(self, folder, file_name, **kwargs):
        # metadata = extra_data if extra_data else {}
        metadata = {}
        metadata['version'] = self._version
        metadata['scaffold_type'] = self._scaffold_type
        metadata['networks_species'] = self.atomic_sequence
        metadata['networks'] = [self[x].to_json()
                                 for x in self.atomic_sequence]
        with open(os.path.join(folder, 'networks_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        for species in self.atomic_sequence:
            for idx_l, wb_l in enumerate(self[species].wbs_tensors):
                name = species+'_l'+str(idx_l)
                np.save(os.path.join(folder, name+'_w.npy'), wb_l[0])
                np.save(os.path.join(folder, name+'_b.npy'), wb_l[1])


def create_panna_model(model_params, validation_params = None) -> PannaModel:
    """Create a new PANNA model with the proper parameters."""
    if validation_params:
        val_input_format = validation_params.input_format
    else:
        val_input_format = model_params.input_format
    panna_model = PannaModel(model_params.g_size, model_params.default_nn_config,
                             model_params.compute_forces,
                             model_params.sparse_derivatives,
                             model_params.examples_name,
                             model_params.input_format,
                             val_input_format,
                             model_params.preprocess,
                             max_atoms=model_params.max_atoms,
                             metrics=model_params.metrics)
    for network_name, config in model_params.networks_config:
        Network = _get_network(network_name)
        if not config.is_ready:
            raise ValueError(f'{config.name} not ready')
        network = Network(**asdict(config))
        panna_model[config.name] = network
    return panna_model
