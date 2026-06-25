import logging
import itertools
import os
import struct

from mendeleev import element

from copy import deepcopy
from typing import Sequence
from typing import Union
from typing import Dict
from dataclasses import asdict
from abc import ABC
import sys
import numpy as np
import tensorflow as tf
import json

from tensorflow.python.keras.utils import losses_utils

from panna.lib.errors import NetworkNotAvailableError
from panna.neuralnet.parse_fn import ParseFn
from panna.neuralnet.force_evaluate import ForceEvaluate
from panna.neuralnet.a2affnetwork import A2affNetwork
from panna.neuralnet.panna_model import PannaModel 
from panna.neuralnet.panna_model import unpack_data 
from panna.neuralnet.panna_model import _get_network 
from panna.neuralnet.force_evaluate import ForceEvaluate
logger = logging.getLogger(__name__)

class PannaModel_with_electrostatics(PannaModel):
    """
    
    """
    _scaffold_type='PANNA_with_electrostatics_model'

    def __init__(self, 
                 g_size: int,
                 default_nn_config=None,
                 compute_forces: bool = False,
                 sparse_derivatives: bool = False,
                 examples_name: bool = False,
                 input_format: str = 'tfr',
                 val_input_format: str = 'tfr',
                 preprocess: ABC = None,
                 name: str = 'PANNA_el',
                 max_atoms: int = -1,
                 metrics: Sequence[str] = ['MAE'],
                 constrain_even_terms: bool = False,
                 min_eigenvalue: bool = 0.0,
                 long_range_el: bool = True):

        super().__init__(g_size, default_nn_config,
                 compute_forces,
                 sparse_derivatives,
                 examples_name,
                 input_format,
                 val_input_format,
                 preprocess,
                 name,
                 max_atoms,
                 metrics)


    
        self._constrain_even_terms = False
        self._min_eigenvalue = 0.0


        

    @property
    def atomic_hardness(self):
        return np.asarray(self._species_hardness)
    @atomic_hardness.setter
    def atomic_hardness(self, value):
        self._atomic_hardness = value
    @property
    def electronegativity(self):
        return np.asarray(self._species_electronegativity)
    @electronegativity.setter
    def electronegativity(self, value):
        self._electronegativity = value
    @property
    def min_eigenvalue(self):
        return self._min_eigenvalue
    @min_eigenvalue.setter
    def min_eigenvalue(self, value):
        self._min_eigenvalue = value
    
    #Define a new function
    @tf.function
    def tf_compute_pa_energy(self, batch_of_species, 
                           batch_of_gvects, 
                           batch_n_atoms,
                           constrain_even_terms=False,
                           min_eigenvalue=0.0):

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

        #taylor series expansion is up to second order (quadratic model): atomic NN returns 3 numbers
        SR_order = 2 # may need to be provided by user in the future.

        # List of atomic energies and derivatives wrt Gs, needed for force evaluation
        atomic_energies_partitions = [[] for k in range(SR_order+1)]
        if self._compute_forces:
            des_dgs_partitions = [[] for k in range(SR_order+1)]

        for species_idx, species_symbol in enumerate(self.atomic_sequence):
            logger.debug('creating network: %d, %s', species_idx, species_symbol)
            # recover the network
            network = self[species_symbol]
            # get the appropriate gvectors
            species_gvects = tf.gather(array_gvects, atomic_index_partitions[species_idx])
            
            e_empty_list = [tf.zeros(0) for i in range(3)]
            gvect_size = tf.size(species_gvects)
            if self._compute_forces:

                dg_empty_list = [tf.zeros((0,self.g_size)) for i in range(3)]

                s_atomic_energies, s_des_dgs = tf.cond(gvect_size>0,
                                                   lambda: network(species_gvects),
                                                   lambda: (e_empty_list, dg_empty_list))

                # s_des_dgs shape = (3, None, 1, g_size)
                # None: n of atoms for the given species in the current batch
                # reshape each to [-1, g_size] to kill the extra [1] dimension
                
                for k in range(SR_order+1):
                    s_des_dgs_k = tf.reshape(s_des_dgs[k], [-1, self.g_size])
                    atomic_energies_partitions[k].append(tf.reshape(s_atomic_energies[k], [-1]))
                    des_dgs_partitions[k].append(s_des_dgs_k)
            else:
                s_atomic_energies = tf.cond(gvect_size>0,
                                                   lambda: network(species_gvects),
                                                   lambda: e_empty_list)
                for k in range(SR_order+1):
                    atomic_energies_partitions[k].append(tf.reshape(s_atomic_energies[k], [-1]))

        # recover how many empty slot we have in the species matrix
        # and fill the predicted energy for those with zeros
        n_placeholder_in_species_matrix = tf.shape(atomic_index_partitions[self.n_species])[0]
        fake_species_e_contrib = tf.zeros(n_placeholder_in_species_matrix)

        batch_energies = [[] for k in range(SR_order+1)]
        for k in range(SR_order+1):
            atomic_energies_partitions[k].append(fake_species_e_contrib)
            batch_energies[k] = tf.reshape(tf.dynamic_stitch(atomic_index_partitions,
                                                         atomic_energies_partitions[k]),
                                       [batch_size, max_number_atoms])

        if self._compute_forces:
            batch_of_des_dgs = [[] for k in range(SR_order+1)]
            for k in range(SR_order+1):
                  des_dgs_partitions[k].append(tf.zeros([n_placeholder_in_species_matrix, self.g_size]))
                  batch_of_des_dgs[k] = tf.reshape(tf.dynamic_stitch(atomic_index_partitions,
                                                            des_dgs_partitions[k]),
                                          [batch_size, max_number_atoms, self.g_size])
            return batch_energies, batch_of_des_dgs
        return batch_energies

    def tf_compute_A_matrix_from_V(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, V, atomic_hardness)
        Elements are sliced (if padded) and reshaped to compute

        """

        Natoms = tf.cast(x[0], tf.int32)
        V = tf.identity(x[1][:Natoms**2])
        V = tf.reshape(V, [Natoms, Natoms])
        atomic_hardness = x[2][:Natoms]
        tmp = tf.zeros((Natoms, Natoms))
        tmp = tf.linalg.set_diag(tmp, atomic_hardness)
        V += tmp
        paddings = [[0, 1], [0, 1]]
        A_mat = tf.pad(V, paddings, 'CONSTANT', constant_values=1.0)
        A_mat = tf.reshape(A_mat, [-1])
        
      
        A_dim = Natoms+1
        A_mat_tmp = tf.identity(A_mat[:A_dim**2-1])
        paddings = [[0, 1]]
        A_mat = tf.pad(A_mat_tmp, paddings, 'CONSTANT', constant_values=0.0)
        #tmp = tf.range(A_dim**2)
        #A_mat = tf.where(tf.less(tmp, A_dim**2-1), A_mat, tf.zeros(A_dim**2))

        return tf.reshape(A_mat, [A_dim, A_dim])

    def tf_compute_charges(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, n_max_diff, V, atomic_hardness, chi,
                     total_charge, des_dqs, _des2_dqs2)
        Elements are sliced (if padded) and reshaped to compute
        
        """
        #compute A matrix from V matrix
        Natoms = tf.cast(x[0][0], tf.int32)
        n_max_diff = tf.cast(x[1][0], tf.int32)
        V = tf.reshape(x[2][:Natoms**2], [Natoms, Natoms])
        atomic_hardness = x[3][:Natoms]

        _chi = x[4][:Natoms]
        tot_charge = x[5]
        _des_dqs = x[6][:Natoms]
        _des2_dqs2 = x[7][:Natoms]

        #compute matrix A

        hessian = _des2_dqs2 + atomic_hardness
        x1 = (Natoms, V, hessian)
        A_mat = self.tf_compute_A_matrix_from_V(x1)
        b_mat = -_chi - _des_dqs
        Adiag = tf.linalg.diag_part(A_mat)
        precond = tf.linalg.diag(Adiag)

        paddings = tf.constant([[0, 1]])
        
        b_mat = tf.pad(b_mat, paddings, 'CONSTANT', constant_values = tot_charge)
        charge = tf.linalg.solve(A_mat, b_mat[:,tf.newaxis])
        charge = tf.reshape(charge, [-1])
        paddings = [[0, n_max_diff]]
        #pad charge to constant shape of the maximum number of atoms
        charge = tf.pad(charge, paddings, 'CONSTANT')
        return charge

    def batch_charges(self, batch_size, max_n_atoms,
                     batch_n_atoms,
                     max_n_atoms_diff,
                     V, chi,
                     total_charge, atomic_hardness,
                     des_dqs, des2_dqs2):
                     
        x = (batch_n_atoms, max_n_atoms_diff, V, atomic_hardness, chi, total_charge, des_dqs, des2_dqs2)
        atomic_charges = tf.map_fn(self.tf_compute_charges, x, dtype=tf.float32)
        atomic_charges = tf.reshape(atomic_charges, [batch_size, max_n_atoms+1])

        return  atomic_charges

    def estimate_atomic_quantities(self, batch_of_species):
        IE = np.asarray([element(sym).ionenergies[1] \
                for sym in self.atomic_sequence])
        EA = np.asarray([0.0 if not element(sym).electron_affinity else \
                element(sym).electron_affinity for sym in self.atomic_sequence])

        species_hardness = IE - EA
        species_electronegativity = 0.5 * (IE + EA)
        species_hardness = tf.constant(species_hardness, dtype=tf.float32)
        species_electronegativity = tf.constant(species_electronegativity, dtype=tf.float32)
        shape_spec_idx = tf.shape(batch_of_species)
        max_n_atoms = shape_spec_idx[1]
        batch_size = shape_spec_idx[0]


        #atomic contribution
        atomic_hardness = tf.zeros((batch_size, max_n_atoms), dtype=tf.float32)
        electronegativity = tf.zeros((batch_size, max_n_atoms), dtype=tf.float32)

        for idx in range(len(self.atomic_sequence)):

            species_location = tf.where(tf.equal(batch_of_species, idx), 1., 0)

            tmp = tf.cast(tf.fill(shape_spec_idx, species_hardness[idx]), dtype=tf.float32)
            atomic_hardness += species_location * tmp

            tmp = tf.cast(tf.fill(shape_spec_idx, species_electronegativity[idx]), dtype=tf.float32)
            electronegativity += species_location * tmp

        return electronegativity, atomic_hardness

    def tf_compute_2body_energy(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, charge, V)
        Elements are sliced (if padded) and reshaped to compute

        """

        Natoms = tf.cast(x[0][0], tf.int32)
        atomic_charges = x[1][:Natoms]
        V = tf.reshape(x[2][:Natoms**2], [Natoms, Natoms])

        q_outer = atomic_charges[:,tf.newaxis] * atomic_charges[tf.newaxis,:]
        ## 2body contribution
        E_tot = 0.5*tf.reduce_sum(q_outer * V)
        return E_tot

    def tf_compute_2body_forces(self, x):
        """ Function operating on each element of the batch.
        x contains: (Natoms, n_max_diff, V_prime, charge)
        Elements are sliced (if padded) and reshaped to compute

        """

        Natoms = tf.cast(x[0][0], tf.int32)
        n_max_diff = tf.cast(x[1][0], tf.int32)
        V_prime = tf.reshape(x[2][:Natoms**2*3], [Natoms, Natoms, 3])
        atomic_charges = x[3][:Natoms]

        q_outer = atomic_charges[:,tf.newaxis] * atomic_charges[tf.newaxis,:]
        f_tot_term = tf.reduce_sum(tf.tile(q_outer[:,:, tf.newaxis], [1,1,3])*V_prime, axis=1)
        f_tot_term = tf.reshape(f_tot_term, [-1])

        paddings = [[0, n_max_diff*3]]

        return tf.reshape(
                tf.pad(f_tot_term, paddings, 'CONSTANT'),
                [Natoms+n_max_diff, 3])
     
    
    @tf.function
    def tf_electrostatics(self, batch_n_atoms, batch_of_species, 
                          batch_of_gvects, V, total_charge,
                          electronegativity, atomic_hardness, V_prime=None): 
        '''
        This routine computes the contribution of electrostatic energies
        and forces.
        '''
        shape_spec_idx = tf.shape(batch_of_species)
        batch_size = shape_spec_idx[0]
        max_n_atoms = shape_spec_idx[1]

        max_n_atoms_extended = tf.tile([max_n_atoms], [batch_size])
        max_n_atoms_diff = tf.cast(max_n_atoms_extended, dtype=tf.int32) - batch_n_atoms

        atoms_location = tf.where(tf.less(batch_of_species, self.n_species),
                                  tf.ones(tf.shape(batch_of_species)),
                                  tf.zeros(tf.shape(batch_of_species)))

        #compute E^0, E^1 and E^2
        if self._compute_forces:
            _E, dE = self.tf_compute_pa_energy(
                             batch_of_species, batch_of_gvects,
                             batch_n_atoms,
                             constrain_even_terms=self._constrain_even_terms,
                             min_eigenvalue=self._min_eigenvalue) 
        else:

            _E = self.tf_compute_pa_energy(
                             batch_of_species, batch_of_gvects,
                             batch_n_atoms,
                             constrain_even_terms=self._constrain_even_terms,
                             min_eigenvalue=self._min_eigenvalue) 

        E = _E[0]
        Ep = _E[1]
        Epp = _E[2]

        atomic_charges = self.batch_charges(
                                           batch_size,
                                           max_n_atoms,
                                           batch_n_atoms,
                                           max_n_atoms_diff,
                                           V, electronegativity, total_charge,
                                           atomic_hardness, Ep, Epp)

        # Remove the Langrange multiplier
        _atomic_charges = atomic_charges[:, :max_n_atoms] * atoms_location

        atomic_term = tf.zeros([batch_size, max_n_atoms], dtype=tf.float32)

        #compute atomic contribution
        atomic_term += (electronegativity + 0.5 * atomic_hardness * _atomic_charges) * _atomic_charges

        E_tot = tf.reduce_sum(atomic_term, axis=-1)
        
        E_2body =  tf.map_fn(self.tf_compute_2body_energy,
                            (batch_n_atoms, _atomic_charges, V),
                            dtype=tf.float32)
        E_tot += E_2body

        if self._compute_forces:
            f_tot_term =  tf.map_fn(self.tf_compute_2body_forces,
                            (batch_n_atoms, max_n_atoms_diff, V_prime, _atomic_charges),
                            dtype=tf.float32)

            return [E_tot, f_tot_term, _atomic_charges, _E, dE]

        # compute 2 body contribution
        #V = tf.reshape(V, [batch_size, max_n_atoms, max_n_atoms])
        #q_outer = _atomic_charges[:,:,tf.newaxis] * _atomic_charges[:,tf.newaxis,:]
        #E_2body = 0.5*tf.reduce_sum(V * q_outer, axis = (1,2))

        #E_tot += E_2body

        #if self._compute_forces:
        #    V_prime = tf.reshape(V_prime, [batch_size, max_n_atoms, max_n_atoms, 3])
        #    f_tot_term = tf.reduce_sum(q_outer[:,:,:,tf.newaxis] * V_prime, axis=2)

       #     return [E_tot, f_tot_term, _atomic_charges, _E, dE]

        return [E_tot, _atomic_charges, _E]


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
            #    # someone feed a sequence by hand
                net_inputs = inputs[0]
                if isinstance(net_inputs[0], Sequence):
                    batch_of_species = _input_to_tensor(net_inputs[0], dtype=tf.int32)
                else:
                    batch_of_species = net_inputs[0]

                if isinstance(net_inputs[1], Sequence):
                    batch_of_gvects = _input_to_tensor(net_inputs[1])
                else:
                    batch_of_gvects = net_inputs[1]

                el_inputs = inputs[1]
                if isinstance(el_inputs[0], Sequence):
                    V = _input_to_tensor(el_inputs[0], dtype=tf.int32)
                else:
                    V = el_inputs[0]
                if isinstance(el_inputs[1], Sequence):
                    total_charge = _input_to_tensor(el_inputs[1])
                else:
                    total_charge = el_inputs[1]

                if self._compute_forces: 
                    if isinstance(el_inputs[2], Sequence):
                        V_prime = _input_to_tensor(el_inputs[2])
                    else:
                        V_prime = el_inputs[2]
                
            elif isinstance(inputs, Dict):
                # This is the most common call to create a new model
                data = unpack_data(inputs, self._compute_forces,
                                   self._sparse_derivatives, self._examples_name,
                                   input_format=self._input_format, preprocess=self._preprocess,
                                   long_range_el=True)
                net_inputs = data[0] 
                batch_of_species = net_inputs[0]
                batch_of_gvects = net_inputs[1]

                el_inputs = data[1]
                V = el_inputs[0]
                total_charge = el_inputs[1]
                if self._compute_forces:
                    V_prime = el_inputs[2]

            else:
                raise ValueError('Unclear input format')
        else:
            # We already called unpack in the train_step
            net_inputs = inputs[0] 
            batch_of_species = net_inputs[0]
            batch_of_gvects = net_inputs[1]

            el_inputs = inputs[1]
            V = el_inputs[0]
            total_charge = el_inputs[1]
            if self._compute_forces:
                V_prime = el_inputs[2]

        atoms_presence = tf.where(batch_of_species<self.n_species,1,0)
        batch_n_atoms = tf.reduce_sum(atoms_presence, axis=1, keepdims=True)
        batch_size = tf.shape(batch_of_species)[0]
        electronegativity, atomic_hardness = self.estimate_atomic_quantities(batch_of_species)
        batch_gsize = self.g_size * tf.ones(batch_size, dtype=tf.int64)

        max_n_atoms = tf.shape(batch_of_species)[1]
        max_n_atoms_extended = tf.tile([max_n_atoms], [batch_size])
        max_n_atoms_diff = tf.cast(max_n_atoms_extended, dtype=tf.int32) - batch_n_atoms

        if self._compute_forces:
            #compute charges from exact minimization
            E_el, F_el, atomic_charges, E, dE =\
                    self.tf_electrostatics(batch_n_atoms, batch_of_species,
                          batch_of_gvects, V, total_charge, electronegativity, atomic_hardness, V_prime=V_prime)
            
            #dE^(0)dg + dE^(1)dg * q + 1/2 dE^(2)dg * q^2

            charge_scaled_dedgs = dE[0] + (dE[1] + 
                                           0.5 * dE[2] * atomic_charges[:,:,tf.newaxis]) * atomic_charges[:,:,tf.newaxis]

            forces_pred = self.force_computer(charge_scaled_dedgs,
                                              batch_n_atoms,
                                              net_inputs[2:],
                                              training=training)

            forces_pred += tf.reshape(F_el, [batch_size, max_n_atoms*3])
            
        else:
            E_el, atomic_charges, E =\
                    self.tf_electrostatics(batch_n_atoms, batch_of_species,
                          batch_of_gvects, V, total_charge, electronegativity, atomic_hardness)

        batch_local_energies = tf.reduce_sum(E[0] + (E[1] +
                                              0.5 * E[2] * atomic_charges) * atomic_charges, axis=1)
        #sum electrostatic and local contribution
        energies_pred = E_el + batch_local_energies
        if self._compute_forces:
            return batch_n_atoms, energies_pred, forces_pred, atomic_charges, E
        return batch_n_atoms, energies_pred, atomic_charges, E


    def compile(self,
                e_loss=None,
                f_loss=None,
                q_loss=None,
                f_cost=0.0,
                q_cost=0.0,
                energy_example_weight=None,
                force_example_weight=None,
                charge_example_weight=None,
                optimizer='adam'):
        super().compile(e_loss=e_loss,
                        optimizer=optimizer,
                        f_loss=f_loss,
                        f_cost=f_cost,
                        energy_example_weight=energy_example_weight,
                        force_example_weight=force_example_weight)
        #tf.keras.Model.compile(self, optimizer=optimizer)

        self._e_loss = e_loss
        self._energy_example_weight = energy_example_weight
        self._force_example_weight = force_example_weight
        self._charge_example_weight = charge_example_weight
        self._f_loss = f_loss
        self._q_loss = q_loss
        self._f_cost = f_cost
        self._q_cost = q_cost
        

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data = unpack_data(data, self._compute_forces,
                                   self._sparse_derivatives, self._examples_name,
                                   input_format=self._input_format, preprocess=self._preprocess,
                                   long_range_el=True)
        net_inputs = data[0]
        
        el_inputs = data[1]
        V = el_inputs[0]
        total_charge = el_inputs[1]
        if self._compute_forces:
            V_prime = el_inputs[2]


        batch_energies_ref, batch_forces_ref = data[2]
        batch_charge_ref = data[3]


        with tf.GradientTape() as tape:
            if self._compute_forces:
                batch_inputs = (net_inputs, (V, total_charge, V_prime))
                batch_n_atoms, energies_pred, forces_pred, charge_pred, E = self(batch_inputs, training=True)
            else:
                batch_inputs = (net_inputs, (V, total_charge, None))
                batch_n_atoms, energies_pred, charge_pred, E = self(batch_inputs, training=True)

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

            if self._q_loss:
                if self._charge_example_weight is not None:
                    q_weight = self._charge_example_weight(batch_n_atoms)
                else:
                    q_weight = None
                q_loss = self._q_loss(batch_charge_ref,
                                      charge_pred,
                                      sample_weight=q_weight)
            else:
                q_loss = 0

            
            reg_loss = self._model_inherited_losses()
            loss = e_loss + self._f_cost * f_loss + reg_loss + self._q_cost * q_loss
           # loss = e_loss + f_loss + q_loss
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
            tf.summary.scalar('1. Losses/3. Forces',f_loss*self._f_cost,self._train_counter)
        if self._q_loss:
            tf.summary.scalar('1. Losses/5. Charges',q_loss*self._q_cost,self._train_counter)

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
        if self._q_loss:
            Nq = tf.cast(tf.reduce_sum(batch_n_atoms),float)
            qrmse = tf.sqrt(tf.reduce_sum((batch_charge_ref-charge_pred)**2)/Nq)
            tf.summary.scalar('2. Metrics/5. charge RMSE',qrmse,self._train_counter)
            qmae = tf.reduce_sum(tf.abs(batch_charge_ref-charge_pred))/Nq
            tf.summary.scalar('2. Metrics/6. charge MAE',qmae,self._train_counter)


        metrics = {'tot_st': self._train_counter}
        if 'MAE' in self._printmetrics:
            metrics.update({'MAE/at': maeat})
            if self._f_loss:
                metrics.update({'F_MAE': Fmae})
            if self._q_loss:
                metrics.update({'Q_MAE': qmae})

        if 'RMSE' in self._printmetrics:
            metrics.update({'RMSE/at': rmseat})
            if self._f_loss:
                metrics.update({'F_RMSE': Frmse})
            if self._q_loss:
                metrics.update({'Q_RMSE': qrmse})
        if 'loss' in self._printmetrics:
            metrics.update({'loss': loss, 'e_loss': e_loss})
            if self._default_network_config[1].kernel_regularizer != None:
                metrics.update({'reg_loss': reg_loss})
            if self._f_loss:
                metrics.update({'f_loss': f_loss})
            if self._q_loss:
                metrics.update({'q_loss': q_loss})
        #historgram of local terms and charges
        tf.summary.histogram("E0", E[0], self._train_counter)
        tf.summary.histogram("E1", E[1], self._train_counter)
        tf.summary.histogram("E2", E[2], self._train_counter)
        tf.summary.histogram("charges", charge_pred, self._train_counter)


        return metrics

    def test_step(self, data):
        data = unpack_data(data, self._compute_forces, self._sparse_derivatives,
                           self._examples_name, input_format=self._val_input_format, 
                           preprocess=self._preprocess, long_range_el=True)
        net_inputs = data[0]
        el_inputs = data[1]
        V = el_inputs[0]
        total_charge = el_inputs[1]
        if self._compute_forces:
            V_prime = el_inputs[2]


        batch_energies_ref, batch_forces_ref = data[2]
        batch_charge_ref = data[3]


        prediction = {}
        if self._examples_name:
            prediction['names'] = data[4]
        if self._compute_forces:
            batch_inputs = (net_inputs, (V, total_charge, V_prime))
            batch_n_atoms, energies_pred, forces_pred, charge_pred, E = self(batch_inputs, training=True)
            prediction['forces'] = forces_pred, batch_forces_ref
        else:
            batch_inputs = (net_inputs, (V, total_charge, None))
            batch_n_atoms, energies_pred, charge_pred, E = self(batch_inputs, training=True)
        
        prediction['charges'] = charge_pred, batch_charge_ref
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
        if self._q_loss:
            Nq = tf.cast(tf.reduce_sum(batch_n_atoms),float)
            qrmse = tf.sqrt(tf.reduce_sum((batch_charge_ref-charge_pred)**2)/Nq)
            tf.summary.scalar('2. Metrics/5. charge RMSE',qrmse,self._train_counter)
            qmae = tf.reduce_sum(tf.abs(batch_charge_ref-charge_pred))/Nq
            tf.summary.scalar('2. Metrics/6. charge MAE',qmae,self._train_counter)


        metrics = {'tot_st': self._train_counter}
        if 'MAE' in self._printmetrics:
            metrics.update({'MAE/at': maeat})
            if self._f_loss:
                metrics.update({'F_MAE': Fmae})
            if self._q_loss:
                metrics.update({'Q_MAE': qmae})

        if 'RMSE' in self._printmetrics:
            metrics.update({'RMSE/at': rmseat})
            if self._f_loss:
                metrics.update({'F_RMSE': Frmse})
            if self._q_loss:
                metrics.update({'Q_RMSE': qrmse})
        if 'loss' in self._printmetrics:
            metrics.update({'loss': loss, 'e_loss': e_loss})
            if self._default_network_config[1].kernel_regularizer != None:
                metrics.update({'reg_loss': reg_loss})
            if self._f_loss:
                metrics.update({'f_loss': f_loss})
            if self._q_loss:
                metrics.update({'q_loss': q_loss})


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
                           self._examples_name, input_format=self._val_input_format,
                           preprocess=self._preprocess, long_range_el=True)

        net_inputs = data[0]
        el_inputs = data[1]
        V = el_inputs[0]
        total_charge = el_inputs[1]
        if self._compute_forces:
            V_prime = el_inputs[2]


        batch_energies_ref, batch_forces_ref = data[2]
        batch_charge_ref = data[3]

        # Padding only for older versions of TF2 where this causes problems
        if self._max_atoms > 0:
            sh = tf.shape(net_inputs[0])
            batch_forces_ref = tf.concat([batch_forces_ref,
                                  tf.zeros( (sh[0],3*(self._max_atoms-sh[1])) )], 1)


        if self._compute_forces:
            batch_inputs = (net_inputs, (V, total_charge, V_prime))
            batch_n_atoms, energies_pred, forces_pred, charge_pred, E = self(batch_inputs, training=False)
            # Padding only for older versions of TF2 where this causes problems
            if self._max_atoms > 0:
                forces_pred = tf.concat([forces_pred,
                                 tf.zeros( (sh[0],3*(self._max_atoms-sh[1])) )], 1)
            output = [energies_pred, forces_pred, charge_pred]
        else:
            batch_inputs = (net_inputs, (V, total_charge, None))
            batch_n_atoms, energy_pred, charge_pred, E  = self(batch_inputs, training=False)
            output = [energies_pred, charge_pred]

        if self._examples_name:
            batch_names = data[4]
        else:
            batch_names = tf.constant([b'N.A.'])
        if batch_forces_ref is None:
            # all the output must be tensors otherwise
            # multithreading fail to stack the outputs
            batch_forces_ref = tf.constant([0])
        if batch_charge_ref is None:
            batch_charge_ref = tf.constant([0])

        return output, tf.reshape(batch_n_atoms, [-1]), \
            (batch_energies_ref, batch_forces_ref, batch_charge_ref), batch_names

    @property
    def tfr_parse_function(self):
        g_size = self[self.atomic_sequence[0]].feature_size
        return ParseFn(g_size=g_size, n_species=self.n_species, long_range_el=True)

    @property
    def name(self):
        return self._name

    def dump_network_lammps(self, folder, file_name, **kwargs):
        logger.info(f'GVERSION {kwargs["gversion"]}')
        s = ''
        # lammps parameters string composition
        if int(kwargs['gversion']) == 1:
             logger.info('MY GVERSION {}'.format(kwargs['gversion']))
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

        gaussian_width = [element(species).covalent_radius * 0.01 * np.sqrt(2.0)
                          for species in self.atomic_sequence]

        IE = np.asarray([element(species).ionenergies[1] \
                for species in self.atomic_sequence])
        EA = np.asarray([0.0 if not element(species).electron_affinity else \
                element(species).electron_affinity for species in self.atomic_sequence])

        species_hardness = IE - EA
        species_electronegativity = 0.5 * (IE + EA)

        species_hardness = species_hardness.tolist()
        species_electronegativity = species_electronegativity.tolist()

        s += f'\n[LONG_RANGE]\n'
        s += f'gaussian_width = {gaussian_width}\n'
        s += f'atomic_hardness = {species_hardness}\n'
        s += f'electronegativity = {species_electronegativity}\n'
        s += f'min_eigenvalue = {0.0}\n'

        with open(os.path.join(folder, file_name), 'w') as f:
            f.write(s)



def create_panna_model_with_electrostatics(model_params, validation_params = None):
    """Create a new PANNA+electrostatic model with the proper parameters."""
    if validation_params:
        val_input_format = validation_params.input_format
    else:
        val_input_format = model_params.input_format
    panna_model_el = PannaModel_with_electrostatics(model_params.g_size, 
                             default_nn_config=model_params.default_nn_config,
                             compute_forces=model_params.compute_forces,
                             sparse_derivatives=model_params.sparse_derivatives,
                             examples_name=model_params.examples_name,
                             input_format=model_params.input_format,
                             val_input_format=val_input_format,
                             preprocess=model_params.preprocess,
                             name='PANNA_el',
                             max_atoms=model_params.max_atoms,
                             metrics=model_params.metrics,
                             long_range_el=True)
    for network_name, config in model_params.networks_config:
        Network = _get_network(network_name)
        if not config.is_ready: 
            raise ValueError(f'{config.name} not ready')
        network = Network(**asdict(config))
        panna_model_el[config.name] = network
    return panna_model_el
