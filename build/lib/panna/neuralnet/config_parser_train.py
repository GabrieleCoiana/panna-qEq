"""Parser for the config file."""

import logging
import os

from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict
from typing import Callable
from typing import Optional
from typing import Sequence
import configparser
from abc import ABC

import numpy as np
import json

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2

from panna.neuralnet.parse_fn import ParseFn
from panna.neuralnet.inputs_iterator import input_pipeline
from panna.neuralnet.energy_losses import get_energy_loss
from panna.neuralnet.force_losses import get_force_loss
from panna.neuralnet.charge_losses import get_charge_loss
import panna.neuralnet.a2affnetwork as a2affnetwork
from panna.neuralnet.a2affnetwork import NetworkConfig
from panna.gvector import GvectmBP
from panna.lib.parser_callable import converters

logger = logging.getLogger(__name__)


@dataclass
class ModelParams():
    """Parameters need to build the model."""
    default_nn_config: Union[NetworkConfig, None]
    networks_config: List[NetworkConfig]
    g_size: int
    sparse_derivatives: bool
    compute_forces: bool
    examples_name: bool = False
    input_format: str = 'tfr'
    preprocess: ABC = None
    max_atoms: Optional[int] = -1
    metrics: List[str] = field(default_factory=list)
    long_range_el: bool = False
    constrain_even_terms: bool = False
    min_eigenvalue: bool = 0.0


@dataclass
class CompileParams():
    """Parameters needed at model compilation time."""

    opt: optimizer_v2.OptimizerV2
    e_loss: tf.keras.losses.Loss
    f_loss: Union[tf.keras.losses.Loss, None]
    q_loss: Union[tf.keras.losses.Loss, None]
    f_cost: float
    q_cost: float
    energy_example_weight: Optional[Callable]
    force_example_weight: Optional[Callable]
    charge_example_weight: Optional[Callable]

@dataclass
class FitParams():
    """Parameters need to perform the fit call.

    Parameters
    ----------
    opt: optimizer
    dataset: dataset
    max_training: (max_epoch, steps_per_epoch)
    io_callbacks: list of callbacks
    restart_info: info for restarting from a ckpt
      after a properly terminated run.
    """

    dataset: tf.data.Dataset
    max_training: Tuple[int, int]
    io_callbacks: List[tf.keras.callbacks.Callback]
    restart_info: Union[Dict, None]


@dataclass
class ValidationParams():
    """Parameters used in the validation step."""

    dataset: Union[tf.data.Dataset, None]
    input_format: str = 'tfr'


@dataclass
class DataInfo():
    """Data information."""

    atomic_sequence: List[str]
    sparse_derivatives: bool
    energy_rescale: float
    examples_name: bool
    max_atoms: Optional[int]

    @property
    def n_species(self):
        return len(self.atomic_sequence)

def parameter_file_parser(filename: str) -> configparser.ConfigParser:
    """Pass converters to configparser."""
    config = configparser.ConfigParser(converters=converters)
    logger.info('reading {}'.format(filename))
    config.read(filename)
    if not config.has_section('PARALLELIZATION'):
        config.add_section('PARALLELIZATION')
    return config

def parse_metadata(config, md):
    if not config[1].feature_size:
        config[1].feature_size = md['feature_size']
    if not config[1].layers_size:
        config[1].layers_size = md['layers']
    if not config[1].layers_activation:
        config[1].layers_activation = md['activations']
    if not config[1].offset:
        config[1].offset = md['offset']
    return config

def parse_network(config):
    default_net_params = config[
        'DEFAULT_NETWORK'] if 'DEFAULT_NETWORK' in config else None
    default_nn_config = _parse_default_network(default_net_params)

    data_info = config['DATA_INFORMATION'] if 'DATA_INFORMATION' in config else None
    data_info = _parse_data_info(data_info, default_nn_config)

    # atomic_sequence contains [(net_type, net_params)]
    networks_config = []
    for species in data_info.atomic_sequence:
        # If there are metadata, we load them here and add only unset values
        if species[1].metadata:
            # Looking for matching species
            md = next((n for n in species[1].metadata['networks'] \
                       if n['name'] == species[1].name), None)
            if md:
                species = parse_metadata(species, md)
        if species[1].name in config:
            logger.info('=={}== Found network specifications'.format(species[1].name))
            species_config = config[species[1].name]
            networks_config.append(_parse_species_network(species[1].name, species_config, species))
        else:
            if species[0] is not None:
                networks_config.append(_parse_species_network(species[1].name, None, species))
                # networks_config.append(species)
    return default_nn_config, networks_config, data_info


def _parse_validation_options(config, batch_size, parse_fn, input_format, preprocess):
    if 'VALIDATION_OPTIONS' in config:
        val_input_format = config['VALIDATION_OPTIONS'].get('input_format', input_format)
        validation_dataset = _create_validation_dataset(config['VALIDATION_OPTIONS'],
                                                        batch_size, parse_fn,
                                                        val_input_format, preprocess)
    else:
        validation_dataset = None
        val_input_format = None
    return ValidationParams(validation_dataset, val_input_format)


def parse_config(config):
    """Main training parser call"""
    _ = config['PARALLELIZATION']

    # Fit parameters
    training_params = config['TRAINING_PARAMETERS']
    kernel_regularizer, bias_regularizer,\
        max_training, compile_params = _parse_training_params(training_params)
    if not config.has_section('LONG_RANGE'):
        long_range_el = False
    else:
        LR_info = config['LONG_RANGE']
        long_range_el = LR_info.getboolean('long_range_el', False)

    f_loss = compile_params.f_loss
    compute_forces = bool(f_loss)
    batch_size = training_params.getint('batch_size')

    # Neural Network and data
    default_nn_config, networks_config, data_info = parse_network(config)

    io_info = config['IO_INFORMATION']
    input_format = io_info.get('input_format', 'tfr')
    parse_fn = ParseFn(g_size=networks_config[0][1].feature_size,
                       n_species=data_info.n_species,
                       forces=compute_forces,
                       sparse_dgvect=data_info.sparse_derivatives,
                       energy_rescale=data_info.energy_rescale,
                       long_range_el=long_range_el)

    default_nn_config[1].kernel_regularizer = kernel_regularizer
    default_nn_config[1].bias_regularizer = bias_regularizer
    if compute_forces:
        default_nn_config[1].compute_jacobian = True

    for _, nn_config in networks_config:
        nn_config.kernel_regularizer = kernel_regularizer
        nn_config.bias_regularizer = bias_regularizer
        if compute_forces:
            nn_config.compute_jacobian = True

    g_size = networks_config[0][1].feature_size

    # fit section
    io_callbacks, restart_info, metrics = _parse_io_callbacks(io_info)
    train_dataset, preprocess, input_format = \
                   _create_training_dataset(io_info, batch_size, parse_fn)
    fit_params = FitParams(train_dataset, max_training, io_callbacks, restart_info)

    model_params = ModelParams(default_nn_config, networks_config, g_size,
                               data_info.sparse_derivatives, compute_forces,
                               input_format=input_format, preprocess=preprocess,
                               max_atoms=data_info.max_atoms, metrics=metrics,
                               long_range_el=long_range_el)

    # validate section
    validation_params = _parse_validation_options(config, batch_size, parse_fn,\
                                                input_format, preprocess)
    
    
    return model_params, compile_params, fit_params, validation_params

def _parse_data_info(data_info, default_nn):
    default_nn_type, default_nn_config = default_nn
    energy_rescale = 1
    sparse_derivatives = False
    examples_name = True
    in_format = 'tfr'
    max_atoms = -1
    if data_info is not None:
        energy_rescale = data_info.getfloat('energy_rescale', energy_rescale)
        sparse_derivatives = data_info.getboolean('sparse_derivatives',
                                                  sparse_derivatives)
        examples_name = data_info.getboolean('examples_name', examples_name)

        atomic_sequence = data_info.get_comma_list('atomic_sequence', None)
        offsets = data_info.get_comma_list_floats('output_offset', None)

        if offsets and atomic_sequence:
            if len(offsets) != len(atomic_sequence):
                raise ValueError('output offset must be of '
                                 'the same size as atomic sequence')
            tmp = zip(atomic_sequence, offsets)
        elif offsets and not atomic_sequence:
            raise ValueError('output offset can not be specified '
                             'without atomic_sequence')
        else:
            tmp = atomic_sequence
        max_atoms = data_info.getint('max_atoms', -1)

    else:
        tmp = []
    atomic_sequence = []
    for x in tmp:
        nn_config = deepcopy(default_nn_config)
        nn_config.name = x[0]
        if len(x) == 2:
            nn_config.offset = x[1]
        atomic_sequence.append((default_nn_type, nn_config))

    return DataInfo(atomic_sequence, sparse_derivatives, energy_rescale, \
                    examples_name, max_atoms=max_atoms)


def _find_last(ckpt_folder):
    if os.path.exists(f'{ckpt_folder}/checkpoint'):
        with open(f'{ckpt_folder}/checkpoint') as f:
            lines = f.readlines()
        file_name = lines[0].strip().split()[1].replace("\"", '')
        _, epoch_n, _, step_n = file_name.split('_')
        return file_name, int(epoch_n), int(step_n)
    else:
        logger.info('No checkpoint found, starting training from scratch')
        return None, 0, 0


class OneTimeSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, step, path):
        super().__init__()
        self._step = step
        self._path = path

    def on_train_batch_end(self, batch, logs=None):
        if logs['tot_st']==self._step:
            self.model.save(self._path)

class SetStepCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_step):
        super().__init__()
        self._initial_step = initial_step

    def on_train_begin(self, logs=None):
        self.model._train_counter.assign(self._initial_step)

# Subclassing the ProgbarLogger callback to set the interval, not exposed
class CustomProgbarCallback(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, count_mode: str = "steps", stateful_metrics=None,
                       interval: float = 0.02):
        super().__init__(count_mode, stateful_metrics)
        self.interval = interval

    def _maybe_init_progbar(self):
        super()._maybe_init_progbar()
        self.progbar.interval = self.interval


def _parse_io_callbacks(io_info):
    train_dir = io_info.get('train_dir')
    max_ckpt_to_keep = io_info.getint('max_ckpt_to_keep', 1000)
    log_frequency = io_info.getint('log_frequency', 1)
    save_checkpoint_steps = io_info.getint('save_checkpoint_steps', log_frequency)
    restart_mode = io_info.get('restart_mode', 'default')
    profile_range = io_info.get_1or2_ints('profile', 0)
    progbar_interval = io_info.getfloat('progbar_interval', 1.)
    metrics = io_info.get_comma_list('metrics', ['MAE'])

    if 'avg_speed_step' in io_info:
        logger.warning('avg_speed_step deprecated')

    if 'log_device_placement' in io_info:
        logger.warning('log_device_placement not used for now')

    # long story short, save_weights_only *MUST* be true.
    # AND THIS DOES NOT SAVE ONLY THE WEIGHTS but the model
    # and ALL ITS DEPENDENCIES.
    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        train_dir + '/_models/epoch_{epoch}_step_{tot_st}',
        save_weights_only=True,
        save_freq=save_checkpoint_steps,
        options=None)
    # Note: hist_freq is in epochs, so we put 1 for now
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_dir,
                                                          histogram_freq=1,
                                                          write_graph=True,
                                                          write_images=True,
                                                          #write_steps_per_second=True,
                                                          update_freq=log_frequency,
                                                          profile_batch=profile_range,
                                                          embeddings_freq=0,
                                                          embeddings_metadata=None)
    # Overriding these function with an empy one to avoid epoch reports in TB
    # If we want to extend them, see originals in
    # https://github.com/keras-team/keras/blob/master/keras/callbacks.py
    tensorboard_callback._log_epoch_metrics = lambda x,y: None
    tensorboard_callback.on_test_end = lambda x: None

    # Adding a custom progress bar callback by hand to tune it:
    # setting tot_st as stateful metric (not averaged)
    # and changing refresh interval to 1s (can be made custom?)
    progbar_callback = CustomProgbarCallback(
                        count_mode= "steps",
                        stateful_metrics=['tot_st'],
                        interval = progbar_interval)

    # Adding a callback to save metrics to a text file
    logger_callback = tf.keras.callbacks.CSVLogger(
            train_dir + '/metrics.dat', separator=' ', append=True)

    callbacks = [model_ckpt_callback, tensorboard_callback, progbar_callback, logger_callback]

    # Adding optional specific one-time saves
    starting_model_save = io_info.getint('starting_model_save', -1)
    if starting_model_save>-1:
        callbacks.append(OneTimeSaveCallback(starting_model_save, train_dir+'/starting_model'))

    final_model_save = io_info.getint('final_model_save', -1)
    if final_model_save>-1:
        callbacks.append(OneTimeSaveCallback(final_model_save, train_dir+'/final_model'))

    restart_info = None
    if restart_mode == 'default' or restart_mode == 'force_default':
        # Checking if there is no tmp_backup but there is a _models
        if ((not os.path.isdir(train_dir + '/tmp_backup')) or \
             len(os.listdir(train_dir + '/tmp_backup'))==0) and \
            os.path.isfile(train_dir + '/_models/checkpoint'):
            logger.info('No files to perform a default restart, but a checkpoint exists!')
            if restart_mode == 'default':
                logger.info('Please move _models folder or use flag force_default to proceed.')
                raise ValueError('Restart incompatible with existing files, training will be stopped.')
            else:
                logger.info('Training will restart from scratch.')
        bk_and_restore_callback = tf.keras.callbacks.BackupAndRestore(
            train_dir + '/tmp_backup')
        callbacks.append(bk_and_restore_callback)
    elif restart_mode == 'continue_from_last_ckpt':
        # this does not contain information on the state
        # of the inputs!! It knows the learning rate though
        file_name, epoch, step = _find_last(train_dir + '/_models')
        if file_name:
            file_name = train_dir + f'/_models/{file_name}'
            restart_info = {'file_name': file_name, 'step': step, 'epoch': epoch}
            callbacks.append(SetStepCallback(step))
    elif restart_mode == 'specific_ckpt':
        epoch = io_info.get('restart_epoch', 0)
        step = io_info.get('restart_step', 0)
        file_name = io_info.get('restart_file', None) 
        restart_info = {'file_name': file_name, 'epoch': epoch, 'step': step}
    elif restart_mode == 'metadata':
        logger.info('Weights will be loaded from metadata.')
    else:
        pass
    return callbacks, restart_info, metrics


def _create_validation_dataset(validation_options, batch_size, parse_fn,
                               input_format='tfr', preprocess=None):
    data_dir = validation_options.get('data_dir', None)
    batch_size = validation_options.getint('batch_size', batch_size)
    if not data_dir:
        logger.info('No validation will be performed on the fly, missing data_dir')
        return
    if input_format=='tfr':
        dataset = input_pipeline(data_dir=data_dir,
                                 batch_size=batch_size,
                                 parse_fn=parse_fn,
                                 oneshot=True)
    elif input_format=='example':
        extra_data = {
            'mincut': min(preprocess.gvect['Rc_rad'],preprocess.gvect['Rc_ang']),
            'maxcut': max(preprocess.gvect['Rc_rad'],preprocess.gvect['Rc_ang']),
            'species': preprocess.species_idx_2str}
        dataset = input_pipeline(data_dir=data_dir,
                                 batch_size=batch_size,
                                 parse_fn=parse_fn,
                                 oneshot=True,
                                 input_format=input_format,
                                 extra_data=extra_data)
    return dataset

def example_param_parser(gvect_ini, forces):
    if gvect_ini==None:
            raise ValueError('Parameter file is needed to compute g.')
    gconfig = configparser.ConfigParser(converters=converters)
    gconfig.read(gvect_ini)
    # For now parsing by hand and we only support mBP
    g_type = gconfig['SYMMETRY_FUNCTION'].get('type')
    species = gconfig['SYMMETRY_FUNCTION'].get('species', None)
    if g_type == 'mBP':
        preprocess = GvectmBP(species=species,compute_dgvect=forces)
        preprocess.parse_parameters(gconfig['GVECT_PARAMETERS'])
    else:
        raise ValueError('Only mBP can be computed at runtime.')        
    extra_data = {
        'mincut': min(preprocess.gvect['Rc_rad'],preprocess.gvect['Rc_ang']),
        'maxcut': max(preprocess.gvect['Rc_rad'],preprocess.gvect['Rc_ang']),
        'species': preprocess.species_idx_2str}
    return preprocess, extra_data

def _create_training_dataset(io_info, batch_size, parse_fn):
    data_dir = io_info.get('data_dir', 'tfrs')
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(data_dir)

    shuffle_buffer_size_multiplier = io_info.getint('shuffle_buffer_size_multiplier',
                                                    10)
    prefetch_buffer_size_multiplier = io_info.getint('prefetch_buffer_size_multiplier',
                                                     10)
    num_parallel_readers = io_info.getint('num_parallel_readers', 1)
    num_parallel_calls = io_info.getint('num_parallel_calls', 1)
    cache = io_info.getboolean('dataset_cache', False)
    shuffle = io_info.getboolean('dataset_shuffle', True)

    input_format = io_info.get('input_format', 'tfr')
    if input_format=='tfr':
        preprocess = None
        extra_data = {}
    elif input_format=='example':
        # Loading gvector parameters
        # For now we reuse the gvect.ini
        # Can be made its own card in train.ini
        preprocess, extra_data = example_param_parser(io_info.get('gvect_ini', None),
                                                      parse_fn._forces)
    else:
        raise ValueError('Unknown input format.')
    
    dataset = input_pipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        parse_fn=parse_fn,
        shuffle_buffer_size_multiplier=shuffle_buffer_size_multiplier,
        prefetch_buffer_size_multiplier=prefetch_buffer_size_multiplier,
        num_parallel_readers=num_parallel_readers,
        num_parallel_calls=num_parallel_calls,
        cache=cache,
        shuffle=shuffle,
        input_format=input_format,
        extra_data=extra_data)
    
    return dataset, preprocess, input_format


def _parse_parallel_section(parallel_info):
    old_keys = {
        'shuffle_buffer_size_multiplier', 'prefetch_buffer_size_multiplier',
        'num_parallel_readers', 'num_parallel_calls', 'dataset_cache'
    }
    for key in set(parallel_info.keys()).intersection(old_keys):
        print(f'{key} moved to io section')


def _parse_training_params(training_params):
    # creation of the lr function
    training_schedule = training_params.getfloat('learning_rate', 1e-3)

    if not training_params.getboolean('learning_rate_constant', True):
        training_schedule = ExponentialDecay(
            initial_learning_rate=training_schedule,
            decay_steps=training_params.getfloat('learning_rate_decay_step'),
            decay_rate=training_params.getfloat('learning_rate_decay_factor'),
            staircase=training_params.getboolean('staircase_decay_type', False))

    # creation of the loss quantities

    kernel_regularizations = L1L2(training_params.getfloat('wscale_l1', 0.0),
                                  training_params.getfloat('wscale_l2', 0.0))
    bias_regularizations = L1L2(training_params.getfloat('bscale_l1', 0.0),
                                training_params.getfloat('bscale_l2', 0.0))

    opt = Adam(learning_rate=training_schedule,
               beta_1=training_params.getfloat('beta1', 0.9),
               beta_2=training_params.getfloat('beta2', 0.999),
               epsilon=training_params.getfloat('adam_eps', 1e-7),
               amsgrad=False)
    if training_params.getfloat('clip_value', 0.0) != 0.0:
        # Probably this is just a regularizer that must be passed to the layers.
        raise NotImplementedError('TODO, not reimplemented in panna')

    e_loss, energy_example_weight = get_energy_loss(
        training_params.get('loss_function', 'quad'),
        weight=training_params.get('energy_weight', None),
        reduction=training_params.get('energy_reduction', 'mean'),
        prefact=training_params.getfloat('energy_prefactor', None))

    f_cost = training_params.getfloat('forces_cost', 0.0)
    if f_cost > 0:
        f_loss, force_example_weight = get_force_loss(
            training_params.get('floss_function', 'quad'),
            weight=training_params.get('force_weights', None),
            reduction=training_params.get('force_reduction', 'mean'))
    else:
        f_loss = None
        force_example_weight = None

    q_cost = training_params.getfloat('charge_cost', 0.0)
    if q_cost > 0:
        q_loss, charge_example_weight = get_charge_loss(
            training_params.get('qloss_function', 'quad'),
            weight=training_params.get('charge_weights', None),
            reduction=training_params.get('charge_reduction', 'mean'))
    else:
        q_loss = None
        charge_example_weight = None

    if 'max_steps' in training_params:
        logger.warning('max_step is deprecated, please use max_epochs '
                       'and steps_per_epoch, for now you data will be '
                       'arbitrary split in 10 epochs')
        max_steps = training_params.getint('max_steps')
        max_epochs = 10
        steps_per_epoch = int(max_steps / 10)
        # TODO add to documentation: max_steps = step_per_epoch * max_epoch'
        # there is NO DIFFERENCE between training for max_step an the
        # product of the two new quantities. This is simply a technical choice
        # to make the save procedures easier when using distributed strategies.
        # Be aware than now only the epochs can be restarted seamlessly and
        # creating epoch checkpoint is considerably expensive.
    else:
        max_epochs = training_params.getint('max_epochs')
        steps_per_epoch = training_params.getint('steps_per_epoch')

    max_training = (max_epochs, steps_per_epoch)
    compile_params = CompileParams(opt, e_loss, f_loss, q_loss, f_cost, q_cost, energy_example_weight,
                                   force_example_weight, charge_example_weight)
    return kernel_regularizations, bias_regularizations,\
        max_training, compile_params


def _normalize_nn_type(nn_type: str):
    if nn_type in ['a2aff', 'A2AFF', 'ff', 'a2a', 'FF', 'A2A']:
        return 'a2aff'
    else:
        raise ValueError(
            '{} != a2aff : '
            'Only all-to-all feed forward networks supported'.format(nn_type))


def _parse_network_section(net_params, default_network=None):
    nn_type = net_params.get('nn_type', 'a2aff')

    # If there is a default network, it is a tuple with
    # (type, network_object)
    if default_network is not None:
        if default_network[0] is None:
            # we create a base class, we need to take the proper child class
            if nn_type == 'a2aff':
                network_config = a2affnetwork.NetworkConfig(
                    **asdict(default_network[1]))
        elif nn_type != default_network[0]:
            # here we can load different NN
            raise NotImplementedError()
        else:
            network_config = default_network[1]
    else:
        network_config = None

    if nn_type == 'a2aff':
        g_size = net_params.getint('g_size', None)
        architecture = net_params.get_network_architecture('architecture', None)
        layers_trainable = net_params.get_network_trainable('trainable', None)
        layers_activation = net_params.get_network_act('activations', None)
        offset = net_params.getfloat('output_offset', None)
        if network_config is None:
            network_config = a2affnetwork.NetworkConfig(
                feature_size=g_size,
                layers_size=architecture,
                layers_trainable=layers_trainable,
                layers_activation=layers_activation,
                offset=offset)
        else:
            if g_size is not None:
                network_config.feature_size = g_size
            if architecture is not None:
                network_config.layers_size = architecture
            if layers_trainable is not None:
                network_config.layers_trainable = layers_trainable
            if layers_activation is not None:
                network_config.layers_activation = layers_activation
            if offset is not None:
                network_config.offset = offset
        norm_tech = net_params.get('normalization_technique', None)
        if norm_tech:
            raise NotImplementedError()

    # We check if there are metadata, this should only happen in the default network card
    # so we store the parameters to be used when parsing single networks
    metadata = net_params.get('networks_metadata', None)
    if metadata:
        metafile = os.path.join(metadata, 'networks_metadata.json')
        if os.path.isfile(metafile):
            with open(metafile, 'r') as f:
                network_config.metadata = json.load(f)            
            network_config.metadata_dir = metadata
            logger.info('Loaded network metadata.')
        else:
            raise ValueError('Could not find metadata file {}'.format(metafile))

    return nn_type, network_config


def _parse_default_network(default_net_params):
    """Parse a default network for PANNA.

    Parameters
    ----------
    default_net_params: default network section from config

    Returns
    -------
    network: network object or None
        A default network if one has been found, None otherwise
    """
    if default_net_params is not None:
        _nn_type, network_config = _parse_network_section(default_net_params)
        logger.info('Found a default network!')
        logger.info('This network size will be used as default for all species unless '
                    'specified otherwise')
    else:
        _nn_type, network_config = None, a2affnetwork.NetworkConfigBase()
        logger.info('No default network configuration is found, '
                    'all species must be fully specified.')

    return _nn_type, network_config


def _parse_species_network(species: str, species_config, net_config):
    """Parse a specie config.

    Parameters
    ----------
    species_name:
        a species name
    species_config:
        a species config
    net_config:
        existing (partial) network configuration
    """
    if species_config:
        nn_type, network_config = _parse_network_section(species_config, net_config)
        # Parsing prunable state and loading masks
        layers_prunable = species_config.get_network_trainable('prunable', None)
        if layers_prunable:
            if nn_type != 'a2aff':
                raise NotImplementedError('Sorry, pruning is only supported for a2aff.')
            network_config.layers_prunable = layers_prunable
            layers_sparsity = species_config.get_comma_list_floats('sparsity', None)
            starting_masks = species_config.get('starting_masks', None)
            pruning_weights = species_config.get('pruning_weights', None)
            layers_mask = make_layer_masks(layers_prunable, layers_sparsity, starting_masks, \
                pruning_weights, [network_config.feature_size]+network_config.layers_size, species)
            if layers_mask is not None:
                network_config.layers_mask = layers_mask
    else:
        nn_type, network_config = net_config

    # layers behavior flag: only makes sense if there is a metadata
    # and if there is no string (or config) we load everything
    if network_config.metadata:
        layers_behavior = ['load' for _ in network_config.layers_size]
        if species_config:
            layers_behavior = species_config.get_network_behavior('behavior', 
                               layers_behavior)
        preload_wbs = []
        for layer_number, behavior in enumerate(layers_behavior):
            if behavior == 'load':
                preload_wbs.append((
                    np.load(os.path.join(network_config.metadata_dir, 
                        network_config.name+'_l'+str(layer_number)+'_w.npy')),
                    np.load(os.path.join(network_config.metadata_dir, 
                        network_config.name+'_l'+str(layer_number)+'_b.npy'))))
            elif behavior == 'new':
                preload_wbs.append(None)
            else:
                raise ValueError('Unknown behavior: {}'.format(behavior))
        network_config.preload_wbs = preload_wbs

    return nn_type, network_config

def make_layer_masks(layers_prunable, layers_sparsity, starting_masks, pruning_weights, sizes, species):
    """
    Create pruning masks: parse previous mask and weights if provided,
    then prune to desired sparsity, or create random sparse mark
    """
    nlay = len(sizes)-1
    nprune = sum(layers_prunable)
    # Consistency checks, formatting lists, loading models
    if len(layers_prunable) != nlay:
        raise ValueError('Wrong size of prunable booleans')
    if layers_sparsity and len(layers_sparsity)>0:
        if len(layers_sparsity)<nprune:
            layers_sparsity = [layers_sparsity[0]]*nlay
        else:
            for i, l in enumerate(layers_prunable):
                if not l:
                    layers_sparsity.insert(i,0.0)
    else:
        layers_sparsity = [0.0]*nlay
    maskmod = None
    weimod = None
    if starting_masks:
        model = tf.keras.models.load_model(starting_masks, compile=False)
        maskmod = model._networks[species]
    if pruning_weights:
        if pruning_weights==starting_masks:
            weimod=maskmod
        else:
            model = tf.keras.models.load_model(pruning_weights, compile=False)
            weimod = model._networks[species]

    masks = []
    for l, prune in enumerate(layers_prunable):
        if prune:
            lsize = (sizes[l],sizes[l+1])
            if weimod:
                ww = np.abs(np.asarray(weimod._layers[l].get_weights()[0]))
                if maskmod:
                    ww *= np.asarray(maskmod._layers[l].get_config()['kernel_constraint']['config']['mask'])
                if ww.shape != lsize:
                    raise ValueError('Mask has incompatible size for species {} layer {} ({} vs {})'.format( \
                        species, l, ww.shape, lsize))
                wlist = ww.flatten()
                ncutoff = int(layers_sparsity[l]*len(wlist))
                wlist.sort()
                thrs = wlist[ncutoff]
                masks.append(np.where(ww<thrs, 0.0, 1.0).astype(np.float32))
            elif maskmod:
                # If a mask exists, but no weights, we keep it if the sparsity is reasonable
                mm = np.asarray(maskmod._layers[l].get_config()['kernel_constraint']['config']['mask'])
                if mm.shape != lsize:
                    raise ValueError('Mask has incompatible size for species {} layer {} ({} vs {})'.format( \
                        species, l, mm.shape, lsize))
                spars = np.sum(mm)/np.sum(np.ones(nn.shape))
                if np.abs(spars-layers_sparsity[l])<0.02:
                    masks.append(mm)
                else:
                    raise ValueError('Existing mask has wrong sparsity for species {} layer {} ({} vs {})'.format( \
                        species, l, spars, layers_sparsity[l]))
            else:
                if layers_sparsity[l]==0.0:
                    masks.append(np.ones(lsize))
                else:
                    masks.append(np.random.choice([0.0,1.0], size=lsize, \
                        p=[layers_sparsity[l],1-layers_sparsity[l]]))
        else:
            masks.append(np.empty(0))
    return masks

# NORM TECH CODE
# check for normalizations
# in the future this operations must be moved in the
# input pipeline

# if norm_tech == 'pca':
#     raise NotImplementedError()
# pca_matrix = np.load(net_params.get('pca_matrix'))
# bias_vector = np.load(net_params.get('g_mean'))
# assert pca_matrix.shape[0] == bias_vector.shape[0]
# # creating the normalization layer
# norm_layer = Layer(pca_matrix.shape, False, 0, 'norm_PCA')
# norm_layer.w_value = pca_matrix
# norm_layer.b_value = -bias_vector @ pca_matrix
# params['norm_layer'] = norm_layer
# elif norm_tech == 'std':
#     raise NotImplementedError()
# path_to_normalized_gvec = net_params.get('path_to_gvec_normalization')
# with open(path_to_normalized_gvec) as f:
#     normalization_json = json.load(f)
#     average_g = np.array(normalization_json['average_G'])
#     sigma_g = np.array(normalization_json['sigma_G'])

# e_n = np.identity(sigma_g.shape[0])
# norm_layer = Layer(e_n.shape, False, 0, 'norm_std')
# norm_layer.w_value = e_n / sigma_g
# norm_layer.b_value = -average_g / sigma_g

# if net_params.getboolean('masked_gvec', False):
#     path_to_gvec_mask = net_params.get('path_to_gvec_mask')
#     index_to_use = np.loadtxt(path_to_gvec_mask, dtype='int')
#     bias_mask = np.zeros(params['feature_size'])
#     bias_mask[index_to_use] = 1
#     weight_mask = np.diagflat(bias_mask)
#     norm_layer.w_value = norm_layer.w_value * weight_mask
#     norm_layer.b_value = norm_layer.b_value * bias_mask

# params['norm_layer'] = norm_layer
