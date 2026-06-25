import configparser
import logging
logger = logging.getLogger('panna')
import pickle
import numpy as np
import jax.numpy as jnp

from panna.lib.parser_callable import converters

def trainjax_parameter_parser(filename):
    config = configparser.ConfigParser(converters=converters)
    logger.info('reading {}'.format(filename))
    config.read(filename)

    pars = {}

    io_info = config['IO_INFORMATION']
    pars['task'] = io_info.get('task', 'train')
    pars['train_dir'] = io_info.get('train_dir', '.')
    pars['num_targets'] = io_info.getint('num_targets',1)
    if pars['num_targets']==1:
        pars['data_dir'] = io_info.get('data_dir', None)
        pars['val_data_dir'] = io_info.get('val_data_dir', None)
    else:
        pars['data_dir'] = []
        pars['val_data_dir'] = []
        for i in range(pars['num_targets']):
            pars['data_dir'].append(io_info.get('data_dir_'+str(i+1), ''))
            pars['val_data_dir'].append(io_info.get('val_data_dir_'+str(i+1), ''))
        if all([d=='' for d in pars['data_dir']]):
            pars['data_dir'] = None
        if all([d=='' for d in pars['val_data_dir']]):
            pars['val_data_dir'] = None
    pars['val_dir'] = io_info.get('val_dir', None)
    pars['load_weights'] = io_info.getboolean('load_weights', False)
    pars['weights_file'] = io_info.get('weights_file', None)
    pars['extract_dir'] = io_info.get('extract_dir', None)
    pars['extract_format'] = io_info.get('extract_format', 'torch')
    pars['extract_target'] = io_info.getint('extract_target', -1)
    pars['extf'] = io_info.get('extract_flag', 'std')
    pars['save_epoch_freq'] = io_info.getint('save_epoch_freq',1)
    pars['data_cache'] = io_info.getboolean('data_cache', False)
    pars['num_parallel_calls'] = io_info.getint('num_parallel_calls', 4)
    pars['metrics'] = io_info.get_comma_list('metrics', ['MAE'])
    pars['mix_factor'] = io_info.getfloat('mix_factor', 0.1)

    data_info = config['DATA_INFORMATION']
    pars['input_format'] = data_info.get('input_format', 'example')
    pars['species'] = data_info.get_comma_list('atomic_sequence')
    pars['species_str'] = data_info.get('atomic_sequence')
    if pars['num_targets']==1:
        pars['offsets'] = data_info.get_comma_list_floats('output_offset', 
                                           [0.]*len(pars['species']))
    else:
        pars['offsets'] = [data_info.get_comma_list_floats('output_offset_'+str(i+1), 
                                           [0.]*len(pars['species']))
                           for i in range(pars['num_targets'])]

    model_info = config['MODEL_INFORMATION']
    pars['model_type'] = model_info.get('model_type', 'MLP')
    pars['pair_mod'] = model_info.getint('pair_mod', 100)
    pars['at_mod'] = model_info.getint('at_mod', 10)
    pars['rand_seed'] = model_info.getboolean('random_weights_seed', False)
    if pars['model_type'] == 'MLP':
        pars['mlp_arch'] = model_info.get_network_architecture('architecture')
        pars['mlp_act'] = model_info.get('mlp_act', 'exp')
        pars['mlp_sp_weights'] = model_info.get('mlp_sp_weights', None)
        pars['mlp_trainable'] = model_info.get('mlp_trainable', None)
        pars['spec_weights'] = model_info.get('spec_weights', 'specific')
        pars['neigh_emb'] = model_info.getboolean('neigh_emb', False)
        pars['descriptor_type'] = model_info.get('descriptor_type', 'LATTE')
        pars['descriptor_shape'] = model_info.get('descriptor_shape', None)
        pars['descriptor_params'] = model_info
        pars['learnsig'] = model_info.getboolean('learnsig', False)
        if not pars['learnsig']:
            pars['sigma'] = model_info.getfloat('sig', 0.0)
        pars['cutoff'] = model_info.getfloat('Rc', 0.0)
        pars['sp_cutoff'] = model_info.get('sp_cutoff', None)
        pars['fixed_descr'] = model_info.getboolean('fixed_descr', False)
        pars['sp_emb_file'] = model_info.get('sp_emb_file', None)


    training_params = config['TRAINING_PARAMETERS']
    pars['batch_size'] = training_params.getint('batch_size')
    pars['learning_rate'] = training_params.get_comma_list('learning_rate')
    pars['l1_coef'] = training_params.getfloat('l1_coef', 0.0)
    pars['epochs'] = training_params.getint('max_epochs')
    pars['epoch_size'] = training_params.getint('steps_per_epoch')
    if pars['num_targets']==1:
        pars['energy_cost'] = training_params.getfloat('energy_cost', 1.0)
        pars['forces_cost'] = training_params.getfloat('forces_cost', 0.0)
        pars['forces'] = pars['forces_cost']>0.0
    else:
        pars['energy_cost'] = training_params.get_comma_list_floats('energy_cost', [1.0])
        if len(pars['energy_cost'])==1:
            pars['energy_cost'] = [pars['energy_cost'][0]]*pars['num_targets']
        if len(pars['energy_cost'])!=pars['num_targets']:
            raise ValueError("Energy cost should have size 1 or num_targets.")
        pars['energy_cost'] = jnp.asarray(pars['energy_cost'])
        pars['forces_cost'] = training_params.get_comma_list_floats('forces_cost', [0.0])
        if len(pars['forces_cost'])==1:
            pars['forces_cost'] = [pars['forces_cost'][0]]*pars['num_targets']
        if len(pars['forces_cost'])!=pars['num_targets']:
            raise ValueError("Forces cost should have size 1 or num_targets.")
        pars['forces'] = any(c>0.0 for c in pars['forces_cost'])
        pars['forces_cost'] = jnp.asarray(pars['forces_cost'])
    pars['energy_loss'] = training_params.get('energy_loss', 'per_mol')

    pars['fixed_variables'] = []

    # Loading optional extra data
    if pars['sp_emb_file']:
        with open(pars['sp_emb_file'], 'rb') as f:
            full_sp_embedding = pickle.load(f)
        pars['sp_embedding'] = np.asarray([full_sp_embedding[s] for s in pars['species']])
        pars['sp_emb_size'] = pars['sp_embedding'].shape[1]
        pars['descriptor_params']['sp_emb_size'] = str(pars['sp_emb_size'])
    else:
        pars['sp_embedding'] = None


    return pars
