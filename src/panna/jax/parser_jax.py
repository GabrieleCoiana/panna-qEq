import configparser
import logging
logger = logging.getLogger('panna')
import pickle
import numpy as np
import shutil, os

from panna.lib.parser_callable import converters
from panna.jax.metrics_jax import metrics#, metrics_LR


def trainjax_parameter_parser(filename, train=True):
    config = configparser.ConfigParser(converters=converters)
    logger.info('reading {}'.format(filename))
    config.read(filename)

    pars = {}
    
    # --- IO INFO ---
    io_info = config['IO_INFORMATION']
    pars['task'] = io_info.get('task', 'train')
    pars['train_dir'] = io_info.get('train_dir', '.')
    pars['data_dir'] = io_info.get('data_dir', None)
    pars['val_data_dir'] = io_info.get('val_data_dir', None)
    pars['val_dir'] = io_info.get('val_dir', None)
    pars['load_weights'] = io_info.getboolean('load_weights', False)
    pars['weights_file'] = io_info.get('weights_file', None)
    pars['extract_dir'] = io_info.get('extract_dir', None)
    pars['extract_format'] = io_info.get('extract_format', 'bin')
    pars['extf'] = io_info.get('extract_flag', 'std')
    pars['save_epoch_freq'] = io_info.getint('save_epoch_freq',1)
    pars['data_cache'] = io_info.getboolean('data_cache', False)
    pars['num_parallel_calls'] = io_info.getint('num_parallel_calls', 4)
    pars['metrics'] = io_info.get_comma_list('metrics', ['MAE'])
    pars['mix_factor'] = io_info.getfloat('mix_factor', 0.1)
    # -----------------
    
    # --- DATA INFO ---
    data_info = config['DATA_INFORMATION']
    pars['input_format'] = data_info.get('input_format', 'example')
    pars['species'] = data_info.get_comma_list('atomic_sequence')
    pars['species_str'] = data_info.get('atomic_sequence')
    pars['offsets'] = data_info.get_comma_list_floats('output_offset', 
                                       [0.]*len(pars['species']))
    pars['J0_offsets'] = data_info.get_comma_list_floats('J0_offsets', 
                                       [np.nan]*len(pars['species']))
    pars['chi0_offsets'] = data_info.get_comma_list_floats('chi0_offsets', 
                                       [np.nan]*len(pars['species']))
    pars['kappa0_offsets'] = data_info.get_comma_list_floats('kappa0_offsets', 
                                       [0.]*len(pars['species']))
    pars['Zed_ref']  = data_info.get_comma_list_floats('Zed_ref', 0.0)
    pars['pbc'] = data_info.getboolean('pbc', True)
    pars['BORN'] = data_info.get('BORN', '')
    # ------------------------
    
    # --- MODEL INFO ---
    model_info = config['MODEL_INFORMATION']
    pars['model_type'] = model_info.get('model_type', 'MLP')
    pars['pair_mod'] = model_info.getint('pair_mod', 100)
    pars['pair_max'] = model_info.getint('pair_max', 0)
    pars['at_mod'] = model_info.getint('at_mod', 10)
    pars['rand_seed'] = model_info.getboolean('random_weights_seed', False)
    # if pars['model_type'] in ['MLP' , 'MLP2', 'MLP_LR', 'MLP_LR_mod', 'MLP_LR_modsr', 'MLP_LR_pol', 'MLP_LR_fq', 'MLP_LR_fq_modsr', 'MLP_LR_fq_pol', 'MLP_LR_fq_pol_modsr']:
    pars['mlp_arch'] = model_info.get_network_architecture('architecture')
    pars['mlp_act'] = model_info.get('mlp_act', 'exp')
    pars['mlp_sp_weights'] = model_info.get('mlp_sp_weights', None)
    pars['mlp_trainable'] = model_info.get('mlp_trainable', None)
    pars['spec_weights'] = model_info.get('spec_weights', 'specific')
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
    # pars['sp_emb_size'] = model_info.getint('sp_emb_size', 0)
    # -----------------


    training_params = config['TRAINING_PARAMETERS']
    pars['batch_size'] = training_params.getint('batch_size')
    pars['learning_rate'] = training_params.get_comma_list('learning_rate')
    pars['l1_coef'] = training_params.getfloat('l1_coef', 0.0)
    pars['epochs'] = training_params.getint('max_epochs')
    pars['epoch_size'] = training_params.getint('steps_per_epoch')
    pars['energy_cost'] = training_params.getfloat('energy_cost', 1.0)
    pars['forces_cost'] = training_params.getfloat('forces_cost', 0.0)
    pars['charges_cost'] = training_params.getfloat('charges_cost', 0.0)
    pars['dipoles_cost'] = training_params.getfloat('dipoles_cost', 0.0)
    pars['born_cost'] = training_params.getfloat('born_cost', 0.0)
    pars['stress_cost'] = training_params.getfloat('stress_cost', 0.0)
    pars['energy_loss'] = training_params.get('energy_loss', 'per_mol')
    pars['fixed_variables'] = []
    
    # for long range models
    if 'LR' in pars['model_type']: #pars['model_type'] in ['MLP_LR', 'MLP_LR_mod', 'MLP_LR_modsr', 'MLP_LR_pol','MLP_LR_fq', 'MLP_LR_fq_modsr', 'MLP_LR_fq_pol', 'MLP_LR_fq_pol_modsr']:
        lr_info = config['LONG_RANGE']
        pars['precompute_ewald_sums'] = lr_info.getboolean('precompute_ewald_sums', False)
        pars['verbose'] = lr_info.getboolean('verbose', False)
        pars['gaussian_charges'] = lr_info.get_comma_list('gaussian_charges', [b'False'])

        try: # to find if the user has provided Ewald parameters
            acc_factor = lr_info.getfloat('acc_factor', 1e-06)
            real_space_cutoff = lr_info.getfloat('real_space_cutoff', 0.0)#, dtype=float)
            rec_space_cutoff = lr_info.getint('rec_space_cutoff', 0)#, dtype=float)
            gaussian_smearing = lr_info.getfloat('gaussian_smearing', 0.0)#, dtype=float)
            # lets bundle the Ewald parameters into an array
            ewald_input_params = np.array([acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing])
            pars['ewald_input_params'] = ewald_input_params

        except KeyError: # otherwise use the default values
            ewald_input_params = np.array([1e-06, 0.0, 0.0, 0.0])
            pars['ewald_input_params'] = ewald_input_params
            
    
    # Loading optional extra data
    if pars['sp_emb_file']:
        with open(pars['sp_emb_file'], 'rb') as f:
            full_sp_embedding = pickle.load(f)
        pars['sp_embedding'] = np.asarray([full_sp_embedding[s] for s in pars['species']])
        pars['sp_emb_size'] = pars['sp_embedding'].shape[1]
        pars['descriptor_params']['sp_emb_size'] = str(pars['sp_emb_size'])
    else:
        pars['sp_embedding'] = None
    
    
    
    if train:
        create_traindir(filename, pars)
    return pars


#added by Gabri 
def create_traindir(inp_filename, parameters):
    if not os.path.exists(parameters['train_dir']):
        os.mkdir(parameters['train_dir'])
    if not os.path.exists(parameters['train_dir']+'/models'):
        os.mkdir(parameters['train_dir']+'/models')
    # if parameters['acc_factor']:
    #     met = metrics_LR(parameters)
    # else:
    met = metrics(parameters)
    if not os.path.exists(parameters['train_dir']+'/metrics.dat'):
        with open(parameters['train_dir']+'/metrics.dat', 'w') as f:
            f.write(met.header())
    # copy input file to traindir
    shutil.copy(inp_filename, parameters['train_dir'])
    
    