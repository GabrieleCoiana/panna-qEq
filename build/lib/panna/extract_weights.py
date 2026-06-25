###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""
  code used to extract weights from a checkpoint
"""
import argparse
import configparser
import logging
import os
import shutil
import glob
import numpy as np

from panna.neuralnet.panna_model import create_panna_model
from panna.neuralnet.LongRange_electrostatic_model import create_panna_model_with_electrostatics
from panna.neuralnet.config_parser_train import parse_config, parameter_file_parser
from panna.gvect_calculator import extract_gvect_func
from panna.lib.log import emit_splash_screen
from panna.lib.parser_callable import converters

logger = logging.getLogger('panna')


def recover_gvect(gvect_config):
    config = configparser.ConfigParser(converters=converters)
    config.read(gvect_config)
    gvect_func = extract_gvect_func(config)
    return gvect_func, config


def _main(conf_file):
    config = configparser.ConfigParser()
    config.read(conf_file)
    folder_info = config['IO_INFORMATION']
    model_dir = folder_info.get('network_dir', None)
    step_number = folder_info.getint('step_number', -1)
    # subsample = folder_info.getint('subsample', -1)
    out_dir = folder_info.get('output_dir', './saved_network')

    train_input = folder_info.get('train_input', None)
    gvector_input = folder_info.get('gvector_input', None)

    output_type = folder_info.get('output_type', 'LAMMPS')
    output_file = folder_info.get('output_file', 'panna.in')
    long_range_el = folder_info.getboolean('long_range_el', False)
    # 0 is for OPENKIM compatible version
    gversion = folder_info.getint('gvector_version', 1)
    extra_data = {}
    extra_data['gversion'] = gversion

    if gvector_input:
        gvect_func, gvect_config = recover_gvect(gvector_input)
        if gversion == 1:
            extra_data['gvect_params'] = gvect_func.gvect
        else:
            extra_data['gvect_params'] = gvect_func.gvect_v0

    elif output_type=='LAMMPS' or output_type=='ASE':
        logger.info('{} dump requires gvect_params input file'.format(output_type))
        exit(1)

    if step_number==-1:
        chkp = model_dir+'/checkpoint'
        if os.path.exists(chkp):
            with open(chkp) as f:
                lines = f.readlines()
                restart_file = lines[0].strip().split()[1].replace("\"", '')
                restart_file = model_dir+'/'+restart_file
        else:
            raise ValueError('Cannot find checkpoint file')
    else:
        files = glob.glob(model_dir+f'/epoch_*{step_number}.index')
        if len(files)>0:
            restart_file = files[0].split('.index')[0]
        else:
            raise ValueError(\
                f'Cannot find model file {model_dir}/epoch_*{step_number}')

    logger.info('Will load weights from file %s', restart_file)

    # Loading parameters from training file and constructing+compiling model
    parameters = parameter_file_parser(train_input)
    
    # For ASE we just need to take the relevant info from the input file
    # and copy the weights file, since we will load it as is
    if output_type=='ASE':
        ASEconfig = configparser.ConfigParser()
        ASEconfig['DATA_INFORMATION'] = parameters['DATA_INFORMATION']
        ASEconfig.add_section('IO_INFORMATION')
        ASEconfig.set('IO_INFORMATION', 'input_format', 'example')
        ASEconfig.set('IO_INFORMATION', 'network_file', out_dir+'/'+restart_file.split(model_dir)[1])
        ASEconfig['TRAINING_PARAMETERS'] = parameters['TRAINING_PARAMETERS']
        ASEconfig['DEFAULT_NETWORK'] = parameters['DEFAULT_NETWORK']
        for c in parameters['DATA_INFORMATION'].get_comma_list('atomic_sequence', []):
            if parameters.has_section(c):
                ASEconfig[c] = parameters[c]
        ASEconfig['SYMMETRY_FUNCTION'] = gvect_config['SYMMETRY_FUNCTION']
        ASEconfig['GVECT_PARAMETERS'] = gvect_config['GVECT_PARAMETERS']
        try:
            ASEconfig['LONG_RANGE'] = gvect_config['LONG_RANGE']
        except:
            pass
        with open(output_file, 'w') as f:
            ASEconfig.write(f)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for f in glob.glob(restart_file+'*'):
            shutil.copyfile(f,out_dir+'/'+f.split(model_dir)[1])
        exit(0)

    
    # Disabling force training since we're just loading for weights
    parameters.set('TRAINING_PARAMETERS', 'forces_cost', '0.0')
    parameters.set('TRAINING_PARAMETERS', 'charge_cost', '0.0')
    # Setting restart mode to None because we do it by hand
    parameters.set('IO_INFORMATION', 'restart_mode', 'no')
    model_params, compile_params, fit_params, validation_params = \
        parse_config(parameters)    

    if long_range_el:
        panna_model = create_panna_model_with_electrostatics(model_params)
    else:
        panna_model = create_panna_model(model_params)
    panna_model.load_weights(restart_file)
    gsize = parameters['DEFAULT_NETWORK'].getint('g_size', None)
    # We have to call the model once to initialize
    if long_range_el:
        dummy_input = [[np.zeros((1,1),dtype=np.int32),np.zeros((1,1,gsize))], 
                       [np.zeros((1,1))+1e-6, np.zeros(1), np.zeros((1,1,3))]]
    else:
        dummy_input = [np.zeros((1,1),dtype=np.int32),np.zeros((1,1,gsize))]
    _ = panna_model(dummy_input)
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    if output_type == 'LAMMPS':
        panna_model.dump_network_lammps(out_dir, output_file, **extra_data)
    elif output_type == 'PANNA':
        panna_model.dump_network_panna(out_dir, output_file, **extra_data)
    else:
        raise ValueError('Unknown dump type')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file', required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug flag',
                        required=False)
    args = parser.parse_args()
    emit_splash_screen(logger)

    _main(args.config)


if __name__ == '__main__':
    main()
