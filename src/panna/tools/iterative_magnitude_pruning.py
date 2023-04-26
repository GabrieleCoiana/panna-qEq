###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import sys
import argparse
import configparser
import numpy as np
import copy
# from panna import train

def main(configfile, **kvargs):
    '''
    Creates all folders and input files needed for IMP
    and starts the training in succession.
    Works with basic configuration (see below for keys).
    Modify from this for more advanced setups.
    '''
    config = configparser.ConfigParser()
    config.read(configfile)

    IMP_params = config['IMP']
    base_input = IMP_params.get('base_input', 'train.ini')
    prunable = IMP_params.get('prunable', 'train.ini')
    ratio = IMP_params.getfloat('ratio', 0.3)
    N_iter = IMP_params.getint('N_iter', 2)
    start_step = IMP_params.get('start_step', '1')
    weights_step = IMP_params.get('weights_step', '1')

    original_config = configparser.ConfigParser()
    original_config.read(base_input)
    species = original_config['DATA_INFORMATION'].get('atomic_sequence', '')
    species = [s.strip() for s in species.split(',')]
    for i in range(N_iter):
        iterpath = 'IMP'+str(i)
        if os.path.exists(iterpath):
            print('Folder {} already exists. Stopping.')
            exit()
        os.mkdir(iterpath)
        conf = copy.deepcopy(original_config)
        if i==0:
            conf.set('IO_INFORMATION', 'starting_model_save', start_step)
        else:
            conf.set('IO_INFORMATION', 'restart_mode', 'specific_ckpt')
            conf.set('IO_INFORMATION', 'restart_file', '../IMP0/starting_model/variables/variables')
        conf.set('IO_INFORMATION', 'final_model_save', weights_step)
        for s in species:
            if not conf.has_section(s):
                conf.add_section(s)
            conf.set(s, 'prunable', prunable)
            conf.set(s, 'sparsity', str(1.-(1.-ratio)**(i)))
            if i>0:
                conf.set(s, 'starting_masks', '../IMP'+str(i-1)+'/final_model/')
                conf.set(s, 'pruning_weights', '../IMP'+str(i-1)+'/final_model/')
            with open(iterpath+'/train.ini', 'w') as f:
                conf.write(f)

    print('Inputs created. Starting training.')
    orig_dir = os.getcwd()
    for i in range(N_iter):
        os.chdir(orig_dir+'/IMP'+str(i))
        os.system('panna_train -c train.ini')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gvector plain text converter')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='configuration file',
        required=True)
    args = parser.parse_args()
    main(args.config)
