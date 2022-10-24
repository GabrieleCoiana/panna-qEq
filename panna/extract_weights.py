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
import os
import logging
import argparse
import configparser

from tensorflow.errors import NotFoundError

import neuralnet as net
from panna.gvect_calculator import extract_gvect_func
from lib import init_logging
from lib.parser_callable import converters

# logger
logger = logging.getLogger('panna')  # pylint: disable=invalid-name


def recover_gvect(gvect_config):
    config = configparser.ConfigParser(converters=converters)
    config.read(gvect_config)
    gvect_func = extract_gvect_func(config)
    return gvect_func


def main(conf_file):
    config = configparser.ConfigParser()
    config.read(conf_file)
    folder_info = config['IO_INFORMATION']
    in_dir = folder_info.get('train_dir', None)
    step_number = folder_info.getint('step_number', 0)
    out_dir = folder_info.get('output_dir', './saved_network')

    gvector_input = folder_info.get('gvector_input', None)

    output_type = folder_info.get('output_type', 'PANNA')
    output_file = folder_info.get('output_file', 'panna.in')

    extra_data = {}

    json_file = os.path.join(in_dir, 'networks_metadata.json')

    if gvector_input:
        gvect_func = recover_gvect(gvector_input)
        extra_data['gvect_params'] = gvect_func.gvect
    else:
        logger.info('missing gvect_params input file')
        exit(1)

    ck_steps = net.Checkpoint.checkpoint_step_list(in_dir)

    if step_number in ck_steps:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        ckpt = net.Checkpoint(
            os.path.join(in_dir, 'model.ckpt-{}'.format(step_number)),
            json_file)
        try:
            if output_type == 'PANNA':
                ckpt.dump_PANNA_checkpoint_folder(folder=out_dir,
                                                  extra_data=extra_data)
            elif output_type == 'LAMMPS':
                ckpt.dump_LAMMPS_checkpoint_folder(folder=out_dir,
                                                   filename=output_file,
                                                   extra_data=extra_data)
            else:
                logger.info('Unknown output type %s', output_type)
        except NotFoundError:
            logger.warning('%s not found because of TF bug', ckpt.filename)
        logger.info('Weights saved in %s', out_dir)
    else:
        logger.info('step number %d not found', step_number)


if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-c',
                        '--config',
                        type=str,
                        help='config file',
                        required=True)
    PARSER.add_argument('--debug',
                        action='store_true',
                        help='debug flag',
                        required=False)
    ARGS = PARSER.parse_args()

    main(ARGS.config)
