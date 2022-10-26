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

from tensorflow.errors import NotFoundError

if __package__ != 'panna':
    import neuralnet as net
    from gvect_calculator import extract_gvect_func
    from lib.log import emit_splash_screen, init_logging
    from lib.parser_callable import converters
else:
    from . import neuralnet as net
    from .gvect_calculator import extract_gvect_func
    from .lib.log import emit_splash_screen, init_logging
    from .lib.parser_callable import converters


logger = logging.getLogger('panna')


def recover_gvect(gvect_config):
    config = configparser.ConfigParser(converters=converters)
    config.read(gvect_config)
    gvect_func = extract_gvect_func(config)
    return gvect_func


def _main(conf_file):
    config = configparser.ConfigParser()
    config.read(conf_file)
    folder_info = config['IO_INFORMATION']
    in_dir = folder_info.get('train_dir', None)
    step_number = folder_info.getint('step_number', -1)
    subsample = folder_info.getint('subsample', -1)
    out_dir = folder_info.get('output_dir', './saved_network')

    gvector_input = folder_info.get('gvector_input', None)

    output_type = folder_info.get('output_type', 'PANNA')
    output_file = folder_info.get('output_file', 'panna.in')
    # 0 is for OPENKIM compatible version
    gversion = folder_info.get('gvector_version', 1)
    extra_data = {}

    if gvector_input:
        gvect_func = recover_gvect(gvector_input)
        if int(gversion) == 1:
            extra_data['gvect_params'] = gvect_func.gvect
        else:
            extra_data['gvect_params'] = gvect_func.gvect_v0

    else:
        logger.info('missing gvect_params input file')
        exit(1)

    ck_steps = net.Checkpoint.checkpoint_step_list(in_dir)

    if step_number == -1 and subsample == -1:
        step_numbers = [ck_steps[-1]]
    if step_number >= 0 and subsample == -1:
        step_numbers = [step_number]

    if subsample > 0:
        step_numbers = ck_steps[::subsample]
    folder = out_dir
    for step_number in step_numbers:
        if step_number in ck_steps:
            if len(step_numbers) > 1:
                folder = os.path.join(out_dir, str(step_number))
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            ckpt = net.Checkpoint(in_dir, step_number)
            try:
                if output_type == 'PANNA':
                    ckpt.dump_PANNA_checkpoint_folder(folder=folder,
                                                      gversion=gversion,
                                                      extra_data=extra_data)
                elif output_type == 'LAMMPS':
                    ckpt.dump_LAMMPS_checkpoint_folder(folder=folder,
                                                       filename=output_file,
                                                       gversion=gversion,
                                                       extra_data=extra_data)
                else:
                    logger.info('Unknown output type %s', output_type)
            except NotFoundError:
                logger.warning('%s not found because of TF bug', ckpt.filename)
            logger.info('Weights saved in %s', out_dir)
        else:
            logger.info('step number %d not found', step_number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file', required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug flag',
                        required=False)
    args = parser.parse_args()
    init_logging()
    emit_splash_screen(logger)

    _main(args.config)


if __name__ == '__main__':
    main()
