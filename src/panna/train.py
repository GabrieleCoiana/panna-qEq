###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import logging

from panna.lib.log import emit_splash_screen
from panna.neuralnet.panna_model import create_panna_model
from panna.neuralnet.LongRange_electrostatic_model import create_panna_model_with_electrostatics
from panna.neuralnet.config_parser_train import parse_config, parameter_file_parser

logger = logging.getLogger('panna')


def train(panna_model, train_data, max_training, callbacks, validation_params,
          initial_epoch):
    """Fit wrapper."""
    max_epochs, steps_per_epoch = max_training
    panna_model.fit(train_data,
                    epochs=max_epochs,
                    steps_per_epoch=steps_per_epoch,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    validation_data=validation_params.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default='',
                        help="config file",
                        required=True)
    parser.add_argument("--debug",
                        action='store_true',
                        help='enable debug mode, partial support only')
    flags, _unparsed = parser.parse_known_args()

    emit_splash_screen(logger)
    parameters = parameter_file_parser(flags.config)

    # I am open to suggestions when it comes to how to handle all these
    # values because this is a little bit messy.
    model_params, compile_params, fit_params, validation_params = \
        parse_config(parameters)

    long_range_el = model_params.long_range_el

    if long_range_el:
        panna_model = create_panna_model_with_electrostatics(model_params, validation_params)
    else:
        panna_model = create_panna_model(model_params, validation_params)

    initial_epoch = 0
    if fit_params.restart_info is not None:
        if 'step' in fit_params.restart_info:
            logger.info('Restarting from ckpt epoch %s, step %s',
                fit_params.restart_info['epoch'], fit_params.restart_info['step'])
        else:
            logger.info('Restarting from ckpt %s', fit_params.restart_info['file_name'])
        # Restart form checkpoint required. Loading MUST be done
        # before compilation to allow new values of e_loss, f_loss
        # and optimizer to be loaded.
        # batch = next(iter(fit_params.dataset))
        # first we need to build the model, model.build(input_shape)
        # at the moment does not work on ragged tensor.
        # TODO override it to make it work.
        # We will call the model on a real batch.
        # _ = panna_model(batch)
        panna_model.load_weights(fit_params.restart_info['file_name'])
        initial_epoch = fit_params.restart_info['epoch']
    if long_range_el:
        panna_model.compile(e_loss=compile_params.e_loss,
                        f_loss=compile_params.f_loss,
                        q_loss=compile_params.q_loss,
                        f_cost=compile_params.f_cost,
                        q_cost=compile_params.q_cost,
                        energy_example_weight=compile_params.energy_example_weight,
                        force_example_weight=compile_params.force_example_weight,
                        charge_example_weight=compile_params.charge_example_weight,
                        optimizer=compile_params.opt)
    else:
        panna_model.compile(e_loss=compile_params.e_loss,
                        f_loss=compile_params.f_loss,
                        f_cost=compile_params.f_cost,
                        energy_example_weight=compile_params.energy_example_weight,
                        force_example_weight=compile_params.force_example_weight,
                        optimizer=compile_params.opt)

    train(
        panna_model,
        fit_params.dataset,
        fit_params.max_training,
        fit_params.io_callbacks,
        validation_params=validation_params,
        initial_epoch=initial_epoch,
    )


if __name__ == '__main__':
    main()
