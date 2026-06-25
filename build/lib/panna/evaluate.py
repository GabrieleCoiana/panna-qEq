###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
""" Code use to evaluate a dataset on a checkpoint/s
"""
import argparse
import logging
import os
import time
import numpy as np

from panna.lib.log import emit_splash_screen
from panna.neuralnet.config_parser_train import parameter_file_parser
from panna.neuralnet.config_parser_validation import parse_validation_config
from panna.neuralnet.checkpoint import ModelFile
from panna.neuralnet.panna_model import create_panna_model
from panna.neuralnet.LongRange_electrostatic_model import create_panna_model_with_electrostatics

logger = logging.getLogger('panna')


def _filter_model_files(output_folder, model_files):
    """Filter already evaluated models."""
    model_files_not_computed = []
    for model_file in model_files:
        name = f'epoch_{model_file.epoch}_step_{model_file.step}'
        file_name = os.path.join(output_folder, f'{name}.dat')

        if os.path.isfile(file_name) and os.path.getsize(file_name) > 100:
            logger.info('%s already computed', file_name)
            continue

        model_files_not_computed.append(model_file)
    return model_files_not_computed


def _prepare_model(model_params, model_file, dummy_batch):

    if model_params.long_range_el:
        panna_model = create_panna_model_with_electrostatics(model_params)
    else:
        panna_model = create_panna_model(model_params)
    _ = panna_model(dummy_batch)
    panna_model.load_weights(model_file.file_name)
    return panna_model


class Writers():
    """Writers to file (context manager)."""
    def __init__(self, 
                 output_folder: str, 
                 model_file: ModelFile, 
                 forces: bool = False,
                 long_range_el: bool = False):
        self._output_folder = output_folder
        self._model_file = model_file
        self._forces = forces
        self._long_range_el = long_range_el

        self._file_base_name = f'epoch_{model_file.epoch}_step_{model_file.step}'

    def _open_energy_writer(self):
        file_name = os.path.join(self._output_folder, f'{self._file_base_name}.dat')
        stream = open(file_name, 'w')
        stream.write('#filename n_atoms e_ref e_nn\n')
        return stream

    def _open_force_writer(self):
        file_name = os.path.join(self._output_folder,
                                 f'{self._file_base_name}_forces.dat')
        stream = open(file_name, 'w')
        stream.write('#filename atom_id fx_nn fy_nn fz_nn fx_ref fy_ref fz_ref\n')
        return stream

    def _open_charge_writer(self):
        file_name = os.path.join(self._output_folder,
                                 f'{self._file_base_name}_charges.dat')
        stream = open(file_name, 'w')
        stream.write('#filename atom_id q_nn q_ref\n')
        return stream

    def _energy_write(self, names, ns_atoms, en_refs, en_predictions):
        for name, n_atoms, en_ref, en_prediction in zip(names, ns_atoms, en_refs,
                                                        en_predictions):
            if isinstance(name, bytes):
                name = name.decode()
            string = f'{name} {n_atoms} {en_ref} {en_prediction}\n'
            self._energy_stream.write(string)
    
    def _charge_write(self, names, n_atoms, charge_pred, charge_ref):
        # loop over examples
        for name, nat, example_pred, example_ref in zip(names, n_atoms, 
                                                   charge_pred,
                                                   charge_ref):
            if isinstance(name, bytes):
                name = name.decode()
            # loop over atoms
            example_pred = np.reshape(example_pred,[-1])
            example_ref = np.reshape(example_ref,[-1])
            for idx, (pred, ref) in enumerate(zip(example_pred[:nat],
                                                  example_ref[:nat])):
                pred = '{}'.format(pred)
                ref = '{}'.format(ref)
                string = f'{name} {idx} {pred} {ref}\n'
                self._charge_stream.write(string)


    def _force_write(self, names, n_atoms, forces_prediction, forces_ref):
        # loop over examples
        for name, nat, example_pred, example_ref in zip(names, n_atoms, 
                                                   forces_prediction,
                                                   forces_ref):
            if isinstance(name, bytes):
                name = name.decode()
            # loop over atoms
            example_pred = np.reshape(example_pred,(-1,3))
            example_ref = np.reshape(example_ref,(-1,3))
            for idx, (pred, ref) in enumerate(zip(example_pred[:nat],
                                                  example_ref[:nat])):
                pred = '{} {} {}'.format(*pred)
                ref = '{} {} {}'.format(*ref)
                string = f'{name} {idx} {pred} {ref}\n'
                self._force_stream.write(string)

    def write(self, data):
        """Write data on proper writers."""
        if self._forces:
            if self._long_range_el:
                energies, forces, charges = data[0]
            else:
                energies, forces = data[0]
        else:
            if self._long_range_el:
                energies, charges = data[0]
            else:
                energies = data[0]

        n_atoms = data[1]
        if self._long_range_el:
            batch_energies_ref, batch_forces_ref, batch_charges_ref = data[2]
        else:
            batch_energies_ref, batch_forces_ref = data[2]
        names = data[3]

        if names[0] == b'N.A.':
            logger.info('file names are not available')
            names = ['N.A.'] * energies.shape[0]

        tt = time.time()
        self._energy_write(names, n_atoms, batch_energies_ref, energies)
        if self._forces:
            self._force_write(names, n_atoms, forces, batch_forces_ref)
        if self._long_range_el:
            self._charge_write(names, n_atoms, charges, batch_charges_ref)

        tt = (time.time() - tt) * 1000
        logger.info(f'write time = {tt/1000:2.2f} s')

    def __enter__(self):
        """Open writers."""
        self._energy_stream = self._open_energy_writer()
        if self._forces:
            self._force_stream = self._open_force_writer()
        if self._long_range_el:
            self._charge_stream = self._open_charge_writer()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close writers."""
        self._energy_stream.close()
        if self._forces:
            self._force_stream.close()
        if self._long_range_el:
            self._charge_stream.close()

def run_eval(model_params, model_files, dataset, output_folder):
    """Model evaluation script."""
    force_flag = model_params.compute_forces
    long_range_el = model_params.long_range_el

    n_models = len(model_files)
    dummy_batch = next(iter(dataset))

    logger.info('----start evaluation----')
    for idx, model_file in enumerate(model_files):
        logger.info('validating network: %d/%d, epoch: %d step: %d', idx + 1, n_models,
                    model_file.epoch, model_file.step)
        model = _prepare_model(model_params, model_file, dummy_batch)
        tt = time.time()
        out = model.predict(dataset, use_multiprocessing=True)
        tt = (time.time() - tt) * 1000
        logger.info(f'eval time = {tt/1000:2.2f} s')

        with Writers(output_folder, model_file, force_flag, long_range_el) as f:
            f.write(out)


def main(*args, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        help='configuration filename',
                        required=True)
    parser.add_argument('--debug',
                        action='store_true',
                        help='debug flag, not working for now',
                        required=False)
    args = parser.parse_args()

    emit_splash_screen(logger)

    valid_config = parameter_file_parser(args.config)
    train_config_file = valid_config['IO_INFORMATION'].get('train_ini', 'train.ini')
    train_config = parameter_file_parser(train_config_file)

    model_params, model_files, dataset = parse_validation_config(
        valid_config, train_config)

    output_folder = valid_config['IO_INFORMATION'].get('eval_dir')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    model_files = _filter_model_files(output_folder, model_files)

    run_eval(model_params, model_files, dataset, output_folder)


if __name__ == '__main__':
    main()
