###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import glob
import json
import logging
import os
import struct

import numpy as np
from tensorflow.python import pywrap_tensorflow

from .scaffold_selector import scaffold_selector

logger = logging.getLogger(__name__)


def _tf_ckpt_search(seed, obj, reader):
    """ LOAD the network contained in a scaffold form a tf checkpoint

    Parameters
    ----------
    seed: str
        seed of the tree to match in the tensorflow checkpoint
    obj: obj
        An object that has tf_ckpt_elements properties and can be filled,
        with the information contained in the tree matched by the seed.
        eg. scaffolds are ok
    reader: a tf reader
        a reader

    Returns
    -------
    None

    Notes
    -----
    Passed object gets filled with the information found in the
    checkpoint.
    """
    seed_extensions, sub_objs = obj.tf_ckpt_elements
    sub_seeds = []

    for seed_extension in seed_extensions:
        if seed == '':
            new_seed = seed_extension
        else:
            new_seed = seed + '/' if seed_extension != '' else seed
            new_seed += seed_extension
        sub_seeds.append(new_seed)

    if sub_objs:
        for new_seed, new_obj in zip(sub_seeds, sub_objs):
            _tf_ckpt_search(new_seed, new_obj, reader)
    else:
        tensors = []
        for new_seed in sub_seeds:
            logger.debug('loading: %s', new_seed)
            tensors.append(reader.get_tensor(new_seed))
        obj.tf_ckpt_elements = tensors


class Checkpoint(object):
    """ A wrapper around a TF checkpoint

    Parameters
    ----------
    ckpt_file: str
       complete path to a checkpoint file eg: a/path/to/model.ckpt-0
    config_file: str
       file used to train the network
    """
    def __init__(self, in_dir, step_number, name=None):
        super().__init__()

        self._step_number = step_number
        self._ckpt_file = os.path.join(in_dir, 'model.ckpt-{}'.format(step_number))
        self._in_dir = in_dir

        with open(os.path.join(in_dir, 'networks_metadata.json')) as f:
            metadata = json.load(f)

        scaffold_type = metadata['scaffold_type']
        self._Scaffold = scaffold_selector(scaffold_type)

        self._name = name if name else self._ckpt_file

        reader = pywrap_tensorflow.NewCheckpointReader(self._ckpt_file)
        self._reader = reader

    @property
    def get_scaffold(self):
        """
        return
        ------
           The encapsulated scaffold filled with the ckpt
        """
        for name, shape in self._reader.get_variable_to_shape_map().items():
            logger.debug('available variables: %s ---- %s', name, str(shape))

        scaffold = self._Scaffold.load_dump(self._in_dir, name=str(self._step_number))
        _tf_ckpt_search('', scaffold, self._reader)
        usable = True
        for net in scaffold:
            for layer in net:
                w, b = layer.wb_value
                if w.size == 0 or b.size == 0:
                    #logger.info(f'loaded {net.name} network is not suitable '
                    #            'for validation')
                    usable = False
        if usable:
            scaffold._usable_for_validation = True
            logger.info('Scaffold ready')
        return scaffold

    def dump_PANNA_checkpoint_folder(self,
                                     folder='dump_checkpoint',
                                     gversion=1,
                                     extra_data=None):
        """Dump weights and biases of a checkpoint to a directory, also metadata

        Parameters
        ----------
        folder: str
            where to dump the files
        extra_data: dict
            extra dictionary to be added to the json,
            must be serializable

        Returns
        ------
        None

        Notes
        -----
        This method create a folder and put inside that folder
        - one folder for each species
          - one file of weights for each layer
          - one file of biases for each layer
        - a file named networks_metadata.json
          a json containing extra info
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        metadata = extra_data if extra_data else {}
        metadata.update({'gversion': gversion})

        scaffold = self.get_scaffold
        scaffold.dump_scaffold(folder, metadata)

    def dump_LAMMPS_checkpoint_folder(self,
                                      folder='network_potential',
                                      filename='panna.in',
                                      gversion=1,
                                      extra_data={}):
        """Dump weights and biases of a checkpoint to a directory, also metadata

        Args:
            dump: where to dump the files
            extra_data: extra dictionary to be added to the json
            version: 0 for the equidistant binned version, 1 for the flexible one
                     0 is compatible with OPENKIM while 1 is not

        Return:
            None

        This method creates a folder containing
        - a text file defining a NN potential of 'pair_panna' type for LAMMPS
        - one file with all weights for each species
        """
        # TODO this thing is superlegacy and must be changed.
        if not os.path.exists(folder):
            os.makedirs(folder)

        scaffold = self.get_scaffold
        extra_data['gversion'] = gversion
        scaffold.dump_scaffold_lammps(folder, filename, **extra_data)

    @property
    def filename(self):
        return self._ckpt_file.split('/')[-1]

    @property
    def step(self):
        return int(self._ckpt_file.split('model.ckpt-')[1].split('.')[0])

    @classmethod
    def laststep_finder(cls, directory):
        """ Find last checkpoint in a checkpoint directory
        """
        ckpts = Checkpoint.checkpoint_file_list(directory)
        return os.path.join(directory, ckpts[-1])

    @classmethod
    def checkpoint_file_list(cls, directory):
        """ Find all the available checkpoints in a checkpoint directory

        Args:
            directory: where to look for checkpoints

        Return:
            List of checkpoints.
        """
        ckpts = [x[:-6] for x in os.listdir(directory) if x.split('.')[-1] == 'index']

        tmp = []
        for ckp in ckpts:
            parts = glob.glob(os.path.join(directory, '{}.data*'.format(ckp)))
            if len(parts) == 0:
                logger.info('file {} failed to be loaded, no parts'.format(ckp))
                continue

            if len(parts) < int(parts[0].split('/')[-1].split('-')[4]):
                logger.info('file {} failed to be loaded, '
                            'not enough parts'.format(ckp))
                continue

            if not os.path.isfile(os.path.join(directory, '{}.meta'.format(ckp))):
                logger.info('file {} failed to be loaded, no meta'.format(ckp))
                continue
            tmp.append(ckp)
        ckpts = tmp
        ckpts.sort(key=lambda x: int(x.split('-')[-1]))
        if len(ckpts) == 0:
            raise ValueError('No ckpt available')
        return ckpts

    @classmethod
    def checkpoint_file_analizer(cls, ckpt):
        """Analyze checkpoint file.

        Args:
            ckpt: checkpoint file

        Return:
            list of steps where ckpts were created.

        Rise:
            FileNotFoundError: if file not found
            Value Error: if the checkpoint file is incoherent
        """
        ckpts = []
        try:
            with open(ckpt) as f:
                line0 = f.readline()
                line0 = line0.split('"')[1].split('-')[1]
                for x in f:
                    xt = x.split('"')[1].split('-')[1]
                    ckpts.append(int(xt))
            ckpts.sort()
            if ckpts[-1] != int(line0):
                raise ValueError('error on last line')
            return ckpts
        except FileNotFoundError as e:
            raise e

    @classmethod
    def checkpoint_step_list(cls, directory):
        """checkpoints step list
        """
        lst = Checkpoint.checkpoint_file_list(directory)
        lst = [int(x.split('-')[-1]) for x in lst]
        return lst
