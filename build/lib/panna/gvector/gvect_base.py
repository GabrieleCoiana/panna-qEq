###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

from abc import ABC, abstractmethod
from copy import deepcopy
import json
import os

import numpy as np

from panna.lib.units_converter import distances_to_angstrom


def requires_parameters(method):
    """A decorator for parameters to be loaded."""
    def check_parameters(self, *args, **kwargs):
        for parameter in self._gvect_parameters:  # pylint: disable=protected-access
            if self.__getattr__(parameter) is None:
                raise ValueError(f'Parameter {parameter} is not set')
            if not self.units:
                raise ValueError(
                    f'Internal units have not been specified'
                    'you might need to fix {self.__class__.__name__}.units'
                    '= Angstrom')
        return method(self, *args, **kwargs)

    return check_parameters


class GvectBase(ABC):
    """Base class for descriptors

    Parameters
    ----------
    species : string
        sequence of species eg: C,H,N,O
    compute_dgvect : Boolean, optional
        (the default value is False)
    sparse_dgvect : Boolean, optional
        if the dgvector are returned as sparse matrix or dense matrix
        This flag is available only if the dgvector are computed.
        (the default value is False)
    pbc_directions: list of Boolean, optional
        pbc directions to override those in the json

    Notes
    -----
    self.units default is set to Angstrom 
    """
    # parameter, setter function
    _gvect_parameters = {}

    def __init__(self,
                 species,
                 compute_dgvect=False,
                 sparse_dgvect=False,
                 pbc_directions=None):

        # ==== generic parameters ====
        self.compute_dgvect = compute_dgvect
        self.sparse_dgvect = sparse_dgvect
        self.species = species
        self.pbc_directions = pbc_directions
        self.units = 'angstrom'
        self._my_parameters = {}

    @property
    def descriptor_parameters(self):
        """ return a string with all the descriptor parameters
        """
        return ", ".join(self._gvect_parameters.keys())

    @property
    def parameters(self):
        return deepcopy(self._my_parameters)

    @property
    def species_idx_2str(self):
        return [x.strip() for x in self.species.split(',')]

    @property
    def unit2A(self):
        return distances_to_angstrom(self.units)

    @property
    def number_of_species(self):
        return len(self.species_idx_2str)

    @property
    def species_str_2idx(self):
        return dict(zip(self.species_idx_2str, range(self.number_of_species)))

    @property
    @requires_parameters
    def gsize(self):
        """ return the size of the descriptor for one atom
        """

    @abstractmethod
    def __call__(self, *args):
        """ This call must compute the gvectors
        """

    @abstractmethod
    def parse_parameters(self, gv_param):
        """ parser to recover parameters from configuration file
        """

    def parse_panna_dump(self, path):
        """ parser to recover parameters from a panna dump folder
        """
        with open(os.path.join(path, 'networks_metadata.json')) as fs:
            data = json.load(fs)
        gv_params = data.get('gvect_params', None)
        if not gv_params:
            raise ValueError('dump does not contain information about the g vector params')
        self.units = 'angstrom'
        for key, value in gv_params.items():
            try:
                self.update_parameter(key, value)
            except ValueError as err:
                print(err)

    def update_parameter(self, parameter, value):
        """ update all the internal parameters and apply the setter
        """
        if parameter not in self._gvect_parameters.keys():
            raise ValueError(f'parameter {parameter} cannot be set')
        self._my_parameters[parameter] = self._gvect_parameters[parameter](
            self, np.array(value))

    def __getattr__(self, name):
        if name.startswith('__'):
            # No attribute should start with __ because it creates an infinite recursive copy 
            # stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
            raise AttributeError()
        if name in self._my_parameters:
            return self._my_parameters.get(name, None)
        raise AttributeError(f'{self.__class__.__name__}.{name} is invalid.')
