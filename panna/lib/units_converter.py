###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""This module contains all units conversion
"""

import logging

import numpy as np

# Conversion constants
BOHR_2_A = np.float32(0.52917721067)
RY_2_EV = np.float32(13.6056980659)

# logger
logger = logging.getLogger(__name__)


def energy_to_ev(energy_units):
    """ Factor to convert energy units to eV

    Parameters
    ----------
    energy_units: string with current energy units

    Returns
    -------
    float: factor to perform conversions
    """

    if energy_units in ["Ry", "Ryd", "ryd", "ry", "RY"]:
        unit_to_ev = np.float32(1) * RY_2_EV
    elif energy_units == "Ha":
        unit_to_ev = np.float32(2) * RY_2_EV
    elif energy_units in ["eV", "ev", "EV"]:
        unit_to_ev = np.float32(1)
    else:
        raise ValueError('energy units unrecognized')

    return unit_to_ev


def distances_to_angstrom(distance_units):
    """ Factor to convert distance units to angstrom

    Parameters
    ----------
    distance_units: string with current distance units

    Returns
    -------
    float: factor to perform conversions
    """

    if distance_units in ['au', 'bohr', 'bohr_radius', 'BOHR']:
        unit_to_angstrom = np.float32(1) * BOHR_2_A
    elif distance_units in ['a', 'angstrom', 'A', 'Angstrom', 'ANGSTROM']:
        unit_to_angstrom = np.float32(1)
    else:
        raise ValueError('distance units unrecognized')
    return unit_to_angstrom
