###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
from os.path import join

import numpy as np


def binary_encoder(example_dict, binary_out=None):
    """Transform an example_dict, which is an example simulation with gvector, 
    to a binary output.

    Parameters
    ----------
    example_dict: dictionary
        dictionary for a simulation with the gvector

    binary_out: string, optional
        where to save, if None save is not performed

    Returns
    -------
    binary data
    """
    int_format = {'length': 4, 'byteorder': 'little', 'signed': False}

    # init flags
    derivative_flag = '0'
    force_flag = '0'
    per_atom_quantities_flag = '0'
    sparse_derivative_flag = '0'
    long_range_el = '0'

    bin_version = 0
    n_atoms_bytes = len(example_dict['species']).to_bytes(**int_format)
    g_vect_size_bytes = len(example_dict['Gvect'][0]).to_bytes(**int_format)

    np_pkg = [
        np.asarray(example_dict['E'], dtype=np.float32).flatten(),
        np.asarray(example_dict['species'], dtype=np.float32).flatten(),
        np.asarray(example_dict['Gvect'], dtype=np.float32).flatten(),
    ]

    if ('dGvect' in example_dict) or ('dGvect_val' in example_dict):
        derivative_flag = '1'
        if 'dGvect' in example_dict:
            np_pkg.append(
                np.asarray(example_dict['dGvect'], dtype=np.float32).flatten())
        if 'dGvect_val' in example_dict:
            sparse_derivative_flag = '1'
            np_pkg.append(
                np.asarray(len(example_dict['dGvect_val']),
                           dtype=np.float32).flatten())
            np_pkg.append(
                np.asarray(example_dict['dGvect_val'],
                           dtype=np.float32).flatten())
            np_pkg.append(
                np.asarray(example_dict['dGvect_ind'],
                           dtype=np.float32).flatten())
        if 'forces' in example_dict:
            force_flag = '1'
            np_pkg.append(
                np.asarray(example_dict['forces'], dtype=np.float32).flatten())
    if 'per_atom_quantity' in example_dict:
        per_atom_quantities_flag = '1'
        np_pkg.append(
            np.asarray(example_dict['per_atom_quantity'],
                       dtype=np.float32).flatten())

     ### use el_*_kernel key to check if there are long_range terms
    if 'el_energy_kernel' in example_dict or 'el_force_kernel' in example_dict:
        long_range_el = '1'
        np_pkg.append(np.asarray(example_dict['el_energy_kernel'],
                       dtype=np.float32).flatten())
        if derivative_flag == '1':
            np_pkg.append(np.asarray(example_dict['el_force_kernel'],
                       dtype=np.float32).flatten())
        np_pkg.append(np.asarray(example_dict['atomic_charges'],
                       dtype=np.float32).flatten())
        np_pkg.append(np.asarray(example_dict['total_charge'],
                       dtype=np.float32).flatten())

    flags = '0000' + '0000' + \
            '000' + long_range_el + sparse_derivative_flag + per_atom_quantities_flag + \
            force_flag + derivative_flag

    bin_version_bytes = bin_version.to_bytes(**int_format)
    flags_bytes = int(flags, 2).to_bytes(2, byteorder='little', signed=False)

    contents = bin_version_bytes + \
        flags_bytes + \
        n_atoms_bytes + g_vect_size_bytes + \
        b"".join(np.concatenate(np_pkg, axis=0))

    if binary_out:
        with open(join(binary_out, example_dict['key'] + '.bin'),
                  'wb') as file_stream:
            file_stream.write(contents)

    return contents
