###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import json
import os
import shutil as st

import numpy as np
import pandas as pd

ry2ev = 13.6056980659
b2a = 0.529177


def unpack_bin(single_file):

    with open(single_file, "rb") as binary_file:
        bin_version = int.from_bytes(binary_file.read(4),
                                     byteorder='little',
                                     signed=False)
        if bin_version != 0:
            raise NotImplementedError()
        # converting to int to avoid handling little/big endian
        flags = int.from_bytes(binary_file.read(2),
                               byteorder='little',
                               signed=False)

        derivative_flag = flags & 0b00000001
        force_flag = flags & 0b00000010
        per_atom_quantities_flag = flags & 0b00000100
        sparse_derivative_flag = flags & 0b00001000
        long_range_flag = flags & 0b00010000
        n_atoms = int.from_bytes(binary_file.read(4),
                                 byteorder='little',
                                 signed=False)
        g_size = int.from_bytes(binary_file.read(4),
                                byteorder='little',
                                signed=False)
        payload = binary_file.read()
    # assuming machine that created the binary file is little endian
    data = np.frombuffer(payload, dtype='<f4')

    energy = np.reshape(data[0], [1])[0]

    spec_tensor_bytes = n_atoms
    gvect_tensor_bytes = n_atoms * g_size
    prev_bytes = 1

    spec_tensor = data[prev_bytes:prev_bytes + spec_tensor_bytes]
    spec_tensor = np.int64(np.reshape(spec_tensor, [n_atoms]))

    prev_bytes += spec_tensor_bytes
    gvect_tensor = np.reshape(
        data[prev_bytes:prev_bytes + gvect_tensor_bytes], [n_atoms, g_size])

    # choose a random environment
    #N_rad = int(np.random.random()*n_atoms)
    #rad_env = gvect_tensor[N_rad]

    return n_atoms, g_size, energy, gvect_tensor


def extract_gvectors(files):

    grand_Natoms = 0
    grand_gvects = []
    for file in files:
        n_atoms, gsize, energy, gvector_tensor = unpack_bin(file)
        grand_gvects = np.append(
            grand_gvects, gvector_tensor.flatten(), axis=0)
        grand_Natoms += n_atoms
    grand_gvects = np.reshape(np.asarray(grand_gvects), [-1, gsize])

    return gsize, grand_Natoms, grand_gvects


def main(source, outpath, restart=False, restart_file=None):
    '''
        restart to enable the extension of the average:
        ingredients to be read from file:
                   old_average
                   old_sigma
                   old_n_config
        input:
             source: binary path
             outpath: outpath to write the quantities
    '''

    if not os.path.exists(outpath):
        os.mkdir(outpath)
    files = [os.path.join(source, x) for x in os.listdir(source)]
    n_atoms, gsize, energy, gvector_tensor = unpack_bin(files[0])
    # zero of sigma_G**2
    epsilon = 1e-12

    data = dict()
    average_G = np.zeros(gsize, dtype=np.float64)
    sigma_G = np.zeros(gsize, dtype=np.float64)
    nconfig = 0
    if restart:
        if restart_file:
            data = json.load(open(restart_file))
            nconfig = int(data['nconfig'])
            average_G = np.asarray(data['average_G'], dtype=np.float64)
            sigma_G = np.asarray(data['sigma_G'], dtype=np.float64)**2
            sigma_G += (average_G**2 - epsilon)
            average_G *= nconfig
            sigma_G *= nconfig

    grand_Natoms = 0
    nconfig += len(files)

    all_gvectors = []
    for file in files:
        n_atoms, gsize, energy, gvector_tensor = unpack_bin(file)
        all_gvectors = np.append(
            all_gvectors, gvector_tensor.flatten(), axis=0)
        _mini_mean = np.mean(np.asarray(
            gvector_tensor, dtype=np.float64), axis=0)
        average_G += _mini_mean
        _mini_mean_sq = np.mean(np.asarray(
            gvector_tensor, dtype=np.float64)**2, axis=0)
        sigma_G += _mini_mean_sq
        grand_Natoms += n_atoms

    all_gvectors = np.reshape(np.asarray(all_gvectors), [-1, gsize])
    average_G /= nconfig
    sigma_G /= nconfig
    sigma_G -= average_G**2
    # treat the zero components: such that small variance components contribute epsilon to the
    # variance
    sigma_G = np.sqrt(sigma_G + epsilon)
    # alternative averaging
#    gsize, grand_Natoms, all_gvectors = extract_gvectors(files)
#    average_G, sigma_G = np.mean(all_gvectors, axis=0), np.std(all_gvectors, axis=0)

    data["average_G"] = average_G.tolist()
    data["sigma_G"] = sigma_G.tolist()
    data["nconfig"] = nconfig

    with open(os.path.join(outpath, 'average_G_sigma_G.json'), 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='color cluster')
    parser.add_argument('-s', '--source', type=str,
                        help='source file or folder', required=True)
    parser.add_argument('-o', '--outpath', type=str,
                        help='outpath', required=True)
    parser.add_argument('-r', '--restart', type=bool,
                        help='whether of restart', required=False)
    parser.add_argument('-rf', '--restart_file', type=str,
                        help='restart filename', required=False)

    args = parser.parse_args()
    main(args.source, args.outpath, args.restart, args.restart_file)
