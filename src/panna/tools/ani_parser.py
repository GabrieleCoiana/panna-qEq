###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
# expander for ani data

import argparse
import json
import os
from os.path import join

import h5py

from keys_generator import hash_key_v2

# constants
HA2EV = 27.2113966413079

class AniWrapper():

    def __init__(self, file_name):
        self._file_name = file_name
        self._wrapped = h5py.File(file_name, 'r')

    def __iter__(self):
        for folder in self._wrapped.values():
            dt = dict()
            dt['parent'] = folder.name
            for molecule_group, molecule_info in folder.items():
                dt['group'] = molecule_group
                dt['smile'] = ''.join([x.decode() for x in molecule_info['smiles'][()].tolist()])
                dt['species'] = ''.join([x.decode() for x in molecule_info['species'][()].tolist()])
                # all the config for a given molecule
                cords = molecule_info['coordinates']
                # all the energies
                es = molecule_info['energies']
                for coord, e in zip(cords, es):
                    dt['coordinates'] = coord
                    dt['energy'] = e
                    yield dt

def main(args):
    print('begin')
    # Set the HDF5 file containing the data
    hdf5file = args.hdf5file
    # Set output folder
    output_folder = args.output_folder
    # not a good idea to have verbose mode on big files
    verbose = args.verbose
    if verbose:
        print("verbose has been deprecated")

    # Construct the data loader class
    ani_wrapper = AniWrapper(hdf5file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for config in ani_wrapper:
        species = config['species']
        molecule = config['smile']

        if not os.path.exists(join(output_folder, molecule)):
            os.makedirs(join(output_folder, molecule))

        if not os.path.exists(join(output_folder, molecule, 'examples')):
            os.makedirs(join(output_folder, molecule, 'examples'))

        # cycle over all the available configurations of a molecule
        cords = config['coordinates'].tolist()
        energy = config['energy']
        sim = {}
        for i, (cord, kind) in enumerate(zip(cords, species)):
            cs = sim.get('atoms', [])
            cs.append((i + 1,
                       kind,
                       cord,
                       (0, 0, 0)
                       ))
            sim['atoms'] = cs
        sim['energy'] = (energy * HA2EV, 'ev')
        sim['lattice_vectors'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        sim['atomic_position_unit'] = 'cartesian'
        sim['unit_of_length'] = 'angstrom'
        sim['name'] = molecule
        key = hash_key_v2(sim)

        with open(os.path.join(output_folder, molecule, 'examples',
                               key + '.example'), 'w') as f:
            json.dump(sim, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANI data expander ')
    parser.add_argument('-in', '--hdf5file', type=str,
                        help='file to decompress', required=True)
    parser.add_argument('-out', '--output_folder', type=str,
                        help='output main folder', required=True)
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='verbose mode', required=False)
    args = parser.parse_args()
    main(args)
