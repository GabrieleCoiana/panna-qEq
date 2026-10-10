###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import json
import logging
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import pandas as pd

import keys_generator

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

#======= SCRIPT PARAMETERS =====
# units
ENERGY = 'eV'
ATOMIC_POSITIONS = 'cartesian'
UNIT_OF_LENGH = 'Angstrom'

# key to be recovered
# not case sensitive
ENERGY_KEY = 'dft_energy'
FORCE_KEY = 'dft_force'
POS_KEY = 'pos'
#====== END SCRIPT PARAMETERS =====

atom = ['', '', '', '']


def _parse_lattice_key(lattice):
    return np.asarray(lattice.split(), dtype=np.float).reshape(3, 3)


def _parse_properties_key(properties):

    properties = properties.split(':')

    columns_base_name = properties[::3]
    columns_kind = properties[1::3]
    n_elements = list(map(int, properties[2::3]))

    columns_name = []
    columns_dtype = []

    dtype_conversion = {
        'S': str,
        'R': float,
        'I': int,
    }

    for element, kind, name in zip(n_elements, columns_kind,
                                   columns_base_name):
        name = name.lower()
        if element == 1:
            columns_name.append(name)
            columns_dtype.append(dtype_conversion[kind])
        else:
            for x in range(element):
                columns_name.append(f'{name}_{x}')
                columns_dtype.append(dtype_conversion[kind])

    return columns_name, columns_dtype


def _parse_info(info):
    """ split info section in key values pair
    """
    value = []
    elements = []

    inside = False
    for char in info:
        if char == '=':
            output = ''.join(value)
            elements.append(output)
            value = []
            continue

        if char == '"':
            if inside:
                inside = False
                output = ''.join(value)
                elements.append(output)
                value = []
            else:
                inside = True
            continue

        if char == ' ' and not inside:
            output = ''.join(value)
            if not output:
                continue
            elements.append(output)
            value = []
            continue

        value.append(char)
    elements.append(''.join(value))

    for idx, element in enumerate(elements):
        if idx % 2 == 0:
            elements[idx] = element.lower()

    return dict(zip(elements[::2], elements[1::2]))


def parse_exyz(xyz_file):
    """ parse a file and return a series of snapshot

    Parameters
    ----------
    xyz_file: string containing the path to a xyz file

    Return
    ------
    List of dictionary
    each dictionary has the key/values contained in the
    extended-format
    the 'dataframe' key will contain the data frame with the table
    each element of the list is a different snapshot, if the xyz file
    has more snapshot every snap will be a entry
    """
    stream = open(xyz_file)
    snapshoots = []

    while True:
        first_line = stream.readline().strip()
        if not first_line:
            break
        n_atoms = int(first_line)

        info = stream.readline().strip()
        snapshoot = _parse_info(info)
        col_names, col_dtype = _parse_properties_key(snapshoot['properties'])

        atoms = []
        for idx in range(n_atoms):
            atom = stream.readline().strip()
            atom = [
                func(x.strip()) for x, func in zip(atom.split(), col_dtype)
            ]
            atoms.append(atom)

        snapshoot['dataframe'] = pd.DataFrame(atoms, columns=col_names)
        snapshoots.append(snapshoot)
    stream.close()
    return snapshoots


def _snapshoots_to_panna_json(outdir, snapshoots, xyz_file, addhash=False):
    for idx, snapshoot in enumerate(snapshoots):
        panna_json = dict()
        # general info
        panna_json['name'] = os.path.splitext(os.path.basename(xyz_file))[0]
        panna_json['step'] = idx

        panna_json['source'] = os.path.abspath(xyz_file)
        panna_json['lattice_vectors'] = _parse_lattice_key(
            snapshoot['lattice']).tolist()

        # TODO ADD A WARNING
        panna_json['atomic_position_unit'] = ATOMIC_POSITIONS
        panna_json['unit_of_length'] = UNIT_OF_LENGH

        if ENERGY_KEY:
            panna_json['energy'] = (float(snapshoot[ENERGY_KEY]), ENERGY)

        panna_json['atoms'] = []

        for r_idx, row in snapshoot['dataframe'].iterrows():
            pos = row[[*[POS_KEY + f'_{x}' for x in range(3)]]].values
            atom = [r_idx, row.species, pos.tolist()]
            if FORCE_KEY:
                forces = row[[*[FORCE_KEY + f'_{x}' for x in range(3)]]].values
                atom.append(forces.tolist())

            panna_json['atoms'].append(atom)

            # Extended XYZ format does not seem to have place for a unique comment etc.
            # So PANNA adds a unique key to it.
            panna_json['key'] = 'KeyByPANNA-' + keys_generator.hash_key_v2(
                panna_json)

            if addhash:
                panna_json_name = keys_generator.hash_key(panna_json)
            else:
                panna_json_name = '{}_{}'.format(
                    idx, panna_json['name'].split('.xyz')[0])

            with open(
                    outdir.rstrip('/') + "/" + panna_json_name + ".example",
                    'w') as outfile:
                json.dump(panna_json, outfile)


def _parallel_process_support(xyz_file, outdir, addhash):
    logger.info('working on: %s', xyz_file)
    # recover snapshoots within the file
    snapshoots = parse_exyz(xyz_file)
    # convert
    _snapshoots_to_panna_json(outdir, snapshoots, xyz_file, addhash)


def main(indir, outdir, addhash, nproc):

    if not os.path.isdir(outdir):
        logger.info("outdir not found - making outdir")
        os.makedirs(outdir)

    outdir = os.path.abspath(outdir)

    logger.info('outdir: %s', outdir)
    logger.info('==== PARAMETERS ====')
    logger.info('energy: %s', ENERGY)
    logger.info('atomic positions: %s', ATOMIC_POSITIONS)
    logger.info('unit of length: %s', UNIT_OF_LENGH)
    logger.info('energy key: %s', ENERGY_KEY)
    logger.info('force key: %s', FORCE_KEY)
    logger.info('====================')

    p = mp.Pool(nproc)

    # find files
    xyzfiles = []
    for rt, dirs, files in os.walk(indir):
        for f in files:
            logger.debug('found: %s', f)
            if f.endswith('xyz'):
                xyzfiles.append(os.path.join(rt, f))

    logger.info('num of files found: %d', len(xyzfiles))

    converter = partial(_parallel_process_support,
                        outdir=outdir,
                        addhash=addhash)

    i = 0
    while i <= int(len(xyzfiles) / nproc):
        try:
            fi = xyzfiles[nproc * i:nproc * (i + 1)]
        except IndexError:
            fi = xyzfiles[nproc * i:len(xyzfiles)]
        data = p.map(converter, fi)
        i += 1


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Extended XYZ to PANNA json converter')
    PARSER.add_argument(
        '-i',
        '--indir',
        type=str,
        help=
        'input directory that holds all the xyz files in any subdir structure',
        required=True)
    PARSER.add_argument('-o',
                        '--outdir',
                        type=str,
                        help='output directory',
                        required=True)
    PARSER.add_argument('--addhash',
                        type=bool,
                        required=False,
                        help='use hash to name jsons',
                        default=False)
    PARSER.add_argument('--nproc',
                        type=int,
                        help='num threads',
                        required=False,
                        default=1)
    ARGS = PARSER.parse_args()

    main(indir=ARGS.indir,
         outdir=ARGS.outdir,
         addhash=ARGS.addhash,
         nproc=ARGS.nproc)
