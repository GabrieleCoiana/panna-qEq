###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
""" convert a lammps dump to json format, not granted to work with any dump
at the moment
assumptions : units are metal
for a list of to-do read the main
tested dump:
  dump name all custom step file_name  id type x y z fx fy fz
  dump_modify name append yes
"""
import os
import json
import hashlib
import logging
import argparse
from io import StringIO

import numpy as np
import pandas as pd

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# helper to parse pos file
def _number_of_atoms(parameters, lines):
    """ parse number of atom section token
    """
    data = [','.join(parameters)]
    for line in lines:
        data.append(','.join(line.split()))
    dataframe = pd.read_csv(StringIO('\n'.join(data)))
    return dataframe


def _box_bounds(parameters, lines):
    """parse box bounds token"""
    lines = [line.split() for line in lines]
    if parameters[:3] != ['xy', 'xz', 'yz']:
        raise ValueError('Box bounds not recognized')
    periodicity = [x == 'pp' for x in parameters[-3:]]
    return np.asarray(lines, dtype=np.float32), periodicity


_RESERVED_WORDS = {
    # first element = number of row
    # second element = parsin function for the data
    'TIMESTEP': [1, lambda x, y: int(y[0])],
    'NUMBER OF ATOMS': [1, lambda x, y: int(y[0])],
    'BOX BOUNDS': [3, _box_bounds],
    'ATOMS': ['NUMBER OF ATOMS', _number_of_atoms]
}

# end of helpers for pos files


# other helpers
def _recover_energy(lammps_run, time_step):
    detected_energies = []
    for data_frame in lammps_run:
        # TODO, this is not very nice... one should detect the the column
        # based on type of calculation
        if 'E_pair' in data_frame:
            energy = data_frame[data_frame.Step == time_step].E_pair.values
        else:
            energy = data_frame[data_frame.Step == time_step].PotEng.values
        detected_energies.extend(energy)
    if len(detected_energies) > 1:
        if not (np.asarray(detected_energies) == detected_energies[0]).all():
            raise ValueError('undefined behavior')
    return detected_energies[0]


# other helpers


def convert_box_bounds_to_lattice(lammps_lattice):  # pylint: disable=too-many-locals
    """ convert box bounds in from lamps format to lattice

    Parameters
    ----------
    lammps_lattice: 3,3 numpy array
                    xlo_bound, xhi_bound, xy
                    ylo_bound, yhi_bound, xz
                    zlo_bound, zhi_bound, yz
    Returns
    -------
    lattice vectors: 3,3 numpy array
    """
    # pylint: disable=invalid-name
    xlo_bound, xhi_bound, xy = lammps_lattice[0]
    ylo_bound, yhi_bound, xz = lammps_lattice[1]
    zlo_bound, zhi_bound, yz = lammps_lattice[2]

    xl = xlo_bound - np.min([0.0, xy, xz, xy + xz])
    xh = xhi_bound - np.max([0.0, xy, xz, xy + xz])
    yl = ylo_bound - np.min([0.0, yz])
    yh = yhi_bound - np.max([0.0, yz])
    zl = zlo_bound
    zh = zhi_bound

    a1 = [xh - xl, 0., 0.]
    a2 = [xy, yh - yl, 0.0]
    a3 = [xz, yz, zh - zl]
    # pylint: enable=invalid-name
    lattice_vectors = [a1, a2, a3]
    return np.asarray(lattice_vectors)


def lammps_run_parser(log_file):
    """ parse the  run sections of the log

    Parameter
    ---------
    log_file: log_file

    Return
    ------
    list of pandas df, one for each run, columns name is autodetected
    units info.
    """
    with open(log_file) as file_stream:
        tables = []
        in_table = False
        for line in file_stream:
            if line[:5] == 'units':
                units = line.strip().split()[-1]
            if line[:4] == 'Step':
                new_table = [','.join(line.strip().split())]
                in_table = True
                continue
            elif line[:5] == 'ERROR':
                tables.append(pd.read_csv(StringIO('\n'.join(new_table))))
                raise ValueError('Error label found', tables, units)
            elif line[:4] == 'Loop':
                tables.append(pd.read_csv(StringIO('\n'.join(new_table))))
                in_table = False
                continue

            if in_table:
                new_table.append(','.join(line.strip().split()))
    return tables, units


class Pos():
    """class to centralize operation on parsed pos entry

    Parameters
    ----------
    posdict: a dictionary containing the information inside pos entry

    Note
    ----
    This class should be a kind of base class ExampleJsonWrapper..
    """

    def __init__(self, posdict):

        # mandatory info
        atoms_df = posdict['ATOMS']
        self._key = posdict['key']
        self._pos = atoms_df[['x', 'y', 'z']].values
        self._species = atoms_df.type.values - 1
        self._lattice = convert_box_bounds_to_lattice(posdict['BOX BOUNDS'][0])
        self._pbc = posdict['BOX BOUNDS'][1]
        self._time_step = posdict['TIMESTEP']

        # optional info (beg forgiveness)
        try:
            self._forces = atoms_df[['fx', 'fy', 'fz']].values
        except KeyError:
            logger.info('%s forces not available', self._key)
            self._forces = None

        try:
            self._per_atom_energy = atoms_df['c_2'].values
        except KeyError:
            logger.info('%s per atom energy not available', self._key)
            self._per_atom_energy = None

        self._atom_df = atoms_df

    def to_example(self, lammps_runs, units):
        """create the example dict ready to be saved

        Parameters
        ----------
        lammps_run: list of all parsed run in the lammps file
        units: TBD, not implemented yet. only metal is hard coded

        Returns
        -------
        dictionary ready to be saved
        """
        if units != 'metal':
            raise ValueError('different units not yet supported')

        system_info = {}
        system_info['key'] = self._key
        if lammps_runs is None:
            system_info['energy'] = [0.0, 'fake']
        else:
            system_info['energy'] = [
                _recover_energy(lammps_runs, self._time_step), 'eV'
            ]
        system_info['lattice_vectors'] = self._lattice.tolist()
        system_info["unit_of_length"] = "angstrom"
        system_info["atomic_position_unit"] = "cartesian"
        atomic_pos = []

        if self._forces is not None:
            for atom_id, (pos, force) in enumerate(
                    zip(self._pos, self._forces)):
                typ = 'C'
                atomic_pos.append((int(atom_id), typ, pos.tolist(),
                                   force.tolist()))
        else:
            for atom_id, pos in enumerate(self._pos):
                typ = 'C'
                atomic_pos.append((int(atom_id), typ, pos.tolist()))
        if self._per_atom_energy is not None:
            system_info['per_atom_energy'] = self._per_atom_energy.tolist()
        system_info['atoms'] = atomic_pos
        system_info['time_step'] = self._time_step
        return system_info


class PosFileParser():
    """Class to wrap a pos file
    provide an iterator over entry in the file

    Parameters
    ----------
    filename: file name.
    """

    def __init__(self, filename):
        self._filename = filename
        self._file_stream = open(filename)

    def __iter__(self):
        return self

    def __next__(self):  # pylint: disable=too-many-locals
        parsed_data = {}
        key = ''
        sha_buffer = b''

        def _read_line(file_stream):
            nonlocal sha_buffer
            line = file_stream.readline()
            sha_buffer += line.encode('utf-8')
            return line

        while True:
            line = _read_line(self._file_stream)
            if line == '':
                self._file_stream.seek(0)
                raise StopIteration
            if 'ITEM' in line:
                _dummy, function = line.split(':')
                for key, value in _RESERVED_WORDS.items():
                    for k_word, f_word in zip(key.split(), function.split()):
                        if k_word != f_word:
                            # this key is not the correct one
                            break
                    else:
                        # key found!
                        parameters = function.split()[len(key.split()):]
                        parsing_function = value[1]
                        break
                else:
                    # no key fuond!
                    raise ValueError('reserved word not found')
                # pylint: disable=undefined-loop-variable
                number_of_lines = value[0] if value[
                    0] not in _RESERVED_WORDS else parsed_data[value[0]]
                # pylint: enable=undefined-loop-variable
                payload = [
                    _read_line(self._file_stream).strip()
                    for _ in range(number_of_lines)
                ]
                parsed_data[key] = parsing_function(parameters, payload)
            if key == 'ATOMS':
                # atom is assumed to be the last token for each entry
                break

        # TODO this key to be more general should include more info...
        # like units, energy..
        parsed_data['key'] = hashlib.sha256(sha_buffer).hexdigest()
        return Pos(parsed_data)


def search_in_file(lammps_log_parsed, pos_file, out_dir):
    """ read pos_file and make a jsons for each entry
    """
    for pos in PosFileParser(pos_file):
        example_dict = pos.to_example(*lammps_log_parsed)

        out_file = "{}.example".format(example_dict['key'])
        with open(os.path.join(out_dir, out_file), 'w') as file_stream:
            json.dump(example_dict, file_stream)


def search_in_directory(lammps_log_parsed, directory, extension, out_dir):
    """ read files in indir and make a json out of them
    """

    files = [x for x in os.listdir(directory) if x.endswith(extension)]
    logger.debug(directory)
    logger.debug(extension)
    for file in files:
        for pos in PosFileParser(os.path.join(directory, file)):
            example_dict = pos.to_example(*lammps_log_parsed)
            out_file = "{}.example".format(example_dict['key'])
            with open(os.path.join(out_dir, out_file), 'w') as file_stream:
                json.dump(example_dict, file_stream)


def main(args):
    if args.log_file is None and not args.ignore_energies:
        logger.info('log file not specified, '
                    'if you want really to run the code '
                    'specify --ignore_energies')
        exit(0)

    if args.ignore_energies:
        logger.warning('--ignore_energies will have no effect')

    if args.ignore_errors:
        logger.warning('--ignore_errors active')

    if not ((args.dump_directory is None) != (args.dump_file is None)):
        logger.info('dump directory OR dump file ' 'must be specified')
        exit(0)

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    if args.log_file is None:
        logger.info('no log file passed, energy value will be faked to 0')
        # TODO fix this super duper hack
        lammps_log_parsed = [None, 'metal']
    else:
        try:
            lammps_log_parsed = lammps_run_parser(args.log_file)
        except ValueError as error:
            if args.ignore_errors:
                lammps_log_parsed = error.args[1:3]
            else:
                raise error

    if args.dump_file:
        search_in_file(lammps_log_parsed, args.dump_file, args.outdir)
    if args.dump_directory:
        search_in_directory(lammps_log_parsed, args.dump_directory,
                            args.dump_extension, args.outdir)


if __name__ == '__main__':

    PARSER = argparse.ArgumentParser(description='creating json from dumps')

    PARSER.add_argument(
        '-log', '--log_file', type=str, help='log_file', required=False)
    PARSER.add_argument(
        '-ign_en',
        '--ignore_energies',
        help='fake an energy value',
        action='store_true',
        required=False)

    PARSER.add_argument(
        '-ign_err',
        '--ignore_errors',
        help='try recover some simple errors',
        action='store_true',
        required=False)

    PARSER.add_argument(
        '-dumpd',
        '--dump_directory',
        type=str,
        help='dump directory',
        required=False)
    PARSER.add_argument(
        '-ext',
        '--dump_extension',
        type=str,
        default='atom',
        help='dump extension',
        required=False)

    PARSER.add_argument(
        '-dumpf', '--dump_file', type=str, help='dump file', required=False)

    PARSER.add_argument(
        '-o', '--outdir', type=str, help='output directory', required=True)

    # TODO add also a column selector to chose which column must be used as
    # energy
    # TODO for now only one pos file can be read each time
    ARGS = PARSER.parse_args()

    logger.warning('UNDER DEVELOPMENT')
    logger.warning('be careful with units!!!!')
    main(args=ARGS)
