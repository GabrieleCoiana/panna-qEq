###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
""" Parsing the qe xml files to construct -> panna simulation json
"""

from xml.etree import ElementTree as et
import os
import argparse
import json
import hashlib
import numpy as np

import logging

from panna.lib.log import emit_splash_screen

logger = logging.getLogger(__name__)

class QeXmlWrapper():
    def __init__(self, root_dir, file_name):
        self._file_name = file_name
        xml_file = os.path.join(root_dir, file_name)

        self._source = os.path.abspath(xml_file)
        # is this a valid xml file?
        try:
            tree = et.parse(xml_file)
            self.valid = True
        except et.ParseError:
            logger.error('Error parsing XML file:  {xml_file}')
            self.valid = False
            return
        self._root = tree.getroot()

        hasher = hashlib.sha256()
        with open(xml_file, 'rb') as file_stream:
            hasher.update(file_stream.read())
            computed_hash = hasher.hexdigest()
        self.sha = computed_hash

    @property
    def is_converged(self):
        try:
            path = './/convergence_info//scf_conv//convergence_achieved'
            converged = self.output.find(path).text == 'true'
        except AttributeError:
            logger.warning('conv. info not found in xml. Assumed converged')
            converged = True

        return converged

    @property
    def is_valid(self):
        return self.valid

    @property
    def atomic_position_unit(self):
        return 'cartesian'

    @property
    def unit_of_lenght(self):
        return 'bohr'

    @property
    def input(self):
        return self._root.find('input')

    def _atomic_structure(self):
        atomic_str = self.input.find('atomic_structure')

        atomic_pos = atomic_str.find('atomic_positions')

        atoms = []
        for atom in atomic_pos.findall('atom'):
            index = int(atom.attrib['index']) - 1
            name = str(atom.attrib['name'])
            pos = [float(x) for x in atom.text.split()]
            atoms.append([index, name, pos])

        cell = atomic_str.find('cell')
        lattice_vectors = []
        for vector in cell:
            lattice_vectors.append([float(x) for x in vector.text.split()])

        return lattice_vectors, atoms

    @property
    def symbols(self):
        cell, atoms = self._atomic_structure()
        symbols = [atom[1] for atom in atoms]
        return symbols

    @property
    def pos(self):
        cell, atoms = self._atomic_structure()
        pos = [atom[2] for atom in atoms]
        return np.array(pos)

    @property
    def cell(self):
        cell, atoms = self._atomic_structure()
        return np.array(cell)

    @property
    def output(self):
        return self._root.find('output')

    @property
    def energy(self):
        total_energy = self.output.find('total_energy')
        etot = float(total_energy.find('etot').text)
        return etot, "Ha"

    @property
    def forces(self):
        return np.array(self._forces())

    def _forces(self):
        forces = []
        force_array = self.output.find('forces').text.split()
        for i in range(0, len(force_array), 3):
            forces.append([float(force_array[i + x]) for x in range(3)])
        return forces

    def to_panna_json(self):
        panna_json = dict()
        # INPUT
        panna_json['key'] = self.sha

        # QE defaults - unchecked
        panna_json['atomic_position_unit'] = self.atomic_position_unit
        panna_json['unit_of_length'] = self.unit_of_lenght
        panna_json['energy'] = self.energy

        # CELL
        cell, poss = self._atomic_structure()
        panna_json['lattice_vectors'] = cell

        # FORCES
        forces = self._forces()

        atoms = []
        for pos, force in zip(poss, forces):
            atoms.append([*pos, force])
        panna_json['atoms'] = atoms
        return panna_json


def main(indir, outdir, add_hash=False, xml_filename='data-file-schema'):
    if os.path.isdir(outdir):
        outdir = os.path.abspath(outdir)
    else:
        os.mkdir(outdir)
        outdir = os.path.abspath(outdir)

    logger.info('input directory: %s', indir)
    logger.info('output directory: %s', outdir)

    #find QE xmls with prefix names
    for root_dir, dummy_dirs, files in os.walk(indir):
        for file in files:
            if file.endswith('xml') and file != '{}.xml'.format(xml_filename):
                wrapper = QeXmlWrapper(root_dir, file)
                if not wrapper.is_converged:
                    logger.warning(f'not converged {file} skipping')
                    continue
                if not wrapper.is_valid:
                    logger.warning(f'not valid {file} skipping')
                    continue

                computed_hash = wrapper.sha
                panna_json = wrapper.to_panna_json()

                new_file_name = file.split('.xml')[0] + ".example"
                if add_hash:
                    new_file_name = computed_hash + '_' + new_file_name
                with open(os.path.join(outdir, new_file_name), 'w') as outfile:
                    json.dump(panna_json, outfile)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='QE xml to PANNA json converter')
    PARSER.add_argument('-i',
                        '--indir',
                        type=str,
                        help='input directory that holds all the outdirs',
                        required=True)
    PARSER.add_argument('-o',
                        '--outdir',
                        type=str,
                        help='output directory',
                        required=True)
    PARSER.add_argument('--addhash',
                        action='store_true',
                        help='add hash to the file name',
                        required=False)
    ARGS = PARSER.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s')
    emit_splash_screen(logger)
    main(ARGS.indir, ARGS.outdir, ARGS.addhash)
