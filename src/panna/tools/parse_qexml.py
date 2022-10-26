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

import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

atom = ['', '', '', '']


def main(indir, outdir, addhash, xml_filename='data-file-schema'):
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
                #initialize json dictionary
                panna_json = dict()
                panna_json['atoms'] = []

                xml_file = os.path.join(root_dir, file)
                #panna_json['name'] = os.path.splitext( os.path.basename(xml_file) )[0]
                panna_json['source'] = os.path.abspath(xml_file)
                # is this a valid xml file?
                try:
                    tree = et.parse(xml_file)
                except et.ParseError:
                    logger.error('Error parsing XML file:  %s', file)
                    continue
                root = tree.getroot()

                #INPUT
                inp = root.find('input')
                control_variables = inp.find('control_variables')
                # compute the hash
                hasher = hashlib.sha256()
                with open(xml_file, 'rb') as file_stream:
                    hasher.update(file_stream.read())
                panna_json['key'] = hasher.hexdigest()

                atomic_str = inp.find('atomic_structure')
                # alat = atomic_str.attrib['alat'] - panna doesnt use
                # nat = atomic_str.attrib['nat'] - panna doesnt use - could be used for sanity check?

                # ATOMIC POSITIONS
                atomic_pos = atomic_str.find('atomic_positions')
                # QE defaults - unchecked
                panna_json['atomic_position_unit'] = 'cartesian'
                panna_json['unit_of_length'] = 'bohr'

                # CELL
                cell = atomic_str.find('cell')
                panna_json['lattice_vectors'] = []
                for vector in cell:
                    panna_json['lattice_vectors'].append([
                        float(vector.text.split()[0]),
                        float(vector.text.split()[1]),
                        float(vector.text.split()[2])
                    ])
                # INFO FROM OUTPUT
                outp = root.find('output')
                # TOTAL ENERGY
                total_energy = outp.find('total_energy')
                etot = float(total_energy.find('etot').text)
                unit_of_energy = "Ha"  #QE default energy unit in current xml files.
                panna_json['energy'] = [etot, unit_of_energy]
                # FORCES
                forces = []
                force_array = outp.find('forces').text.split()
                for i in range(0, len(force_array), 3):
                    forces.append([
                        float(force_array[i]),
                        float(force_array[i + 1]),
                        float(force_array[i + 2])
                    ])
                for idx, at in enumerate(atomic_pos.findall('atom')):
                    atom[0] = int(at.attrib['index'])
                    atom[1] = str(at.attrib['name'])
                    atom[2] = at.text.split(
                    )  # scientific notation atomic positions list
                    atom[2] = [
                        float(atom[2][0]),
                        float(atom[2][1]),
                        float(atom[2][2])
                    ]
                    panna_json['atoms'].append(
                        [atom[0], atom[1], atom[2], forces[idx]])

                out_name = file.split('.xml')[0]
                if addhash:
                    out_name = panna_json['key'] + '_' + out_name

                with open(
                        os.path.join(outdir, out_name + ".example"),
                        'w') as outfile:
                    json.dump(panna_json, outfile)

                #i_read += 1


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='QE xml to PANNA json converter')
    PARSER.add_argument(
        '-i',
        '--indir',
        type=str,
        help='input directory that holds all the outdirs',
        required=True)
    PARSER.add_argument(
        '-o', '--outdir', type=str, help='output directory', required=True)
    PARSER.add_argument('--addhash',
        action='store_true',
        help='add hash to the file name',
        required=False)
    ARGS = PARSER.parse_args()
    main(ARGS.indir, ARGS.outdir, ARGS.addhash)
