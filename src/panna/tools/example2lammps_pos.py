###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import os

from lib import ExampleJsonWrapper
from lib import init_logging

def example2lammp(indir, outdir):
    """ convert PANNA example format to lammps pos format

    Parameters
    ----------
    indir: directory con example files
    outdir: path to export lammps files
            files are organized in folders, file name is ini.pos
    Return
    ------
    None
    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    files_name = [os.path.join(indir, x) for x in os.listdir(indir)]

    for file_name in files_name:
        key = file_name.split('/')[-1].split('.')[0]

        if not os.path.exists(os.path.join(outdir, key)):
            os.mkdir(os.path.join(outdir, key))

        example = ExampleJsonWrapper(file_name)

        with open(os.path.join(outdir, key, "ini.pos"), "w") as file_stream:
            file_stream.write(example.to_lammps())

if __name__ == '__main__':
    init_logging()
    PARSER = argparse.ArgumentParser(
        description='Making lammps pos file from example')
    PARSER.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)
    PARSER.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
    ARGS = PARSER.parse_args()
    if ARGS.indir and ARGS.outdir:
        example2lammp(indir=ARGS.indir, outdir=ARGS.outdir)
