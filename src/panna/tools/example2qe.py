###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import argparse
import hashlib
import logging
import os
from string import Template

from lib import ExampleJsonWrapper, init_logging

# pylint: disable=invalid-name
logger = logging.getLogger('panna.tools')

_card_template = Template("""
&control
    calculation = 'scf',
    restart_mode='from_scratch',
    prefix='silicon',
    tstress = .true.
    tprnfor = .true.
    pseudo_dir = '$pseudo_dir'
    max_seconds = 14100
    outdir='.'
/
&system
    ibrav=  0,
    nat=  $nat,
    ntyp= $ntyp,
    ecutwfc = $ecutwfc,
    ecutrho = $ecutrho,
    occupations = 'smearing',
    smearing = 'mv',
    degauss = $degauss,
/
&electrons
    diagonalization='david'
    mixing_mode = 'plain'
    mixing_beta = 0.7
    conv_thr =  1.0d-8
    electron_maxstep = 100
/
""")


# pylint: enable=invalid-name


def example2qe(example,
               template_parameters=None,
               kpoint_card=None,
               **kwargs):
    """ convert PANNA example format to qe input, adding a template

    Parameters
    ----------
    example: example to export
    outdir: path to export lammps files
            files are organized in folders, file name is ini.pos
    template_parameters: dict, parameters to use in the template,
                         if empty $value will appear in final text
    kpoint_card: optional, string for the kpoint section
    kpoints_density: optional, density a_0 * k_0 for kpoint in angstrom

    Notes
    -----
    kpoint_card and kpoint_density can not be provided together


    Return
    ------
    file as string
    """
    if not template_parameters:
        template_parameters = {}

    if ('kpoints_density' in kwargs) and ('kpoint_card' in kwargs):
        raise ValueError(
            'kpoint_density and kpoint_card can not be provided together')

    # find internal values
    template_parameters['nat'] = example.number_of_atoms
    template_parameters['ntyp'] = example.number_of_species


    qe_cards = example.to_qe_cards(**kwargs)

    if kpoint_card:
        logger.info('overriding kpoints card with provided one')
        qe_cards['K_POINTS'] = kpoint_card

    qe_string = _card_template.safe_substitute(template_parameters) + '\n'
    for _dummy, value in qe_cards.items():
        qe_string += value.safe_substitute(template_parameters)
        qe_string += '\n\n'

    logger.debug(qe_string)
    return qe_string


def main(indir, outdir, kpoints_density):
    """ does stuff, do your changes here

    Parameters
    ----------
    outdir: path to export lammps files
            files are organized in folders, file name is ini.pos

    Return
    ------
    None
    """
    files_name = [os.path.join(indir, x) for x in os.listdir(indir)]
    # template is a dictionary, for now to discover the key run the code
    # and look at placeholder names.
    template_parameters = {}
    template_parameters['Si_pseudo'] = 'Si.pbesol-n-rrkjus_psl.1.0.0.UPF'
    template_parameters['pseudo_dir'] = '/home/rlot/QE/pseudo_potentials'
    template_parameters['ecutwfc'] = 50
    template_parameters['ecutrho'] = 240
    template_parameters['degauss'] = 0.02
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    run_all = open(os.path.join(outdir, 'run_all.sh'), 'w')
    run_all.write('declare -a arr=(\n')
    for file_name in files_name:
        key = file_name.split('/')[-1].split('.')[0]
        example = ExampleJsonWrapper(file_name)
        qe_string = example2qe(
            example, template_parameters, kpoints_density=kpoints_density)
        key_2 = hashlib.sha1(qe_string.encode()).hexdigest()
        final_outdir = os.path.join(outdir, key, key_2)
        if not os.path.exists(final_outdir):
            os.makedirs(final_outdir)
        os.symlink('../../run.job', os.path.join(final_outdir, 'run.job'))
        run_all.write(os.path.join(key, key_2) + '\n')
        with open(os.path.join(final_outdir, "qe.ini"), "w") as file_stream:
            file_stream.write(qe_string)
    run_all.write(')\n\n')
    run_all.write('for i in "${arr[@]}"\n')
    run_all.write('do\n')
    run_all.write('\tcd $i;\n')
    run_all.write('\tcd ../..;\n')
    run_all.write('done')
    run_all.close()


if __name__ == '__main__':
    init_logging()
    logger.warning('this script need to be personalized')
    PARSER = argparse.ArgumentParser(
        description='Making lammps pos file from example')
    PARSER.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)
    PARSER.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
    PARSER.add_argument(
        '-kpp_density',
        '--kpp_density',
        type=float,
        help='kpp_density (a0 * k0)',
        required=False)
    PARSER.add_argument(
        '-dgauss', '--dgauss', type=float, help='dgauss value', required=False)
    ARGS = PARSER.parse_args()
    if ARGS.indir and ARGS.outdir:
        main(
            indir=ARGS.indir,
            outdir=ARGS.outdir,
            kpoints_density=ARGS.kpp_density)
