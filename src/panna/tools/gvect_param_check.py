###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os
import argparse
import configparser
#Depending on your system you might need to uncomment these lines
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Conversion constants
#A2BOHR = 1.889725989
BOHR2A = np.float32(0.52917721067)


def main(conf_file, **kvargs):
    config = configparser.ConfigParser()
    config.read(conf_file)
    # parameters:
    folder_info = config['IO_INFORMATION']
    input_json_dir = folder_info.get('input_json_dir', None)
    binary_out_dir = folder_info.get('output_gvect_dir', './gvects')
    log_dir = folder_info.get('log_dir', './logs')
    symmetry_function = config['SYMMETRY_FUNCTION']
    symmetry_function_type = symmetry_function.get('type')
    #
    gv_par = config['GVECT_PARAMETERS']
    # the default unit of parameters is angstrom.
    param_unit = gv_par.get('gvect_parameters_unit', 'angstrom')
    if param_unit in ['a', 'angstrom', 'A', 'Angstrom', 'ANGSTROM']:
        unit2A = np.float32(1)
    elif param_unit in ['au', 'bohr', 'bohr_radius', 'BOHR']:
        unit2A = np.float32(1) * BOHR2A
    else:
        print('Parameter unit not recognized, assuming Angstrom')
        unit2A = np.float32(1)

    #
    if symmetry_function_type == 'mBP':
        gvect_parameters = {
            # RADIAL_COMPONENTS
            'eta_rad':
            gv_par.getfloat('eta_rad') / (unit2A * unit2A),
            'Rc_rad':
            gv_par.getfloat('Rc_rad') * unit2A,
            'Rs0_rad':
            gv_par.getfloat('Rs0_rad') * unit2A,
            'RsN_rad':
            gv_par.getint('RsN_rad'),
            # infer if not present:
            'Rsst_rad':
            gv_par.getfloat(
                'Rsst_rad',
                (gv_par.getfloat('Rc_rad') - gv_par.getfloat('Rs0_rad')) /
                gv_par.getint('RsN_rad')) * unit2A,

            # ANGULAR_COMPONENTS
            'eta_ang':
            gv_par.getfloat('eta_ang') / (unit2A * unit2A),
            'Rc_ang':
            gv_par.getfloat('Rc_ang') * unit2A,
            'Rs0_ang':
            gv_par.getfloat('Rs0_ang') * unit2A,
            'RsN_ang':
            gv_par.getint('RsN_ang'),
            # infer if not present
            'Rsst_ang':
            gv_par.getfloat(
                'Rsst_ang',
                (gv_par.getfloat('Rc_ang') - gv_par.getfloat('Rs0_ang')) /
                gv_par.getint('RsN_ang')) * unit2A,
            'zeta':
            gv_par.getfloat('zeta'),
            'ThetasN':
            gv_par.getint('ThetasN')
        }

    elif symmetry_function_type == 'BP':
        eta_rad_list = gv_par.get('eta_rad', None)
        eta_ang_list = gv_par.get('eta_ang', None)
        zeta_list = gv_par.get('zeta', None)

        gvect_parameters = {
            # RADIAL_COMPONENTS
            'eta_rad': [
                float(x.strip()) / (unit2A * unit2A)
                for x in eta_rad_list.split(',')
            ],
            'Rc_rad':
            gv_par.getfloat('Rc_rad') * unit2A,
            'Rs0_rad':
            gv_par.getfloat('Rs0_rad') * unit2A,
            'RsN_rad':
            gv_par.getint('RsN_rad'),
            # infer if not present:
            'Rsst_rad':
            gv_par.getfloat(
                'Rsst_rad',
                (gv_par.getfloat('Rc_rad') - gv_par.getfloat('Rs0_rad')) /
                gv_par.getint('RsN_rad')) * unit2A,

            # ANGULAR_COMPONENTS
            'eta_ang': [
                float(x.strip()) / (unit2A * unit2A)
                for x in eta_ang_list.split(',')
            ],
            'Rc_ang':
            gv_par.getfloat('Rc_ang') * unit2A,
            'zeta': [float(x.strip()) for x in zeta_list.split(',')],
            'ThetasN':
            gv_par.getint('ThetasN')
        }
    else:
        #logger.error('unknown symmetry_function_type')
        raise ValueError('unknown symmetry_function_type')
    # end of parameters

    # Plot what these gvect parameters look like
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
    if symmetry_function_type == 'mBP':
        # Radial
        rdiff = np.arange(0.0, gvect_parameters['Rc_rad'], 0.1)
        #
        for index in range(gvect_parameters['RsN_rad']):
            #for idx_rs_rad in range(RsN_rad):
            Grad = np.exp( - gvect_parameters['eta_rad'] * (rdiff -gvect_parameters['Rs0_rad'] - index *  \
                         gvect_parameters['Rsst_rad'])**2 ) * 0.5 * \
                         (1.0 + np.cos(np.pi *  rdiff / gvect_parameters['Rc_rad']))
            ax1.plot(rdiff, Grad)
        # Angular
        theta = np.arange(0.0, np.pi, 0.1)
        Thetasst = np.pi / gvect_parameters['ThetasN']
        for indexTh in range(gvect_parameters['ThetasN']):
            Gang = 2.0 * (0.5 + np.cos(theta - (indexTh + 0.5) * Thetasst) *
                          0.5)**gvect_parameters['zeta']
            ax2.plot(theta, Gang)
        # Anglo-radial :)
        rdiff_ave = np.arange(0.0, gvect_parameters['Rc_ang'], 0.1)
        for indexR in range(gvect_parameters['RsN_ang']):
            Gang_R = np.exp(-gvect_parameters['eta_ang'] * \
                     (rdiff_ave - gvect_parameters['Rs0_ang'] - indexR * gvect_parameters['Rsst_ang'])**2) * \
                      0.25 * (1.0 + np.cos(np.pi * rdiff_ave / gvect_parameters['Rc_ang']))**2
            ax3.plot(rdiff_ave, Gang_R)
        #
        ax1.set_title('G_radial')
        ax2.set_title('G_angular')
        ax3.set_title('G_angular_radial')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gvector parameter plotter')
    parser.add_argument(
        '-c', '--config', type=str, help='config file', required=True)
    args = parser.parse_args()
    main(args.config)
