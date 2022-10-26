###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import os
import sys
import re
from string import ascii_letters, punctuation, digits
from random import choice, randint
import argparse
import json
import hashlib

def create2_json(posfile, efile, outdir):
    ''' This module read the configurations from xyz format
        in posfile
        and the energy from the efile

        return: jsons equals number of configs
    '''
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    coll=open('energyfile.dat','a+')
    energy_unit = 'ev' # at the moment
            ##########################################################
    with open(posfile,'r') as fp:
        lines=fp.readlines()
    try:

        if lines[0].split()[1] == 'TIMESTEP':
            Natoms = int(lines[3])
    except IndexError:
        print(posfile,'file format not supported')
        exit()

    Nconfig=int(int(len(lines))/(Natoms+9))
    print(Nconfig)
    ii=0
    start_old=0
    count = 0
    header2 = []
    for j in range(Nconfig):
        kk=1
        atomic_pos=[]
        system_info={}
        for k in range(ii*(Natoms+9),(ii+1)*(Natoms+9)):

                    #get the lattice parameters

            if kk == 6 :
                try:
                    a = lines[k]
                    xbl, xbh, xy = a.split()
                    xbl=float(xbl);xbh=float(xbh);xy=float(xy)
                except ValueError:
                    break
            if kk == 7 :
                try:
                    a = lines[k]
                    ybl, ybh, xz = a.split()
                    ybl=float(ybl);ybh=float(ybh);xz=float(xz)
                except ValueError:
                    break
            if kk == 8 :
                try:
                    a = lines[k]
                    zl, zh, yz = a.split()
                    zl=float(zl);zh=float(zh);yz=float(yz)
                except ValueError:
                    break
                xmin = np.min([0.0,xy,xz,xy+xz])
                xmax = np.max([0.0,xy,xz,xy+xz])
                ymin = np.min([0.0,yz])
                ymax = np.max([0.0,yz])

                xl = xbl-xmin
                xh = xbh-xmax
                yl = ybl-ymin
                yh = ybh-ymax

                a1 = [xh-xl, 0., 0.]
                a2 = [xy, yh-yl, 0.0]
                a3 = [xz, yz, zh-zl]

                lattice_vectors=[a1,a2,a3]

            #read header for atomic positions
            if kk == 9:
                header = lines[k].split()
            if kk > 9 and kk <= Natoms + 9:

                pos = []
                atom_info = lines[k].split()
                try:
                    symbol = atom_info[header.index('element')-2]
                    idx = np.int(atom_info[header.index('id')-2])
                    x = float(atom_info[header.index('x')-2])
                    y = float(atom_info[header.index('y')-2])
                    z = float(atom_info[header.index('z')-2])
                    pos.append(idx)
                    pos.append(symbol)
                    pos.append([x,y,z])
                    try:
                        fx = float(atom_info[header.index('fx')-2])
                        fy = float(atom_info[header.index('fy')-2])
                        fz = float(atom_info[header.index('fz')-2])
                        pos.append([fx,fy,fz])
                    except ValueError:
                        fx=0.0; fy=0.0; fz=0.0
                        pos.append([fx,fy,fz])
                        continue
                except ValueError:
                    break

                atomic_pos.append(pos)
            kk += 1
        #define string
        string = str(lines[ii*(Natoms+9)+1].split()[0])

        ii += 1
        if j == 0:
            start_idx=0


        with open(efile,'r') as ep:
            lines1 = ep.readlines()

        for idx in range(start_old,len(lines1)):
            line1 = lines1[idx]
                #save header
            if j == 0:
                try:
                    if line1.split()[0]=='Step':
                        start_idx=1
                        header2=line1.split()
                        #continue
                except IndexError:
                    continue
           #print(header)
            try:
                if line1.split()[0] == 'Loop':
                    break
            except IndexError:
                pass

            try:
                #print(string,line1.split()[0],idx,start_old)
                if start_idx == 1 and line1.split()[0] == string:
                    try:
                        e = line1.split()[header2.index('PotEng')]
                    except ValueError:
                        e = line1.split()[header2.index('E_pair')]
                    e = float(e)
                    count += 1
                    T=line1.split()[header2.index('Temp')]
                    start_old = idx
                    #print(count)
                    coll.write(str(e)+"\n")
                    eold=e
                    var = posfile + str(e) + str(count) + str(np.linalg.det(lattice_vectors))
                            #create the outputfile
                    hasher = hashlib.sha256()
                    hasher.update(var.encode('utf8'))
                    outfile = hasher.hexdigest()

                    system_info['atoms'] = atomic_pos
                    system_info['energy'] = [e,energy_unit]
                    system_info['lattice_vectors']  = lattice_vectors
                    system_info["unit_of_length"] = "angstrom"
                    system_info["atomic_position_unit"] = "cartesian"
                    system_info = json.dumps(system_info)

                    fil = open(os.path.join(outdir, outfile+'.example'),'w+')
                    fil.write(str(system_info))
                    fil.close()

                    break
            except IndexError:
                continue
#######################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creating json')
    parser.add_argument('-p', '--pfile', type=str,
                        help='pos file', required=True)
    parser.add_argument('-e', '--efile', type=str,
                        help='energy file', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='out path', required=True)
    args = parser.parse_args()
    if args.outdir:
        create2_json(posfile=args.pfile, efile=args.efile, outdir=args.outdir)
