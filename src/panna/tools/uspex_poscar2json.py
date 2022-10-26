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
import logging as logg

def POSCAR2json(POSCARS_file, enfile, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    ##########################################################

    with open(POSCARS_file,'r') as fp:
        lines_posfile=fp.readlines()
    fp.close()

    with open(enfile,'r') as ep:
        lines_efile=ep.readlines()
    ep.close()
    
   
    # get atom types       
    typ = [x for x in lines_posfile[5].split()] 
    #get num of atoms per type
    natyp = [y for y in lines_posfile[6].split()]
    #create species list 
    species=[]
    for i in range(len(typ)):
        for j in range(int(natyp[i])):   
            species.append(typ[i])

    Natoms = len(species)
    #number of configurations in POSCAR
    nlines_per_config = Natoms + 8
    
    Nconfig=int(len(lines_posfile)/(nlines_per_config))
    print(Nconfig,len(lines_efile)-2)
    
    if Nconfig != len(lines_efile)-2 : 
        logg.error('POSCAR file and the energy file are not compatible')
        repeat = []
        for i in range(len(lines_posfile)):
            if lines_posfile[i].split()[0][0:2] == 'EA':
                
                repeat.append(lines_posfile[i].split()[0][2:])
        repeat = np.asarray(repeat).astype(int)
        repeat.sort()
        xold = 0 
        count=0
        for x in repeat:
            if x == xold:
                count += 1
            if count > 0:
                logg.info(x, 'configuration is repeated in the file',POSCARS_file, '\n check there could be others')
                exit()
            xold = x

    ii=0 
    
    eold=0
    coll=open('energyfile.dat','w+')
    
    for j in range(Nconfig):
        atomic_pos=[]
        system_info={}
        lattice_vectors = [] 
        #control index
        k = ii*nlines_per_config
         
        for i in range(nlines_per_config):

            #get the lattice parameters
            if i < 5 and i > 1 :
                a = lines_posfile[k+i]
                print(a)
                x, y, z = a.split()
                x = float(x); y = float(y); z = float(z)
                lattice_vectors.append([x,y,z])
                
            #get atomic positions
            elif i > 7:
                pos = []
                atom_symbol = species[i-8]
                a = lines_posfile[k+i]
                x, y, z = a.split()
                x = float(x); y = float(y); z = float(z)
                fx=0.;fy=0.;fz=0.
                pos.append(i-7)
                pos.append(atom_symbol)
                pos.append([x,y,z])
                pos.append([fx,fy,fz])
                atomic_pos.append(pos)
                print(pos)

        config_idx = int(lines_posfile[k].split()[0][2:])
        #get header:
        header = lines_efile[0].split()
        energy_unit= lines_efile[1].split()[0]
        print(header,energy_unit)
                
        for lne in lines_efile:
            
                              
            if lne.split()[header.index('ID')]==str(config_idx):
                try:
                    e = float(lne.split()[header.index('Enthalpy')\
                                      +len(typ)+1])
                    if energy_unit.split('/')[-1] == 'atom)':
                        e = e*Natoms
                except ValueError:
                    e = float(lne.split()[header.index('Enthalpies')\
                                      +len(typ)+1])
                    if energy_unit.split('/')[-1] == 'atom)':
                        e = e*Natoms

                coll.write(str(e)+"\n")
                eold=e
                #create the outputfile
                strp=''.join(x for x in re.split(r"\W", str(e)) if x) +\
                "abcdefghijklm" #nopqrstuvwxyz"
                #outfile="".join(choice(strp.lower()) for i in range(randint(50,50)))+".example"
                outfile="".join(choice(strp.lower()) for i in range(randint(5,5)))+"_"+str(config_idx)+".example"

                # write to file
                system_info['atoms'] = atomic_pos
                system_info['energy'] = [e,'ev']
                system_info['lattice_vectors']  = lattice_vectors
                system_info["unit_of_length"] = "angstrom"
                system_info["atomic_position_unit"] = "crystal"
                system_info=json.dumps(system_info)
                fil=open(os.path.join(outdir,outfile),'w+')
                fil.write(str(system_info))
                fil.close()
                break


        ii +=1
#######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='creating json')
    parser.add_argument('-p', '--POSCARS_file', type=str,
                        help='POSCARS', required=True)
    parser.add_argument('-e', '--enfile', type=str,
                        help='energy', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='out path', required=True)
    args = parser.parse_args()
    if args.POSCARS_file and args.enfile and args.outdir:
       POSCAR2json(POSCARS_file=args.POSCARS_file, enfile=args.enfile,\
                        outdir=args.outdir)    
