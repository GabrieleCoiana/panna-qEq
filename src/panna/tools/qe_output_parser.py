###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
#QE relaxation output -> panna json
#
#Looks inside the input directory and operates on all files as if they are QE outputs
#Makes json files with the same name plus two more random characters, even if addhash is off.
#Uses underscore in json filename to indicate relaxation index
#
#The following is assumed on how the units are displayed during relaxation: 
# 
# CELL_PARAMETERS (bohr)
# ATOMIC_POSITIONS (crystal) or ATOMIC_POSITIONS (bohr) 
#
#
import os, sys
import argparse
import json
import numpy as np
import random, string
# 
import keys_generator
#uses hash_key

def mychoices(population, k):
    n=len(population)+0.0
    return [population[int(random.random() * n )] for i in range(k)]


def main(indir, outdir, addhash, **kvargs):
    if os.path.isdir(outdir): 
       outdir = os.path.abspath(outdir)
       print('Outdir exist {}'.format(outdir), flush=True)
    else:
       os.mkdir(outdir)
       outdir = os.path.abspath(outdir)
       print('Outdir created {}'.format(outdir), flush=True)
    unit_of_energy = 'Ry'
    #find QE outputs and process 
    for rt, dirs, files in os.walk(indir):
        for f in files:
            #print('file being processed {}'.format(f), flush=True) 
            #initialize json dictionary
            panna_json = dict()
            panna_json['atoms']=[]
            #
            pwoutput = os.path.join(rt,f)
            panna_json['name'] = os.path.basename(pwoutput) 
            panna_json['source'] = os.path.abspath(pwoutput)
            panna_json['key'] =   os.path.basename(pwoutput)
            ff = open(pwoutput, 'r')
            data = ff.read()
            etot = 0.0 #in case the calculation finished before the energy was found
            lforces = False
            lstress = False
            #Split data into scf calculations
            #First is the initial info about the calculation
######################################################################################
            initpw = data.split('Self-consistent Calculation')[0]    
            data_lines = initpw.split('\n')
            #print(data_lines, flush=True)
            for num, line in enumerate(data_lines): 
                if 'lattice parameter (alat)' in line:
                    alat = float(line.split('=')[1].split('a.u')[0])
                    panna_json['alat'] = alat
#                    print('Alat is {}'.format(alat), flush=True)
                elif 'number of atoms/cell' in line:
                    nat = int(line.split('=')[1])
                elif 'crystal axes: ' in line: 
                    i = num+1
                    lat_list =  [ [data_lines[i+j].split()[3], 
                                   data_lines[i+j].split()[4], 
                                   data_lines[i+j].split()[5]] for j in range(3)]
                    lat_list = [ [float(lat_list[k][i])*alat for i in range(3)] for k in  range(3) ] 
#                    print('Lattice Vectors {}'.format(lat_list), flush=True)
                elif 'Cartesian axes' in line:
                    i = num +2
                    if 'site n.' in data_lines[i] and 'atom' in data_lines[i]:
                       symbol_list = [ data_lines[i+1+j].split()[1] for j in range(nat)]
                       atom_list =[  [data_lines[i+1+j].split()[6], 
                                     data_lines[i+1+j].split()[7], 
                                     data_lines[i+1+j].split()[8]]  for j in range(nat)]
                       pos_list = [ [float(atom_list[k][i])*alat for i in range(3)] for k in  range(nat) ]
                    #print('Atoms {}'.format(symbol_list), flush=True)
                    #print('Atoms {}'.format(pos_list), flush=True)
                    #print('Atoms {}'.format(atom_list), flush=True)
                elif 'nstep' in line:
                    #this is a relaxation with at most nstep steps
                    max_iter = int(line.split()[2])
            #################################################################################
            #Then the first scf calculation is processed
            if len(data.split('End of self-consistent calculation')) > 1:
                firstscf = data.split('End of self-consistent calculation')[1]
            else:
                print(pwoutput, flush=True)
                print("ERROR: NO SCF CALC WAS PERFORMED", flush=True)
                #exit(1)
                continue
            data_lines =  firstscf.split('\n')
            for num, line in enumerate( data_lines ):
                if '!    total energy' in line:
                   etot = line.split()[4]
                   #print('Total energy for this scf run {}'.format(etot), flush=True)
                elif 'Forces acting on atoms' in line:
                   lforces = True
                   i = num+2
                   forces_list =[ [data_lines[i+j].split()[6],
                                     data_lines[i+j].split()[7],
                                     data_lines[i+j].split()[8]]  for j in range(nat)]
                   forces_list = [ [float(forces_list[k][i]) for i in range(3)] for k in  range(nat) ]
                   #print('Forces {}'.format(forces_list), flush=True)
                elif 'P= ' in line:
                   lstress = True
                   i = num+1
                   stress_list =[  [data_lines[i+j].split()[0],
                                     data_lines[i+j].split()[1],
                                     data_lines[i+j].split()[2]]  for j in range(3)]
                   stress_list = [ [float(stress_list[k][i]) for i in range(3)] for k in  range(3) ]
                   #print('Stress in Ry/b^3 {}'.format(stress_list), flush=True)
            # Construct the panna_json for this first calculation
            panna_json['atomic_position_unit']='cartesian'            
            panna_json['unit_of_length']='bohr'
            panna_json['lattice_vectors'] = lat_list
            panna_json['energy'] = (float(etot),unit_of_energy)
            if lforces : 
                for at in range(nat):
                    panna_json['atoms'].append( [at+1, symbol_list[at], pos_list[at][:], 
                                             forces_list[at][:] ] )
            else:
                for at in range(nat):
                    panna_json['atoms'].append([at+1, symbol_list[at], pos_list[at][:] ] )
            if lstress: 
                 panna_json['stress'] = stress_list
           
            panna_json['key'] =  keys_generator.hash_key_v2(panna_json)
            if addhash : 
                    panna_json_name = keys_generator.hash_key_v2(panna_json)
            else : 
                    # panna_json_name = os.path.basename(pwoutput) 
                    # still adding some randomization just in case
                    panna_json_name = os.path.basename(pwoutput) + '.' + ''.join(mychoices(string.ascii_lowercase, k=2))  
                    panna_json_name_rand = panna_json_name
            panna_json['rlx_ind'] = 0
            starter_key = panna_json['key']
            panna_json['starter_key'] =  starter_key
            with open(outdir.rstrip('/')+"/"+panna_json_name+".example",'w') as outfile:
                    json.dump(panna_json, outfile)
######################################################################################################                
            etot = 0.0 #in case the calculation finished before the energy was found

            # Split the relaxation data in scf calculations
            if 'number of bfgs steps' in data:
                lrelax=True #automatically lforces is also true
                relaxsteps = data.split('number of bfgs steps')[1:]
                #print('There are {}  BFGS calculations'.format(len(relaxsteps)), flush=True )
                rlx_ind = 0
                # Here, the remaining file is split into BFGS steps,
                # If bfgs is converged, the last positions are equal to 'final coordinates'
                # hence it is ok to use this keyword to split. 
                # And for the final scf done at the final coordinates:
                # is it was successfully completed, the energy stored for the final coord. is that one,
                # is the calc died before that was finished, the last energy from bfgs is used.
                # 
                for step in relaxsteps:
                    panna_json['atoms'] = []
                    rlx_ind += 1
                    panna_json['rlx_ind'] =  rlx_ind 
                    data_lines = step.split('\n')
                    positions_found = False
                    energy_found = False
                    for num, line in enumerate( data_lines ):
                        if 'CELL_PARAMETERS' in line:
                           i=num+1
                           #print('this is a variable cell simulation') lstress should also be true
                           lat_list =  [ [data_lines[i+j].split()[0],
                                          data_lines[i+j].split()[1],
                                          data_lines[i+j].split()[2]] for j in range(3)]
                           lat_unit = line.split('(')[1].split(')')[0]
                           if lat_unit == panna_json['unit_of_length'] : 
                              lat_list = [ [float(lat_list[k][i]) for i in range(3)] for k in  range(3) ]
                           else :
                              exit(1)  
                        elif 'ATOMIC_POSITIONS' in line:
                           positions_found=True
                           i=num+1
                           pos_list = [ [ data_lines[i+j].split()[1],
                                          data_lines[i+j].split()[2],
                                          data_lines[i+j].split()[3]]  for j in range(nat)]
                           pos_unit = line.split('(')[1].split(')')[0]
                           if pos_unit == 'crystal' :
                               pos = [ [0.0,0.0,0.0] for at in range(nat)]
                               for at in range(nat):
                                   for kk in range(3):
                                       for ii in range(3): 
                                           pos[at][kk] += float(pos_list[at][ii]) * float(lat_list[ii][kk])
                           
                           elif pos_unit == 'bohr' :
                               pos = [ [0.0,0.0,0.0] for at in range(nat)]
                               pos = [ [float(pos_list[at][i]) for i in range(3)] for at in range(nat)]
                           else : 
                               exit(2)
                               
                        elif '!    total energy' in line:
                           energy_found = True
                           etot = line.split()[4]
                           #print('Total energy for this scf run {}'.format(etot), flush=True)
                        elif 'Forces acting on atoms' in line:
                           i = num+2
                           forces_list =[ [data_lines[i+j].split()[6],
                                     data_lines[i+j].split()[7],
                                     data_lines[i+j].split()[8]]  for j in range(nat)]
                           forces_list = [ [float(forces_list[k][i]) for i in range(3)] for k in  range(nat) ]
                           #print('Forces {}'.format(forces_list), flush=True)
                        elif 'P= ' in line:
                           i = num+1
                           stress_list =[  [data_lines[i+j].split()[0],
                                     data_lines[i+j].split()[1],
                                     data_lines[i+j].split()[2]]  for j in range(3)]
                           stress_list = [ [float(stress_list[k][i]) for i in range(3)] for k in  range(3) ]
                           #print('Stress in Ry/b^3 {}'.format(stress_list), flush=True)
                    #
                    panna_json['lattice_vectors'] = lat_list
                    panna_json['energy'] = (float(etot),unit_of_energy)
                    for at in  range(nat):
                        panna_json['atoms'].append( [at+1, symbol_list[at], pos[at][:],
                                             forces_list[at][:] ] )     
                    if lstress:
                        panna_json['stress'] = stress_list
                    #
                    panna_json['key'] =  keys_generator.hash_key_v2(panna_json)
                    if addhash :
                        panna_json_name = keys_generator.hash_key_v2(panna_json)
                    else :
                       # panna_json_name =  os.path.basename(pwoutput)+"_"+ str(panna_json['rlx_ind'])
                       # uses underscore _ only to mark the relaxation step to avoid using this in the filename
                       # or edit the following line accordingly
                       #  panna_json_name = panna_json_name.split('_')[0] + "_"+ str(panna_json['rlx_ind'])
                        panna_json_name = panna_json_name_rand +  "_"+ str(panna_json['rlx_ind'])
                    panna_json['starter_key'] =  starter_key
                    if (positions_found and energy_found):
                        with open(outdir.rstrip('/')+"/"+panna_json_name+".example",'w') as outfile:
                            json.dump(panna_json, outfile)


          


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QE xml to PANNA json converter')
    parser.add_argument('-i', '--indir', type=str,
                        help='input directory that holds all the outdirs', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                        help='output directory', required=True)
    parser.add_argument('--addhash', action='store_true', dest='addhash',
                        help='use hash to name jsons')
    args = parser.parse_args()
   
    main(args.indir, args.outdir, args.addhash)
    
    #main(args.indir, args.outdir)



