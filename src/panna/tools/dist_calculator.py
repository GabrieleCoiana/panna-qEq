###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import os, sys, getopt
import numpy as np
import json
import argparse
import random
import string
# Calculate the distances
#
# user input:
# indir outdir nmax
# where nmax determines whether distance between all structures or only a random subset
# will be calculated
# Currently SpeciesKeyList is hardcoded - TODO

# distance can be calculated in all the dimensions of the SpeciesKeyList.
# In MoS2 case, for exmaple, the dimensions are MoMo MoS SS 

#The final distances and energies are written to randomly named files with .dist and .ener extensions.

#F_size = int(Nspecies * (Nspecies + 1) * 0.5)

def mychoices(population, k):
    n=len(population)+0.0
    return [population[int(random.random() * n )] for i in range(k)]

dist_dict = {}
dist_array = []
#SpeciesKeyList=["MoMo","MoS","SS"]
SpeciesKeyList=["CC"]
#dist_file = open('mydistfile.dat', mode='w', newline='')
#ener_file = open('myenerfile.dat', mode='w', newline='')

def cosineFPDistance(indir, outdir, nmax):

    if os.path.isdir(outdir):
        outdir = os.path.abspath(outdir)
        print('Output dir exists: {}'.format(outdir), flush=True)
    else:
        os.makedirs(outdir)
        outdir = os.path.abspath(outdir)
        print('Output dir created: {}'.format(outdir), flush=True)


    # choices is too new for python 3.5 :-/
    # dfname = ''.join(random.choices(string.ascii_lowercase, k=6))
    dfname = ''.join(mychoices(string.ascii_lowercase, k=6))
    dist_file = open(os.path.join(outdir,str(dfname + '.dist')), mode='w', newline='')
    ener_file = open(os.path.join(outdir,str(dfname + '.ener')), mode='w', newline='')

    # calculate the FP cosine distance of all structures with each other
    fprintfiles = []
    for rt, dirs, files in os.walk(indir):
        for f in files:
            if f.endswith('.fprint'):
               fprintfiles.append( os.path.join(rt,f))
    if nmax < 1 : 
       nmax = len(fprintfiles)
    print('Structures are {} many , distances to be calc are {} many'.format(len(fprintfiles), nmax), flush=True)
    progress=0
    random.shuffle(fprintfiles)
    fp_rand=fprintfiles[0:nmax]
    for f1 in fp_rand:
        print(f1,flush=True)
        df1=open(f1,'r')
        data1 = json.load(df1)
        for f2 in fp_rand:
            df2 = open (f2,'r')
            data2 = json.load(df2)
            for key in SpeciesKeyList:
                k1=np.array(data1[key])
                k2=np.array(data2[key])
                dist_dict[key] = (1-np.dot(k1,k2)/np.linalg.norm(k1)/np.linalg.norm(k2))*0.5
           
            tot_dist = np.linalg.norm(np.array(list(dist_dict.values())))
            dist_file.write(str(tot_dist)+" ")
            df2.close()
            progress+=1
            if progress%1000 == 0 :
                print ('DoneSteps {} out of {}'.format(progress, nmax*nmax), flush=True)
        dist_file.write('\n')
        ener_file.write(str(os.path.basename(f1))+ " " + str(data1['vol']) + " " + str(data1['energy']) + '\n')
        df1.close()

    dist_file.close()
    ener_file.close()
    return    

            #print( dist_dict[key] for key in SpeciesKeyList )
            
            #print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc distances')
    parser.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)

    parser.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
    
    parser.add_argument(
        '-nm', '--nmax' , type=int, help='total # of structures for FP calc', required=True)



    args = parser.parse_args()
    cosineFPDistance(indir=args.indir , outdir=args.outdir, nmax=args.nmax)

