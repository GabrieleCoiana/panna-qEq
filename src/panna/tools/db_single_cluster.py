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

#Calculate the distances

#user input
#Nspecies =
 
#F_size = int(Nspecies * (Nspecies + 1) * 0.5)

# distance can be calculated in all these dimensions.
# In MoS2 exmaple it is MoMo MoS SS 

all_dist_dict = {}
all_ener_dict = {}
names_dict= {}

def cluster_single(infile, outdir, dist, alpha ): # key, distmat, enerlist, dint, alpha ):
        #PATH MANAGEMENT
        if infile.endswith('.dist'):
            print('Infile  {}'.format(infile) , flush=True)
            #outdir = os.path.dirname(os.path.abspath(outfile))
        else:
            print('Provide .dist files only',flush=True)
        #
        print('Output dir requested {}'.format(outdir), flush=True)
        if os.path.isdir(outdir):
            outdir = os.path.abspath(outdir)
            print('Output dir exists: {}'.format(outdir), flush=True)
        else:
            os.makedirs(outdir)
            outdir = os.path.abspath(outdir)
            print('Output dir created: {}'.format(outdir), flush=True)
        #print user input back
        print('Alpha energy coeff is {}'.format(alpha), flush=True)
        print('Clust calculated at distance {}'.format(dist), flush=True)
        #INPUT MANAGEMENT
        f = os.path.basename(infile)
        key = f[:-5]
        distmat = np.loadtxt(os.path.abspath(infile))
        enerfile = infile[:-5]+'.ener'
        enerlist = np.genfromtxt(os.path.abspath(enerfile), delimiter=" ", autostrip=True, usecols = 2)
        #vollist = np.genfromtxt(os.path.abspath(enerfile), delimiter=" ", autostrip=True, usecols = 1)
        #print(enerlist[250],flush=True)
        #print(enerlist[1250], flush=True)
        ff = open(os.path.abspath(enerfile), 'r')
        #print(ff.name, flush=True)
        lines = ff.readlines()
        names = []
        for line in lines:
                    names.append(line.split(' ')[0])
        ff.close()
        #
        
        nconfig = int(len(distmat))
   
        for i in range(nconfig): 
            for j in range(nconfig):
                #if i%1000 == j%1000 == 0 :
                distmat[i,j] = np.sqrt(distmat[i,j]*distmat[i,j] +  alpha**2 * (abs(enerlist[i] - enerlist[j]))**2)
                # print('distmat before and after are {} {} {} {}'.format(i,j,distmat[i,j],alpha * (abs(enerlist[i] - enerlist[j]))))
        #       distmat[i,j] = alpha * (abs(enerlist[i] - enerlist[j]))
        #print('done updating distmat',flush=True) 
        
        #
        num_clust=[]
        clust_num=[]
        Nd = 0
        for i in range(Nd+1) :
            d = dist
            #assign everyone to its own cluster
            cluster = np.ones(nconfig,dtype=int)
            for j in range(nconfig) :
                cluster[j] = j
            #connection event between clusters - at least one expected
            con_event = 1
            #go through all elements and decide if they connect  
            while con_event > 0  : 
                con_event = 0
                for ii in range(nconfig) :
                    for jj in range(nconfig) :
                         if cluster[ii] != cluster[jj] :
                             if distmat[ii,jj] < d   : 
                                #print('make new cluster with conevent {}'.format(con_event+1),flush=True)
                                con_event +=1
                                #cj = cluster[jj]
                                cluster[ cluster == cluster[jj]] = cluster[ii]
            unique_cluster, unique_inverse, unique_counts = np.unique(cluster, return_inverse=True, return_counts=True)
            #print('distace and total num of clusters {} {}'.format(d,len(unique_counts)), flush=True)
            num_clust.append([d,len(unique_counts)])
            # distance, and the cluster everyone belongs to at that distance
            clust_num.append(np.insert(np.asarray(unique_inverse, dtype=float),0,d))

        ncarray = np.asmatrix(num_clust)
        cnarray = np.asmatrix(clust_num)
        np.savetxt(os.path.join(outdir, str(str(alpha)+'_'+key+'_num_clust.dat')), ncarray)
        np.savetxt(os.path.join(outdir, str(str(alpha)+'_'+key+'_clust_num.dat')), cnarray)
        np.savetxt(os.path.join(outdir, str(key+'_names.dat')), names, fmt='%25s')
        print('Done processing {}'.format(key), flush=True)
        # now prepare the e-v graph with the cluster info?
        # find the distance where there are approx 10 clusters
        #maxc = 10
        #pp = -1
        #nc = -1
        #while nc <= maxc:
        #    pp+=1
        #    nc = ncarray[pp,1]
            #print(nc)
            #print(pp)
        #print( ncarray[pp-1,1] )
        #print( ncarray[pp-1,0] )
        #print( cnarray[pp-1,0] )

        return
                                 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc distances')
    parser.add_argument(
        '-i', '--infile', type=str, help='.dist file', required=True)

    parser.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
   
    parser.add_argument(
        '-dist', '--distance', type=float, help='distance threshold interval', required=True)

    parser.add_argument(
        '-alp', '--alpha',  type=float, help='energy multiplier', required=True)

    args = parser.parse_args()
    cluster_single(infile=args.infile , outdir=args.outdir, dist=args.distance, alpha=args.alpha)


            
