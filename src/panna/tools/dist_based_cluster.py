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
def cluster_all(indir, outdir, dint, alpha):
    print('Input dir {}'.format(indir) , flush=True)
    print('Output dir requested {}'.format(outdir), flush=True)
    #print('Number of parallel processes {}'.format(nproc), flush=True)
    #p = mp.Pool(nproc)
    if os.path.isdir(outdir):
        outdir = os.path.abspath(outdir)
        print('Output dir exists: {}'.format(outdir), flush=True)
    else:
        os.makedirs(outdir)
        outdir = os.path.abspath(outdir)
        print('Output dir created: {}'.format(outdir), flush=True)

    ####


    for rt, dirs, files in os.walk(indir):
        #read the distance file
        # which has this format
        # dist11 dist12 dist13 ...
        # dist21 dist22 ...
        #read the energy file
        # which is formatted in the same order
        # name1 E1
        # name2 E2
        # name3 E3
        for f in files: 
            if f.endswith('.dist'):
                #g = f.strip('.dist')
                g = f[:-5]
                all_dist_dict[g] = np.loadtxt(os.path.join(rt,f))
                #f=f.strip('.dist')+'.ener'
                f = f[:-5]+'.ener'
                #print(f,flush=True)
                #print(g,flush=True)
                all_ener_dict[g] = np.genfromtxt(os.path.join(rt,f), delimiter=" ", autostrip=True, usecols = 1)
                # 
                ff = open(os.path.join(rt,f), 'r')
                print(ff.name)
                lines = ff.readlines()
                names = []
                for line in lines:
                    names.append(line.split(' ')[0])
                ff.close()
                names_dict[g]=names        
                #print(names)
        print('{} distance files are found'.format(len(all_dist_dict)),flush=True)
        print('{} energy files are found'.format(len(all_ener_dict)),flush=True)
        #print(all_dist_dict, flush=True)
        #print(all_ener_dict, flush=True)

       # p = mp.Pool(nproc)

       # Cluster_single = partial(cluster_single_distfile, key,dists, eners,dint, alpha ) 
       # nn = 0
       # print(int(len(all_dist_dict) /  nproc))

       # while nn <= int(len(all_dist_dict) / nproc)
       #     ml = nn * nproc
       #     mu = nn*nproc + nproc
       #     try: 
                
 
        for key in all_dist_dict :
            dists = all_dist_dict[key]
            eners = all_ener_dict[key]
            #print(eners)
            # dist vs number of clusters
            # structure vs its cluster number
            num_clust, clust_num = cluster_single_distfile(key,dists, eners,dint, alpha)
            ncarray = np.asmatrix(num_clust)
            cnarray = np.asmatrix(clust_num)
            #print(ncarray, flush=True)
            #print(cnarray, flush=True)
            np.savetxt(os.path.join(outdir, str(str(alpha)+'_'+key+'_num_clust.dat')), ncarray)
            np.savetxt(os.path.join(outdir, str(str(alpha)+'_'+key+'_clust_num.dat')), cnarray)
            np.savetxt(os.path.join(outdir, str(key+'_names.dat')), names_dict[key], fmt='%25s')
            print('Done processing {}'.format(key), flush=True)
        return

def cluster_single_distfile( key, distmat, enerlist, dint, alpha ):

       
        #alpha = 0.05

        nconfig = int(len(distmat))
   
        for i in range(nconfig): 
            for j in range(nconfig):
                distmat[i,j] = np.sqrt(distmat[i,j]*distmat[i,j] +  alpha * (abs(enerlist[i] - enerlist[j]))**2)
        #        distmat[i,j] = alpha * (abs(enerlist[i] - enerlist[j]))
        #print('done updating distmat',flush=True) 
        dmin =  np.min(distmat)
        dmax =  np.max(distmat)
        Nd = int((dmax - dmin )/dint)+1
        #
        print('Distance min and max respectively {} {}'.format(dmin,dmax),flush=True)
        #emin = np.min(enerlist)
        #emax = np.max(enerlist)
        #eint = (emax - emin) / (Nd - 1)
        #print('allowed enerfy difference in cluster {}'.format(eint))
        #eint = 1
        #
        num_clust=[]
        clust_num=[]
        for i in range(Nd+1) :
            d = dmax - i*dint
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
                                cj = cluster[jj]
                                cluster[ cluster == cluster[jj]] = cluster[ii]
            unique_cluster, unique_inverse, unique_counts = np.unique(cluster, return_inverse=True, return_counts=True)
            #print('distace and total num of clusters {} {}'.format(d,len(unique_counts)), flush=True)
            num_clust.append([d,len(unique_counts)])
            # distance, and the cluster everyone belongs to at that distance
            clust_num.append(np.insert(np.asarray(unique_inverse, dtype=float),0,d))
        return num_clust, clust_num
                                 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calc distances')
    parser.add_argument(
        '-i', '--indir', type=str, help='in path', required=True)

    parser.add_argument(
        '-o', '--outdir', type=str, help='out path', required=True)
   
    parser.add_argument(
        '-dint', '--dinterval', type=float, help='distance threshold interval', required=True)

    parser.add_argument(
        '-alp', '--alpha',  type=float, help='energy multiplier', required=True)

    args = parser.parse_args()
    cluster_all(indir=args.indir , outdir=args.outdir, dint=args.dinterval, alpha=args.alpha)


            
