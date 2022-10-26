###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################

import json 
import numpy as np
import pandas as pd
import argparse
import pylab
import os


# PLOTS A SINGLE COMPONENT OF FINGERPRINT FOR  ALL STRUCTURES IN THE DIRECTORY
# USING PANDAS DATAFRAME
# 



#def main(infil):
def main(indir):
    fp_to_plot= []
    for rt, dirs, files in os.walk(indir):
        for f in files:
            if f.endswith('.fprint') and os.stat(os.path.join(rt,f,)).st_size != 0 :
                fp_to_plot.append(os.path.join(rt,f))
    print(fp_to_plot)
    f = open(fp_to_plot[0], 'r')
    data = json.load(f)
    totdf = pd.DataFrame(data['CC'])
    f.close()
    #print(totdf)
    for infil in fp_to_plot[1:]:   
    #pylab.ion()
        f = open(infil, 'r')
        data = json.load(f)
        df = pd.DataFrame(data['CC'])
        totdf = pd.concat([totdf,df], axis=1)
        f.close()
    #print(totdf.shape)
    totdf.plot()
    pylab.show()

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='makes FingerPrint Plots')
    parser.add_argument(
        '-i', '--indir', type=str, help='indir', required=True)

#       '-i', '--infile', type=str, help='in file', required=True)
   
    
    args = parser.parse_args()
    main(indir = args.indir)


