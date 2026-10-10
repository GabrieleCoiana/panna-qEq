# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:39:55 2026

@author: Gabriele Coiana
"""
import numpy as np
evA3_to_GPa = 160.21766208

def read_panna_metrics(file, plots):
    
    A = np.genfromtxt(file, names=True)
    ylabels = {}
    header = A.dtype.names
    # print(header)
    steps = A['step']
    dic = {}
    B = {}
    Bval = {}
    for i,field in enumerate(header):
        # print(field)
        for p in plots:
            # print(p == 'forces')
            if p == 'energy' or p == 'E':
                if field in ['MAE', 'MAEat', 'MAEat', 'RMSE', 'RMSEat', 'RMSE/at']: # energy
                    B['energy'] =  A[field] * 1000  # to meV / atom
                    dic['energy'] = field
                    ylabels['energy'] = '[meV / atom]'
                elif field in ['Val_MAE', 'Val_MAEat', 'Val_RMSE', 'Val_RMSEat']: 
                    Bval['energy'] =  A[field] * 1000  # to meV / atom
            if p == '%energy':
                if field == 'relRMSE/at': # energy
                    B['%energy'] =  A[field] * 100  # to %
                    dic['%energy'] = field
                    ylabels['energy'] = '[%]'
                elif field == 'Val_%RMSE/at': 
                    Bval['%energy'] =  A[field] * 100  # to %
                
                
            if p == 'forces' or p == 'F':
                if field in ['MAEF', 'RMSEF']: # forces
                    B['forces'] =  A[field] * 1000  # to meV / A
                    dic['forces'] = field
                    ylabels['forces'] = '[meV / A]'
                elif field in ['Val_MAEF', 'Val_RMSEF']: 
                    Bval['forces'] = A[field] * 1000  # to meV / A
            if p == '%forces' or p == '%F':
                if field == 'relRMSEF': # forces
                    B['%forces'] =  A[field] * 100  # to %
                    dic['%forces'] = field
                    ylabels['%forces'] = '[%]'
                elif field == 'Val_relRMSEF': 
                    Bval['%forces'] = A[field] * 100  # to %
                    

            if p == 'stress' or p == 'S':
                if field in ['MAES', 'RMSES']: # stress
                    B['stress'] =  A[field] * evA3_to_GPa  # to GPa
                    dic['stress'] = field
                    ylabels['stress'] = '[GPa]'
                elif field in ['Val_MAES', 'Val_RMSES']: 
                    Bval['stress'] = A[field] * evA3_to_GPa  # to GPa
            if p == '%stress' or p == '%S':
                if field == 'relRMSES': # stress
                    B['%stress'] =  A[field] * 100  # to %
                    dic['%stress'] = field
                    ylabels['%stress'] = '[%]'
                elif field == 'Val_relRMSES': 
                    Bval['%stress'] = A[field] * 100  # to %
                    
            if p == 'charges' or p == 'Q':
                if field in ['MAEQ', 'RMSEQ']: # charges
                    B['charges'] =  A[field] # to e
                    dic['charges'] = field
                    ylabels['charges'] = '[e]'
                elif field in ['Val_MAEQ', 'Val_RMSEQ']: 
                    Bval['charges'] = A[field]  # to e
            if p == '%charges' or p == '%Q':
                if field == 'relRMSEQ': # stress
                    B['%charges'] =  A[field] * 100  # to %
                    dic['%charges'] = field
                    ylabels['%charges'] = '[%]'
                elif field == 'Val_relRMSEQ': 
                    Bval['%charges'] = A[field] * 100  # to %
                    
            if p == 'dipoles' or p == 'D':
                if field in ['MAED', 'RMSED']: # charges
                    B['dipoles'] =  A[field] # to e
                    dic['dipoles'] = field
                    ylabels['dipoles'] = '[e A]'
                elif field in ['Val_MAED', 'Val_RMSED']: 
                    Bval['dipoles'] = A[field]  # to e
            if p == '%dipoles' or p == '%D':
                if field == 'relRMSED': # stress
                    B['%dipoles'] =  A[field] * 100  # to %
                    dic['%dipoles'] = field
                    ylabels['%dipoles'] = '[%]'
                elif field == 'Val_relRMSED': 
                    Bval['%dipoles'] = A[field] * 100  # to %
                    
            if p == field:
                B[field] = A[field]
                dic[field] = field
                if field[0:10] == 'avg_charge':
                    ylabels[field] = '[e]'
                elif field[0:10] == 'avg_sigmas':
                    ylabels[field] = '[Ang]'
                elif field[0:10] == 'avg_alphas':
                    ylabels[field] = '[bohr^3]'
                else:
                    ylabels[field] = ''
                
            elif 'Val_'+p == field:
                Bval[field[4:]] = A[field]
            
                    
            # if p == 'avg_charges':
            #     if field[0:10] == 'avg_charge': # avg charges
            #         try:
            #             B['avg_charges'] #= np.vstack((B['charges'], A[field]))
            #             # dic[-1] = dic[-1] + ' + ' + field
            #         except KeyError:
            #             B['avg_charges'] = A[field]
            #             dic['avg_charges'] = field
            #             ylabels['avg_charges'] = '[e]'
            #         # B['charges'] = A[field]   # to e
            #         # dic.append(field)
            #         # ylabels.append('[e]')
            #         # print(field)
                    
            #     elif field[0:14] == 'Val_avg_charge': 
            #         try:
            #             Bval['avg_charges']# = np.vstack((Bval['charges'], A[field]))
            #         except KeyError:
            #             Bval['avg_charges'] = A[field]
            #         # Bval['charges'] = A[field]  # to e
            #         # done = True
                    
            # if p == 'avg_sigmas':
            #     if field[0:10] == 'avg_sigmas': # avg charges
            #         try:
            #             B['avg_sigmas'] #= np.vstack((B['charges'], A[field]))
            #             # dic[-1] = dic[-1] + ' + ' + field
            #         except KeyError:
            #             B['avg_sigmas'] = A[field]
            #             dic['avg_sigmas'] = field
            #             ylabels['avg_sigmas'] = '[A]'
                    
            #     elif field[0:14] == 'Val_avg_sigmas': 
            #         try:
            #             Bval['avg_sigmas']# = np.vstack((Bval['charges'], A[field]))
            #         except KeyError:
            #             Bval['avg_sigmas'] = A[field]
            
            # if p == 'avg_alphas':
            #     if field[0:10] == 'avg_alphas': # avg charges
            #         try:
            #             B['avg_alphas'] #= np.vstack((B['charges'], A[field]))
            #             # dic[-1] = dic[-1] + ' + ' + field
            #         except KeyError:
            #             B['avg_alphas'] = A[field]
            #             dic['avg_alphas'] = field
            #             ylabels['avg_alphas'] = '[bohr^3]'
                    
            #     elif field[0:14] == 'Val_avg_alphas': 
            #         try:
            #             Bval['avg_alphas']# = np.vstack((Bval['charges'], A[field]))
            #         except KeyError:
            #             Bval['avg_alphas'] = A[field]
                    
            
            
                    
                    
    #     if p == 'MAE' or p == 'MAEat': # then it's energy (per atom)
    #         B.append( A['MAEat'] * 1000 ) # to meV / atom
    #         ylabels.append('[meV / atom]')
    #         dic.append('MAE/atom')
    #         try: # to see if there are validation values
    #             Bval.append(A['Val_MAEat'])
    #         except ValueError:
    #             validation = False
    #     if p == 'MAEF': # then it's forces
    #         B.append( A['MAEF'] * 1000 ) # to meV / A
    #         dic.append('MAEF')
    #         ylabels.append('[meV / A]')
    #         try: # to see if there are validation values
    #             Bval.append(A['Val_MAEF'])
    #         except ValueError:
    #             validation = False
    #     if p == 'MAEQ': # then it's charges 
    #         B.append( A['MAEQ'] ) # e
    #         dic.append('MAEQ')
    #         ylabels.append('[e]')
    #         try: # to see if there are validation values
    #             Bval.append(A['Val_MAEQ'])
    #         except ValueError:
    #             validation = False
    #     if 'avg_charge' in p: # then it's avg charges
    #         for j in range(len(header)):
    #             this_field = header[j]
    #             if this_field[0:10] == 'avg_charge':
    #                 B.append( A[this_field] ) # e
    #                 dic.append(this_field)
    #                 ylabels.append('[e]')
    #                 try: # to see if there are validation values
    #                     Bval.append(A['Val_'+this_field])
    #                 except ValueError:
    #                     validation = False
        
    # B = np.array(B)
    # Bval = np.array(Bval)


    return dic, steps, B, Bval, ylabels
