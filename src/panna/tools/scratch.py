# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:17:06 2025

@author: Gabriele Coiana
"""
# this is needed for the Ewald summation 
# TODO: comment better
# t1 = time.time()

# if True: #'acc_factor' in parameters.keys():
#     acc_factor, real_space_cutoff, rec_space_cutoff, gaussian_smearing = parameters['ewald_input_params']
    
#     linsys_masks = create_masks(batch['nats'], parameters['at_mod'])
#     alpha = None#1/(np.sqrt(2)*gaussian_smearing) if not np.isnan(gaussian_smearing) else np.nan
#     real_space_cutoff = None
#     rec_space_cutoff = None
#     ewald_params = set_ewald_parameters_for_batch(batch['pos'], batch['latvecs'], acc_factor, Rc=real_space_cutoff, alpha=alpha, Kc_sc=rec_space_cutoff, verbose=False, q=None)
    
#     # print(parameters['ewald_params'],'////////')
    
#     parameters['ewald_params'] = ewald_params
#     parameters['linsys_masks'] = linsys_masks
    
# t2 = time.time()
# print(t2-t1, ' sec for ewald params and linsys masks')