# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:08:32 2024

@author: Gabriele Coiana
"""
import numpy as np
import itertools
import time
# from .pbc_jax import get_rij, get_rij_qq, get_Scells_images
from .pbc_jax import get_Scells_images, get_indices_and_distances_of_pairs, get_indices_and_distances_of_pairs_forloop, get_indices_and_distances_of_pairs_claude

import tracemalloc
# from scipy.special import erf, erfc
from jax.scipy.special import erf, erfc
# import scipy.constants as constants
import jax
# enable_64 = True
# print('jax_enable_x64 ', enable_64)
# jax.config.update("jax_enable_x64", enable_64) # in case you want float64
import jax.numpy as jnp
from functools import partial

e_charge = 1.60217662e-19 
epsilon_0 = 8.85418782e-12 #F/m unit
CONV_FACT = 14.39964547842567 #1e10 * constants.e / (4 * np.pi * constants.epsilon_0)


def Ewald_summation(
             coords,
             lattice_vectors,
             sigmas,
             alpha_screen,
             realspace_cutoff,
             recspace_cutoff, # unused
             input_realspace=jnp.zeros(1),
             input_recspace=jnp.zeros((1,1,1)),
             verbose = False,
             mode = 'kernels',
             realmode = 'vect',
             rijs_provided = False,
             q=0
             ):
    '''
    '''    
    if verbose:
        print('Ewald sum for Gaussian charges')
        print('\tN of atoms', len(coords))
        print('\tN of images in real space', (len(input_realspace) * 2 + 1)**3)
        print('\tN/2 -1 of g-vecs in rec space', input_recspace.shape)
        print('\tstd of Gaussian screening', *1/(jnp.sqrt(2) * alpha_screen))

    output = {}
    
    # convert to alpha convention
    alphas = 1/(jnp.sqrt(2)* jnp.array(sigmas))
    
    if rijs_provided == False:
        if realmode != 'vect':
            print('EWALDGAUSSqq - Computing rijs in JAX mode')
            # for real space sum
            # max_Npairs = 1000000#estimate_maxNpairs_analytically(len(coords), volume, realspace_cutoff)
            Scells = get_Scells_images(input_realspace, lattice_vectors, coords)
            indices_and_rijs = get_indices_and_distances_of_pairs(coords, Scells, realspace_cutoff, within_cutoff=False)#, max_Npairs)
    else:
        print('EWALDGAUSSqq - unpacking rijs computed in eager mode')
        indices_and_rijs = rijs_provided
        

    if mode == 'kernels': # compute only energy kernels
        # real space Ewald summation
        real_en_kernel, real_force_kernel = 0, 0
        if len(input_realspace) != 0: # do real space sum
            real_en_kernel = realewald_qq_jax_vect(alphas, realspace_cutoff, coords, lattice_vectors, alpha_screen, input_realspace)
            
        # rec space Ewald summation
        rec_en_kernel = recewald_qq_jax_vect(input_recspace,alphas, alpha_screen, coords, lattice_vectors)        

        output['el_energy_kernel'] = rec_en_kernel + real_en_kernel
        # output['el_energy_kernel_real'] = real_en_kernel
        # output['el_energy_kernel_rec'] = rec_en_kernel
        # output['self'] = self_qq
        

        # output['el_force_kernel'] =  rec_force_kernel + real_force_kernel
        # output['el_force_kernel_real'] = real_force_kernel
        # output['el_force_kernel_rec'] = rec_force_kernel
    
    else: # compute forces and stress from (q, d)
        if realmode == 'vect':
            real_force, real_virial = real_forces_and_stress_vect(
                                                alphas,
                                                alpha_screen, 
                                                realspace_cutoff, 
                                                input_realspace, 
                                                coords, 
                                                lattice_vectors, 
                                                q)
        else:
            real_force, real_virial = real_forces_and_stress_nn(
                                                alphas,
                                                alpha_screen, 
                                                realspace_cutoff, 
                                                input_realspace, 
                                                indices_and_rijs,
                                                q)

        rec_force, rec_virial = rec_forces_and_stress(
                                            alphas,
                                            input_recspace,
                                            alpha_screen, 
                                            coords, 
                                            lattice_vectors, 
                                            q)

        output['forces'] = real_force + rec_force
        output['virial'] = real_virial + rec_virial
        
    
    
    # N = coords.shape[0]
    # output['linsys_bvector'] = jnp.zeros(N)
    # output['linsys_masks'] = mask
    
    return output



@jax.jit
def rec_forces_and_stress(alphas,
                          input_recspace,
                          alpha_screen,
                          coords,
                          lattice_vectors,
                          q):
    print('EWALDGAUSSqq - Rec forces and stress')
    # coords = jnp.where(jnp.isinf(coords), 1e8, coords)
    # # sending α_i to zero for padded atoms
    # alphas = jnp.where(alphas<1e-9,1e-6,alphas)
    
    # volume = jnp.linalg.det(lattice_vectors)
    alphai = alphas[:,None] # (N, 1)
    
    # structure properties
    volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)

    # [i,j,k] combinations of G-grid i,j,k integers
    comb = kvectors_for_ewald(input_recspace)

    # G = Hk @ [i,j,k]
    G_vects = jnp.dot(comb , recip_lattice_vectors)
    G_norms = jnp.linalg.norm(G_vects, axis=1)
    # this is not allowed inside a jitted function!
    # mask = G_norms <= Gcut # radial cutoff
    # G_vects = G_vects[mask]
    # G_norms = G_norms[mask]
    # jax.debug.print('Num of Gvecs {}', G_vects.shape)

    # e^iGR
    eiGr = jnp.where(~jnp.isnan(jnp.exp(-1j * coords @ G_vects.T)), 
                      jnp.exp(-1j * coords @ G_vects.T), 
                      0)# (N, NG)
    # 4pi/(V k^2) e^(-k^2/4η^2)
    ew_factor = 4*jnp.pi / volume \
        * 1/G_norms**2 * jnp.exp(-G_norms**2/(4*alpha_screen**2)) # (NG)
    # with Gaussian charges we have a e^{- G^2 / (4 α_i^2)} in the struct. factors
    gauss_factor = jnp.exp(-G_norms**2 / (4*alphai**2) )   # (N, NG)
        
    # S(G) structure factor = \sum_i q_i e^{- i G R_i} e^{- G^2 / (4 α_i^2)}
    S = jnp.sum(q[:,None] * gauss_factor * eiGr, axis=0) # (NG) 
    # |S(G)|^2 = S(G)* S(G)
    S2 = S * S.conj() # (NG)
    
    
    
    
    # dS*(G) / d r_i^α = q_i (i Gα) e^{i G r_i}
    SxiG = q[:,None] * eiGr.conj() * 1j*G_vects.T[:,None,:] * gauss_factor[None] # (3, N, NG)
    

    # force
    # dS*/dr S + c.c. = 2 Re[dS*/dr S]
    forceG = 2*(SxiG * S).real * ew_factor.T # (3, N, NG)
    # force = 1/2 \sum_G ew(G) d/dx Stot* Stot
    forces = -jnp.sum(forceG, axis=-1).T # (N,3)

    # stress
    S2 = S2[:,None,None] # (NG, 1, 1)
    ew_factor = ew_factor[:,None,None] # (NG, 1, 1)
    # stress: δ_{αβ} - 2 Gα Gβ (1/4η^2 + 1/G^2)
    # stress factor: (1/4η^2 + 1/G^2)
    stress_factor = 1/G_norms**2 + 1/(4*alpha_screen**2) # (NG)
    stress_factor = stress_factor[:,None,None]    # (NG, 1, 1)
    # outer product: δ_{αβ} - 2 Gα Gβ 
    GaGb = G_vects[:,:,None] * G_vects[:,None,:] # (NG, 3, 3)
    delta_ab = jnp.eye(3)[None] # (1, 3, 3)



    # STRESS
    virial_1 = S2 * ew_factor * (delta_ab - 2 * GaGb * stress_factor)   # (NG, 3, 3)
    # \sum_G
    virial_1 = jnp.sum(virial_1, axis=0).real  # (3,3)

    
    
    # extra stress from derivative of atom-dependent Gaussian factors
    # the Gauss factor term gets also derived, d gauss_iG / d eps_ab
    dgauss_dG2 = -eiGr * gauss_factor * 1/(4*alphai**2) # (N, NG)
    # dS / d eps_ab from gauss_factor
    dS_gauss = jnp.sum(q[:, None] * dgauss_dG2, axis=0) # (NG)
    
    
    
    virial_3 = -2*(dS_gauss.conj() * S).real[:,None,None] * ew_factor * (-2 *GaGb) # (NG, 3, 3)
    virial_3 = jnp.sum(virial_3, axis=0) # (3,3)

    
    
    virial = virial_1  + virial_3

    
    return forces*CONV_FACT, virial*CONV_FACT


@jax.jit
def real_forces_and_stress_nn(alphas,
                           alpha_screen,
                          rcut,
                          input_realspace, 
                          indices_and_rijs,
                          q):
    print('EWALDGAUSSqq - Real forces and stress NN')
    # # # setting padded atoms' positions to very far away (not to inf, because otherwise rij_diag does 0 * inf)
    # alphas = jnp.where(jnp.isinf(coords[:,0]), 1e-6, alphas)
    # coords = jnp.where(jnp.isinf(coords), 1e9, coords)
    
    # Scells = get_Scells_images(input_realspace, lattice_vectors, coords)
    eye = jnp.eye(3) # this is δ_{γα}
    N = q.size
    
    # # # get indices and rij's. If you uncomment this it'll create NN list
    # indi, indj, rij, xyzij = get_indices_and_distances_of_pairs(coords, Scells, rcut)
    # # ... but actually these are given by e.g. ASE neighbour_list()
    indi, indj, rij, xyzij = indices_and_rijs # unpack indices and rijs
    # ind_row_B, ind_col_B, ind_row_C, ind_col_C = get_indices_of_pairs(indi, indj)
    
    alphai = alphas[indi] # (Npairs,)
    alphaj = alphas[indj] # (Npairs,)
    
    
    
    # the Ewald real sum is ∫∫ gi(r) gj(r') erfc(α_screen |r-r') / |r-r'|
    # = ∫∫ gi(r) gj(r') [1/|r-r'| - erf / |r-r'|] = term1 + term2
    # where term1 is the electrostatic energy of interaction between two Gaussians
    # and term2 is the interaction screended by a Gaussian screening 
    
    # The electrostatic energy of interaction between two Gaussians is q_i q_j erf(μ_ij R_ij) / R_ij
    # where μ_ij = sqrt( μ_i^2 μ_j^2 / (μ_i^2 + μ_j^2) )
    one_over_mij2 = 1/alphai**2 + 1/alphaj**2
    mij = 1/jnp.sqrt(one_over_mij2)   # (Npairs,)
    one_over_csiij2 = one_over_mij2 + 1/alpha_screen**2
    csiij = jnp.sqrt(1/one_over_csiij2)
    
    # forces qq
    qiqj = q[indi,None] * q[indj,None]   # (Npairs, 1)    
    # ∫∫gi(r) gj(r') erfc(α|r-r'|)/|r-r'| = ∫∫gi(r) gj(r') [1 - erf(α|r-r'|)]/|r-r'|
    force_qq_pairs = qiqj * (-(D(rij, mij) - D(rij, csiij) )[:,None]) * xyzij  # (Npairs, 3)
    
    
    
    # F_i = \sum_j f_{ij}
    dimnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(0,),scatter_dims_to_operand_dims=(0,))
    forces = -jax.lax.scatter_add(jnp.zeros((N,3)),
                            indj[:,None],force_qq_pairs,
                            dimnums,mode='drop')
    
    # W_{α β} = 1/2*\sum_{ij} F^α_{ij} r^β_{ij}
    virial = .5 * jnp.einsum('ix, iy -> xy', force_qq_pairs, xyzij)   # (3,3)
    # virial_qq = .5 * jnp.einsum('ix, iy -> xy', force_qq_pairs, xyzij)   # (3,3) 

    return forces*CONV_FACT, virial*CONV_FACT


@jax.jit
def real_forces_and_stress_vect(alphas,
                           alpha_screen,
                          rcut,
                          input_realspace, 
                          coords,
                          lattice_vectors, 
                          q):
    print('EWALDGAUSSqq - Real forces and stress vect')
    N = len(coords)
    # setting padded atoms' positions to very far away (not to inf, because otherwise rij_diag does 0 * inf)
    alphas = jnp.where(jnp.isinf(coords[:,0]), 1e-9, alphas)
    coords = jnp.where(jnp.isinf(coords), 1e9, coords)
    
    # supercell's images creation
    Scells = get_Scells_images(input_realspace, lattice_vectors, coords)
    # was
    # xyzij = coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:] # (Nimages, N, N, 3 )
    # xyzij = jnp.where(~jnp.isnan(coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:]),
    #           coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:],
    #           0)   
    xyzij = coords[None,  :, None] - Scells[:,None,:] # (Nimages, N, N, 3 )
    rij = jnp.linalg.norm(xyzij, axis=-1) # (Nimages, N, N)
    rij_diag = rij[:,:,None,:,None] * jnp.eye(3)[jnp.newaxis, jnp.newaxis,:,jnp.newaxis] # (Nimages, N, 3, N, 3)
    
    # remove rii numbers 
    rij = jnp.where(rij < 1e-6, 1e18, rij) # (Nimages, N, N)
    rij = jnp.where(rij > rcut, 1e18, rij) # (Nimages, N, N)
    # remove rii numbers AND other zeros
    rij_diag = jnp.where(rij_diag < 1e-5, 1e18, rij_diag) # (Nimages, N, 3, N, 3)
    rij_diag = jnp.where(rij_diag > rcut, 1e18, rij_diag) # (Nimages, N, 3, N, 3)
    # # rij = jnp.where(rij > Rc-0.1, jnp.inf, rij)
    
    # the Ewald real sum is ∫∫ gi(r) gj(r') erfc(α_screen |r-r') / |r-r'|
    # = ∫∫ gi(r) gj(r') [1/|r-r'| - erf / |r-r'|] = term1 + term2
    # where term1 is the electrostatic energy of interaction between two Gaussians
    # and term2 is the interaction screended by a screening Gaussian

    # The electrostatic energy of interaction between two Gaussians is q_i q_j erf(μ_ij R_ij) / R_ij
    # where μ_ij = sqrt( μ_i^2 μ_j^2 / (μ_i^2 + μ_j^2) )
    one_over_mij2 = 1/alphas[:,None]**2 + 1/alphas[None]**2
    mij = 1/jnp.sqrt(one_over_mij2)   # np.sqrt( alphas[:,None]**2 * alphas[None,:]**2 / (alphas[:,None]**2 + alphas[None,:]**2) ) # (N, N)
    
    one_over_csiij2 = one_over_mij2 + 1/alpha_screen**2
    # jax.debug.print('aaaaaa {}', 1/alphas**2)
    # jax.debug.print('aaaaaa {}', 1/one_over_csiij2)
    csiij = jnp.sqrt(1/one_over_csiij2)
    # jax.debug.print('ccc {}', csiij)
    
    
    # qd kernel
    # dE/dxk = dE/dRij * dRij/dxk
    # dRij / dxk = xij / Rij * (\delta_{ik} - \delta_{jk})
    # 1/Rij * dE/dRij = -erf(μRij)/Rij^3 + 2μ/sqrt(pi) e^{-μ^2 Rij^2} /Rij^2 = D(Rij, μ)
    fqq_kernel_ij = -(D(rij, mij[None]) - D(rij, csiij[None]) )[:,:,:,None] * xyzij # (Nimages, N, N, 3)
    # jax.debug.print('fqq {}', fqq_kernel_ij.shape)
      
    # force qq
    force_qq_kernel = jnp.sum( fqq_kernel_ij , axis=0) 
    forces = jnp.einsum('i, ijx, j -> ix', q, force_qq_kernel,q) 
    
        
    # stress
    forces_qq_pairs = jnp.einsum('i, Iijx, j -> Iijx ', q, fqq_kernel_ij, q)     # (NI, N, N, 3)
    # W_{α β} = 1/2*\sum_{ij} F^α_{ij} r^β_{ij}
    virial = .5 * jnp.einsum('Iijx, Iijy -> xy', forces_qq_pairs, xyzij)   # (3,3)
    # virial = jnp.zeros((3,3))
    return forces*CONV_FACT, virial*CONV_FACT


@jax.jit
def realewald_qq_jax_vect(alphas,
                          rcut,
                      coords,
                      lattice_vectors,
                      alpha_screen,
                      input_realspace,
                      compute_forces=True):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in angstrom
        gaussian_width: gaussian width, array of atomic gussian width
        coords : atomic coordinates
    Return:
        real space contribution to energy
    '''
    print('EWALDGAUSSqq - Kernels: Real space sum, vectorized')
    
    # setting padded atoms' positions to very far away (not to inf, because otherwise rij_diag does 0 * inf)
    alphas = jnp.where(jnp.isinf(coords[:,0]), 1e-9, alphas)
    coords = jnp.where(jnp.isinf(coords), 1e19, coords)
    
    # supercell's images creation
    Scells = get_Scells_images(input_realspace, lattice_vectors, coords)

    # t1 = time.time()
    xyzij = coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:] 
    rij = jnp.linalg.norm(xyzij, axis=-1)
    rij = jnp.where(rij == 0, 1e19, rij)
    rij = jnp.where(rij > rcut, 1e19, rij)
    
    # the Ewald real sum is ∫∫ gi(r) gj(r') erfc(α_screen |r-r') / |r-r'|
    # = ∫∫ gi(r) gj(r') [1/|r-r'| - erf / |r-r'|] = term1 + term2
    # where term1 is the electrostatic energy of interaction between two Gaussians
    # and term2 is the interaction screended by a screening Gaussian

    # The electrostatic energy of interaction between two Gaussians is q_i q_j erf(μ_ij R_ij) / R_ij
    # where μ_ij = sqrt( μ_i^2 μ_j^2 / (μ_i^2 + μ_j^2) )
    one_over_mij2 = 1/alphas[:,None]**2 + 1/alphas[None]**2
    mij = 1/jnp.sqrt(one_over_mij2)   # np.sqrt( alphas[:,None]**2 * alphas[None,:]**2 / (alphas[:,None]**2 + alphas[None,:]**2) ) # (N, N)
    
    one_over_csiij2 = one_over_mij2 + 1/alpha_screen**2
    csiij = jnp.sqrt(1/one_over_csiij2)
    
    # ∫∫gi(r) gj(r') erfc(α|r-r'|)/|r-r'| = ∫∫gi(r) gj(r') [1 - erf(α|r-r'|)]/|r-r'|
    # and summing over all images
    real_qq_kernel = jnp.sum( ( erf(mij * rij) - erf(csiij * rij) ) / rij, axis=0) * CONV_FACT

    # # force
    # # dE/dxk = dE/dRij * dRij/dxk
    # # dRij / dxk = xij / Rij * (\delta_{ik} - \delta_{jk})
    # # 1/Rij * dE/dRij = -erf(μRij)/Rij^3 + 2μ/sqrt(pi) e^{-μ^2 Rij^2} /Rij^2 = D(Rij, μ)
    # kernel_ij = -(D(rij, mij[None]) - D(rij, csiij[None]) )[:,:,:,None]
    # force_qq_kernel = jnp.sum( kernel_ij * xyzij , axis=0) * CONV_FACT
    
    # # point_charge_force_qq_kernel = jnp.sum(B(rij[:,:,:,jnp.newaxis], alpha_screen)*xyzij, axis=0) * CONV_FACT
    
    
    # WARNING: returning jnp.nan_to_num allows to use this function as is even in
    #       the case of atoms padding. However, it is unsafe in the case where, e.g.,
    #       you want to debug proper NaNs which might be happening if parameters
    #       like alpha get too large and give inf / inf forms.
    return real_qq_kernel#, jnp.nan_to_num(force_qq_kernel)



  
@jax.jit
def recewald_qq_jax_vect(input_recspace, 
                alphas,
                alpha_screen,
                coords,
                lattice_vectors,
                compute_forces=False):
    print('EWALDGAUSSqq - Kernels: Reciprocal space sum, vectorized Stefanos version')
    # create grid of k-points
    comb = kvectors_for_ewald(input_recspace)
    
    # structure properties
    volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)
    
    N = len(coords)
    # sending α_i to zero for padded atoms
    alphas = jnp.where(alphas<1e-19, 1e-9, alphas)
    alphai = alphas[:,jnp.newaxis]
    # C = (4.* jnp.pi / volume)

    k_vects = jnp.dot(comb , recip_lattice_vectors)
    k_norms = jnp.linalg.norm(k_vects, axis=1)
    
    
    # # e^ikR was
    # eikRi = jnp.exp(-1j * coords @ k_vects.T) # (N, NG)
    # e^ikR for right atoms, 0 for padded atoms 
    eikRi = jnp.where(~jnp.isnan(jnp.exp(-1j * coords @ k_vects.T)), 
                      jnp.exp(-1j * coords @ k_vects.T), 
                      0)# (N, NG)
    
    # 4pi/V k^2 e^(-k^2/4α^2)
    sqrt_ew_factor = jnp.sqrt(4*jnp.pi / volume)/k_norms * jnp.exp(-k_norms**2/(8*alpha_screen**2)) # (NG)
    
    # e^{-k^2 / 4 α_i^2} S(k) sqrt(ewald(G))
    Aik = jnp.exp(-k_norms**2/(4*alphai**2)) * eikRi * sqrt_ew_factor # (N, NG)
    
    # qq
    rec_qq_kernel = 2 *(Aik @ Aik.conj().T).real * CONV_FACT
    
    # self term
    # 1/mi^2 = 1/αi^2 + 1/αi^2
    one_over_mi2 = 2/alphas**2
    mi = jnp.sqrt(1/one_over_mi2) # (N)
    # 1/csi^2 = 1/αi^2 + 1/αi^2 + 1/\eta^2
    one_over_csi2 = 2/alphas**2 + 1/alpha_screen**2
    csi = jnp.sqrt(1/one_over_csi2) # (N)
    self_qq = 2/jnp.sqrt(jnp.pi) * (mi - csi) * jnp.eye(N) * CONV_FACT
    
    
    # # force
    # Bxik = Aik[jnp.newaxis] * 1j*k_vects.T[:,jnp.newaxis,:] # (3, N, NG)
    # op = Bxik.conj() @ Aik.T # (3, N, N)
    # force_qq_kernel = -2*op.real.T #( op + op.conj() ).T.real
    # force_qq_kernel *= CONV_FACT

    # WARNING: see warning above
    return rec_qq_kernel+self_qq#, jnp.nan_to_num(force_qq_kernel)


  
# @jax.jit
# def recewald_qq_jax_vect_gaussians(
#                 alphas,
#                 coords,
#                 lattice_vectors,
#                 input_recspace):
#     print('Reciprocal space sum for Gaussians, Stefanos version')
#     N = len(coords)
#     # structure properties
#     volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)
#     C = (4.* jnp.pi / volume)
#     alphas = alphas[:,jnp.newaxis]
    
#     comb = kvectors_for_ewald(input_recspace)

#     k_vects = jnp.dot(comb , recip_lattice_vectors)
#     k_norms = jnp.linalg.norm(k_vects, axis=1)

#     # e^ikR
#     eikr = jnp.exp(1j * coords @ k_vects.T) 
#     # sin = 2 * jnp.sin(coords @ k_vects.T)

#     # 4pi/V k^2 e^(-k^2/4alpha^2)
#     ew_factor = 1/k_norms * jnp.exp(-k_norms**2/(4*alphas**2))
#     Aik = eikr * ew_factor
    
#     # qq
#     qq_kernel = 2 *(Aik @ Aik.conj().T).real
#     qq_kernel *= C * CONV_FACT 
    
#     # # force
#     # ew_factor_force = ew_factor * 1j*k_vects.T
#     # Fik = ew_factor_force[:,jnp.newaxis,:] * eikr[jnp.newaxis,:,:] 
#     # force_qq_kernel = 2 * (Fik @ Aik.T.conj()).T.real
    

#     # force
#     Bxik = Aik[jnp.newaxis] * 1j*k_vects.T[:,jnp.newaxis,:]
#     op = Bxik.conj() @ Aik.T
#     force_qq_kernel = ( op + op.conj() ).T.real
#     force_qq_kernel *= C * CONV_FACT
    
#     # # self terms
#     # self_qq = - 2* jnp.sqrt(alphas**2 / jnp.pi) * jnp.eye(N) * CONV_FACT

#     return ew_factor, jnp.nan_to_num(qq_kernel), jnp.nan_to_num(force_qq_kernel)
    

@jax.jit
def recewald_qq_jax_vect_yusuf(comb, 
                alpha,
                volume, coords,
                recip_lattice_vectors,
                compute_forces=False):
    print('Reciprocal space sum, Yusufs version')
    
    N = len(coords)
    rij = coords[:,np.newaxis] - coords
    k_vects = jnp.dot(comb , recip_lattice_vectors)
    k_norms = jnp.linalg.norm(k_vects, axis=1)
    
    # e^ikR
    cos = 2 * jnp.cos(rij @ k_vects.T)
    sin = 2 * jnp.sin(rij @ k_vects.T)
    # 4pi/V k^2 e^(-k^2/4alpha^2)
    ew_factor = 1/k_norms**2 * jnp.exp(-k_norms**2/(4*alpha**2)) * (4.* jnp.pi / volume)
    
    
    # qq
    qq_kernel = jnp.sum( cos * ew_factor, axis=2)
    qq_kernel *= CONV_FACT 
    

    # qd
    qd_tmp = sin * ew_factor
    force_qq_kernel = jnp.matmul(qd_tmp, k_vects) * CONV_FACT 
    
    
    
    # self terms
    self_qq = - 2* jnp.sqrt(alpha**2 / jnp.pi) * jnp.eye(N) * CONV_FACT
    

    
    return jnp.nan_to_num(qq_kernel+self_qq), jnp.nan_to_num(force_qq_kernel)
 

@jax.jit
def recewald_qq_jax(comb, 
                alpha,
                volume, coords,
                recip_lattice_vectors,
                compute_forces=False):
    print('Reciprocal space sum, for loop')
    N = len(coords)
    rij = coords[:,np.newaxis] - coords
    k_vects = jnp.dot(comb , recip_lattice_vectors)
    k_norms = jnp.linalg.norm(k_vects, axis=1)
    

    qq_kernel = jnp.zeros((N,N))
    qd_kernel = jnp.zeros((N,N,3))
    
    for k in range(len(k_norms)):
        this_k = k_vects[k]
        this_k_norm = k_norms[k]
        
        cosk = 2 * jnp.cos(jnp.dot(rij , this_k)) #np.cos( np.sum(rij * this_k, axis=-1) )
        sink = 2 * jnp.sin(jnp.dot(rij , this_k)) #np.sin( np.sum(rij * this_k, axis=-1) )
        
        # 4pi/V k^2 e^(-k^2/4alpha^2)
        ew_factork = 1/this_k_norm**2 * jnp.exp(-this_k_norm**2/(4*alpha**2)) * (4.* np.pi / volume)
    
        # qq
        qq_kernel += cosk * ew_factork * CONV_FACT
        
        # qd
        qd_tmp = (sink * ew_factork)[:,:,jnp.newaxis] * this_k[jnp.newaxis, jnp.newaxis,:]
        qd_kernel += qd_tmp * CONV_FACT   #.reshape((N,-1))
        

    
    # self terms
    self_qq = - 2* jnp.sqrt(alpha**2 / jnp.pi) * jnp.eye(N) * CONV_FACT
    
    # force_qq_kernel = qd_kernel 
    # force_qd_kernel = dd_kernel 
    
    # A = jnp.zeros((N+3*N, N+3*N))
    # A = A.at[0:N,0:N].set(qq_kernel) # qq
    # A = A.at[0:N, N:].set(qd_kernel.reshape((N,3*N))) # qd
    # A = A.at[N:, 0:N].set(qd_kernel.reshape((N,3*N)).T) # dq
    # A = A.at[N:,N:].set(dd_kernel)  # dd
    
    
    return qq_kernel+self_qq, qd_kernel


def setup_matrix_for_linsys(A, N):
    N = N.astype(int)
    # N = jnp.shape(A)[0]
    print(N, 'actual number of atoms')
    N = jnp.where(N ==0)
    
    
    # print(M,L,'òòò')
    # # shape = jnp.shape(A)[0]
    # a = jnp.arange(M)
    # print(a)
    
    B = jnp.ones((N+1,N+1))
    B = B.at[0:N,0:N].set(A[0:N,0:N])
    B = B.at[N,N].set(0)
        
    # shape_padded = jnp.shape(A)[0]
    # C = jnp.eye(shape_padded+1)
    # C = C.at[0:M+1, 0:M+1].set(B)
    return B

def setup_matrix(qq_kernel, qd_kernel, dd_kernel):
    N = len(qq_kernel)
    A = np.zeros((N+3*N, N+3*N))
    A[0:N,0:N] = qq_kernel # qq
    A[0:N, N:] = qd_kernel # qd
    A[N:, 0:N] = qd_kernel.T # dq
    A[N:,N:]   = dd_kernel  # dd
    return A


def kvectors_for_ewald_was(Kc_sc):
    kcx, kcy, kcz = jnp.shape(Kc_sc)
    l_list = jnp.arange(-kcx, kcx + 1)
    m_list = jnp.arange(-kcy, kcy + 1)
    n_list = jnp.arange(-kcz, kcz + 1)
    replicas = [x for x in itertools.product(l_list, m_list, n_list)]
    # 0 0 0 is removed, g0ik = 0
    comb = jnp.array(replicas[:int(len(replicas)/2)])
    # comb = jnp.array(replicas)
    return comb 

def kvectors_for_ewald(Kc_sc):
    # same as above, but without for loops
    kcx, kcy, kcz = jnp.shape(Kc_sc)
    
    gx, gy, gz = jnp.meshgrid(jnp.arange(-kcx, kcx + 1),
                              jnp.arange(-kcy, kcy + 1),
                              jnp.arange(-kcz, kcz + 1),
                              indexing='ij')
    comb = jnp.stack((gx, gy, gz), axis=-1).reshape((-1,3))
    return comb[0:comb.shape[0] // 2]

def get_Scells_images(inp, latvec, R0):
    xl = jnp.shape(inp)[0]
    
    x1 = jnp.arange(-xl,xl+1, dtype=int)
    comb = jnp.array(jnp.meshgrid(x1,x1,x1)).T.reshape(-1,3)
    comb = jnp.vstack((comb, jnp.array([1e18, 1e18, 1e18])))
    
    displ = comb @ latvec # rows of latvec are lattice vectors #  jnp.dot(latvec, comb.T).T
    displ = displ[:,jnp.newaxis,:]
    Scells = displ + R0
    return Scells


def erf_over_r(r, alpha):
    erf_over_r = erf(alpha*r) / r
    return erf_over_r

def A(r, alpha):
    erfc_over_r  = erfc(alpha*r)/r
    return erfc_over_r

def B(r, alpha):
    erfc_over_r3 = erfc(alpha*r)/r**3 
    gauss_term = 2*alpha / jnp.sqrt(jnp.pi) * jnp.exp(-alpha**2*r**2) /r**2
    return erfc_over_r3 + gauss_term

def C(r, alpha):
    three_erfc_over_r5 = 3*erfc(alpha*r)/r**5
    other_term = 2 * alpha / jnp.sqrt(jnp.pi) * (2*alpha**2 + 3/r**2) * jnp.exp(-alpha**2*r**2)/r**2
    return three_erfc_over_r5 + other_term

def Bprime(r, alpha):
    three_erfc_over_r4 = -3*erfc(alpha*r)/r**4
    gauss_term1 = -6*alpha/jnp.sqrt(jnp.pi) * jnp.exp(-alpha**2*r**2)/r**3
    gauss_term2 = -4*alpha**3/jnp.sqrt(jnp.pi) *jnp.exp(-alpha**2*r**2)/r
    return three_erfc_over_r4 + gauss_term1 + gauss_term2

def Cprime(r, alpha):
    erfc_over_r5 = -15*erfc(alpha*r)/r**6 
    gauss_term = -2*alpha/jnp.sqrt(jnp.pi)*jnp.exp(-alpha**2*r**2) * (15/r**5 + 10*alpha**2/r**3 + 4*alpha**4/r)
    return erfc_over_r5 + gauss_term

def D(r, alpha):
    erf_over_r3 = erf(alpha*r)/r**3 
    gauss_term = 2*alpha / jnp.sqrt(jnp.pi) * jnp.exp(-alpha**2*r**2) /r**2
    return -erf_over_r3 + gauss_term

def real_space_term(alpha,
                      lattice_vectors,
                      coords,
                      Rc,
                      compute_forces=False,
                      gabri=False):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in angstrom
        gaussian_width: gaussian width, array of atomic gussian width
        coords : atomic coordinates
    Return:
        real space contribution to energy
    '''
    print('Real space sum')
    N = len(coords)
    
    t1 = time.time()
    # rij_qq, xyzij_qd, rij_qd, xyzij_dd, rij_dd_diag, rij_dd = get_rij(coords, lattice_vectors, Rc)
    rij_qq, xyzij_qd, rij_qd, xyzij_dd_outer, rij_dd_diag, rij_dd_full, xyzij_dd, xyzij_dd_diag, xyzij_dd_outer_unitvec, xyzij_dd_derivatives = get_rij(coords, lattice_vectors, Rc)
    t2 = time.time()
    print(t2-t1, 'calculating rij')
    
    qq_kernel = np.sum(A(rij_qq, alpha), axis=-1) * CONV_FACT
    qd_kernel = np.sum(B(rij_qd, alpha)*xyzij_qd, axis=-1) * CONV_FACT
    qd_kernel = qd_kernel.reshape((N, -1))
    dd_kernel = np.sum(B(rij_dd_diag, alpha) - C(rij_dd_full, alpha) * xyzij_dd_outer, axis=-1) * CONV_FACT
    t3 = time.time()
    print(t3-t2, 'applying functions and summing over nn')
    if compute_forces:
        force_qq_kernel = qd_kernel.reshape((N,N,3))
        force_qd_kernel = dd_kernel 
        t4 = time.time()
        print(t4-t3, 'merely reshaping for forces')
        print(t4-t1, 'total')
        
        k1 = Bprime(rij_dd_diag, alpha) / rij_dd_diag
        k2 = -Cprime(rij_dd_full, alpha) * xyzij_dd_outer / rij_dd_full
        # k3 = -C(rij_dd_full, alpha) * 2 * xyzij_dd_outer_unitvec
        k3 = -C(rij_dd_full, alpha) * xyzij_dd_derivatives
        f = xyzij_dd * (k1+k2)[np.newaxis]  + k3
        force_dd_kernel = np.sum(f, axis=-1).T
        t5 = time.time()
        print(t5-t4, 'force dd kernel')
        
        
        return [qq_kernel, qd_kernel, dd_kernel], [force_qq_kernel, force_qd_kernel, force_dd_kernel]

    return qq_kernel, qd_kernel, dd_kernel


    

def compute_structure_props(lattice_vectors):
    '''Calculuate the volume, reciprocal lattice vectors
    Args:
        lattice_vectors: lattice vectors
   
    Returns:
        reciprocal lattice vectors,volume
    '''
    # convert to angstrom
    recip_lattice_vectors = 2 * jnp.pi * jnp.linalg.inv(lattice_vectors).T
    # volume = np.abs(np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
    volume = jnp.linalg.det(lattice_vectors)
    return [volume, recip_lattice_vectors]

def fsc(x,rs):
    return np.where(x<=rs, 0.5*(1.0-np.cos(np.pi * x/rs)), 1.0)
def dfsc(x,rs):
    return np.where(x<=rs, 0.5*np.sin(np.pi * x/rs) * np.pi/rs, 0.0)

def gsc(x,rs):
    if rs==0.:
        return 0.0
    sigma = rs/3.
    return 1.0 - np.exp(-x**2/(2*sigma**2))

def dgsc(x,rs):
    if rs==0.:
        return 0.0

    sigma = rs/3.
    return np.exp(-x**2/(2*sigma**2)) / sigma**2



# @partial(jax.jit, static_argnums=(0,1))
def realewald_qq_jax(alpha, Scells, lattice_vectors,coords,
                      Nmax_nn=100, Rc=100,
                      compute_forces=True):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in angstrom
        gaussian_width: gaussian width, array of atomic gussian width
        coords : atomic coordinates
    Return:
        real space contribution to energy
    '''
    print('Real space sum')
    N = len(coords)
    
    t1 = time.time()
    rij_qq, xyzij_qd, rij_qd = get_rij_qq(coords, Scells, Nmax_nn, lattice_vectors, Rc)
    t2 = time.time()
    print('\t',t2-t1, 'calculating rij jax')
    
    qq_kernel = jnp.sum(A(rij_qq, alpha), axis=-1) * CONV_FACT
    qd_kernel = jnp.sum(B(rij_qd, alpha)*xyzij_qd, axis=-1) * CONV_FACT
    qd_kernel = qd_kernel.reshape((N, -1))
    force_qq_kernel = qd_kernel.reshape((N,N,3))
    
    
    t5 = time.time()
    print('\t', t5-t1, ' sec total jax')
    
    
    return jnp.nan_to_num(qq_kernel), jnp.nan_to_num(force_qq_kernel)




def Electrostatics_brute_force(key,
             coords, 
             lattice_vectors, 
             acc_factor, 
             species_sequence,
             PBC=None,
             kmax=None, 
             rs=None,
             compute_forces=False):
    
    #structure properties
    lattice_length = np.linalg.norm(lattice_vectors,axis=1)

    struct_prop = compute_structure_props(lattice_vectors)
    volume, recip_lattice_vectors = struct_prop

    print('Direct suuum')

    output = {'key':key}
    
    # if not PBC: 

    #     LR_terms = real_space_term(gaussian_width, 
    #                               coords, 
    #                               compute_forces)
    #     E_sc = np.zeros(2)

    #     output['el_energy_kernel'] = LR_terms[0] + E_sc[0]
    #     if compute_forces:
    #         output['el_force_kernel'] = LR_terms[1] + E_sc[1]
    #     return output
    

    
    
    rij, xyzij = get_Rij(coords, lattice_vectors)
    rij_inv = np.nan_to_num(1 / rij, posinf=0, neginf=0) 
    output['el_energy_kernel'] = rij_inv * CONV_FACT
    

    if compute_forces:
        # xyzij_inv = np.nan_to_num(1 / xyzij, posinf=0, neginf=0)
        Vprime = (xyzij * rij_inv**3).T 
        output['el_force_kernel'] = Vprime * CONV_FACT
        return output
    return output


def real_space_term_vectorized(alpha,
                      lattice_vectors,
                      coords,
                      Rc,
                      compute_forces=False,
                      gabri=False):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in angstrom
        gaussian_width: gaussian width, array of atomic gussian width
        coords : atomic coordinates
    Return:
        real space contribution to energy
    '''
    print('Real space sum')
    N = len(coords)
    
    t1 = time.time()
    # rij_qq, xyzij_qd, rij_qd, xyzij_dd, rij_dd_diag, rij_dd = get_rij(coords, lattice_vectors, Rc)
    rij_qq, xyzij_qd, rij_qd, xyzij_dd_outer, rij_dd_diag, rij_dd_full, xyzij_dd, xyzij_dd_diag, xyzij_dd_outer_unitvec, xyzij_dd_derivatives = get_rij(coords, lattice_vectors, Rc)
    t2 = time.time()
    print(t2-t1, 'calculating rij')
    
    qq_kernel = np.sum(A(rij_qq, alpha), axis=-1) * CONV_FACT
    qd_kernel = np.sum(B(rij_qd, alpha)*xyzij_qd, axis=-1) * CONV_FACT
    qd_kernel = qd_kernel.reshape((N, -1))
    dd_kernel = np.sum(B(rij_dd_diag, alpha) - C(rij_dd_full, alpha) * xyzij_dd_outer, axis=-1) * CONV_FACT
    t3 = time.time()
    print(t3-t2, 'applying functions and summing over nn')
    if compute_forces:
        force_qq_kernel = qd_kernel.reshape((N,N,3))
        force_qd_kernel = dd_kernel 
        t4 = time.time()
        print(t4-t3, 'merely reshaping for forces')
        print(t4-t1, 'total')
        
        k1 = Bprime(rij_dd_diag, alpha) / rij_dd_diag
        k2 = -Cprime(rij_dd_full, alpha) * xyzij_dd_outer / rij_dd_full
        # k3 = -C(rij_dd_full, alpha) * 2 * xyzij_dd_outer_unitvec
        k3 = -C(rij_dd_full, alpha) * xyzij_dd_derivatives
        f = xyzij_dd * (k1+k2)[np.newaxis]  + k3
        force_dd_kernel = np.sum(f, axis=-1).T
        t5 = time.time()
        print(t5-t4, 'force dd kernel')
        
        
        return [qq_kernel, qd_kernel, dd_kernel], [force_qq_kernel, force_qd_kernel, force_dd_kernel]

    return qq_kernel, qd_kernel, dd_kernel



def recip_space_term_vectorized(Kc_sc, 
        alpha,
        volume, coords, lattice_vectors,
        recip_lattice_vectors,
        accuracy,
        compute_forces=False):
    print('Reciprocal space sum')
    N = len(coords)
    # # Start tracing memory allocations
    # tracemalloc.start()

    
    t1 = time.time()
    rij = coords[:,np.newaxis] - coords
    comb = kvectors_for_ewald(Kc_sc)
    k_vects = np.dot(comb , recip_lattice_vectors)
    k_norms = np.linalg.norm(k_vects, axis=1)
    t2 = time.time()
    print(t2-t1, 'kvect for Ewald', len(k_norms))
    
    # e^ikR
    cos = 2 * np.cos(rij @ k_vects.T)
    sin = 2 * np.sin(rij @ k_vects.T)
    # 4pi/V k^2 e^(-k^2/4alpha^2)
    ew_factor = 1/k_norms**2 * np.exp(-k_norms**2/(4*alpha**2)) * (4.* np.pi / volume)
    t3 = time.time()
    print(t3-t2, 'ew_factor and cos(kr) and sin(kr)')
    
    # qq
    qq_kernel = np.sum( cos * ew_factor, axis=2)
    qq_kernel *= CONV_FACT 
    t4 = time.time()
    print(t4-t3, 'sum over k')

    # qd
    qd_tmp = sin * ew_factor
    qd_kernel0 = np.matmul(qd_tmp, k_vects) * CONV_FACT 
    qd_kernel = qd_kernel0.reshape((N,-1))
    t5 = time.time()
    print(t5-t4, 'qd kernel')
    print(t5-t1, 'total time')

    
    
    # dd
    kakb = k_vects[:,:,np.newaxis] * k_vects[:, np.newaxis, :] 
    Kakb = np.tile(kakb, (N,N))
    cos2 = np.repeat(np.repeat(cos, 3, axis=1), 3, axis=0)
    Ak = cos2 * Kakb.T
    dd_kernel = Ak @ ew_factor
    dd_kernel *= CONV_FACT 
    
    # self terms
    self_qq = - 2* np.sqrt(alpha**2 / np.pi) * np.eye(N) * CONV_FACT
    self_dd = -2*2*np.pi/3*(alpha**2/np.pi)**(3/2) * np.eye(3*N) * CONV_FACT
    
    
    if compute_forces:
        force_qq_kernel = qd_kernel0 
        force_qd_kernel = dd_kernel 
        
        sin2 = np.repeat(np.repeat(sin, 3, axis=1), 3, axis=0)
        Ak = sin2 * Kakb.T
        force_dd_kernel = Ak * ew_factor @ k_vects
        force_dd_kernel *= CONV_FACT 
        
        
        return [qq_kernel+self_qq, qd_kernel, dd_kernel+self_dd], [force_qq_kernel, force_qd_kernel, force_dd_kernel]
    
    # # Current and peak memory usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 1024 / 1024 /1024:.1f} GB")
    # print(f"Peak usage: {peak / 1024 / 1024/ 1024:.1f} GB")

    # tracemalloc.stop()
    return qq_kernel, qd_kernel, dd_kernel


from scipy.optimize import brentq


def set_ewald_parameters_gaussians(N, Cell, sigmas, accuracy, Rc=0.0,
                                    gaussian_smearing=0.0, Kc_sc=0,
                                    verbose=False, q=None):
    """
    Set real-space cutoff (Rc), reciprocal-space cutoff (Kc), and the
    screening-Gaussian parameter alpha_screen for an Ewald sum with
    Gaussian charge distributions, given a target accuracy epsilon.

    Physics (worst-case bounds, alpha_min/alpha_max = smallest/largest
    of the per-site Gaussian widths):

      Real space:  error(Rc, alpha_screen)
                 = exp(-Rc^2 gij^2) / (Rc^2 gij),   1/gij^2 = 1/alpha_screen^2 + 2/alpha_min^2

      Rec space:   error(Kc, alpha_screen)
                 = 4*pi/(V*Kc^2) * exp(-Kc^2/(4*csi^2)),  1/csi^2 = 1/alpha_screen^2 + 2/alpha_max^2

    Exactly one of {Rc, gaussian_smearing, Kc_sc} may be supplied to fix
    the split of work between real and reciprocal space; the other two
    are then solved for from the accuracy target. If none are supplied,
    Rc defaults to the longest cell vector.

    Returns
    -------
    Kc, Rc, input_realspace, alpha_screen, input_recspace
    """
    vol = np.linalg.det(Cell)

    # Reciprocal lattice vectors: rows of Hk are b_i, with a_i . b_j = 2*pi*delta_ij.
    # NOTE: this requires the transpose of inv(Cell), not just inv(Cell).
    Hk = 2 * np.pi * np.linalg.inv(Cell).T
    b_norms = np.linalg.norm(Hk, axis=1)          # |b1|, |b2|, |b3|
    Lmax = np.max(np.linalg.norm(Cell, axis=1))

    sigmas = np.asarray(sigmas)
    sigmas = sigmas[np.isfinite(sigmas)]           # drop padding entries
    alphas = 1 / (np.sqrt(2) * sigmas)
    # exp(-α r^2) --> α_min = slowliest decay, α_max = faster decay
    alpha_min, alpha_max = alphas.min(), alphas.max()

    # ---- error models (docstring eqs. above) ----
    def real_error(Rc_, alpha_screen_):
        gij = 1 / np.sqrt(1 / alpha_screen_**2 + 2 / alpha_min**2)
        return np.exp(-Rc_**2 * gij**2) / (Rc_**2 * gij)

    def rec_error(Kc_, alpha_screen_):
        one_over_csi2 = 1 / alpha_screen_**2 + 2 / alpha_max**2
        return 4 * np.pi / (vol * Kc_**2) * np.exp(-Kc_**2 / 4 * one_over_csi2)

    # real_error decreases monotonically with alpha_screen and with Rc;
    # rec_error decreases monotonically with Kc but INCREASES with alpha_screen
    # (larger alpha_screen pushes more work into reciprocal space).

    def solve_alpha_from_real(Rc_):
        f = lambda a: real_error(Rc_, a) - accuracy
        lo, hi = 1e-6, 1e6
        if f(hi) > 0:          # even alpha_screen -> inf can't reach target accuracy at this Rc
            return None
        return brentq(f, lo, hi)

    def solve_Rc_from_real(alpha_screen_):
        f = lambda R: real_error(R, alpha_screen_) - accuracy
        lo, hi = 1e-3, 1e4
        if f(hi) > 0:
            return None
        return brentq(f, lo, hi)

    def solve_Kc_from_rec(alpha_screen_):
        f = lambda K: rec_error(K, alpha_screen_) - accuracy
        return brentq(f, 1e-3, 1e4)   # rec_error -> 0 as Kc -> inf, root always exists

    def solve_alpha_from_rec(Kc_):
        f = lambda a: rec_error(Kc_, a) - accuracy
        lo, hi = 1e-6, 1e6
        if f(hi) <= 0:          # rec accuracy satisfied for any alpha_screen: no upper bound needed
            return None
        return brentq(f, lo, hi)

    # ---- pick the (mutually exclusive) case ----
    Rc_is_set = Rc != 0.0
    Kc_is_set = Kc_sc != 0
    alpha_is_set = gaussian_smearing != 0.0
    real_space_sum = True

    if Kc_is_set:
        # user fixes the reciprocal-space workload directly
        Kc = Kc_sc * b_norms.min()   # sphere of this radius is covered by Kc_sc images along every axis
        alpha_screen = solve_alpha_from_rec(Kc)
        if alpha_screen is None:
            alpha_screen = 10 * alpha_max   # rec accuracy imposes no upper bound; pick a safe large value
        if not Rc_is_set:
            Rc = solve_Rc_from_real(alpha_screen)
            if Rc is None:
                Rc, real_space_sum = 0.0, False

    elif alpha_is_set:
        # user fixes the screening width directly
        alpha_screen = 1 / (np.sqrt(2) * gaussian_smearing)
        if not Rc_is_set:
            Rc = solve_Rc_from_real(alpha_screen)
            if Rc is None:
                Rc, real_space_sum = 0.0, False
        Kc = solve_Kc_from_rec(alpha_screen)

    else:
        # default: accuracy (and optionally a user-given Rc) drives everything
        if not Rc_is_set:
            Rc = Lmax
        alpha_screen = solve_alpha_from_real(Rc)
        if alpha_screen is None:
            alpha_screen, real_space_sum = np.inf, False
        Kc = solve_Kc_from_rec(alpha_screen)

    # ---- anisotropic k-point cutoffs: one integer per reciprocal lattice vector ----
    Kc_sc_axes = np.ceil(Kc / b_norms).astype(int)      # [n1, n2, n3]
    tot_kvecs = np.prod(2 * Kc_sc_axes + 1) // 2
    input_recspace = np.zeros(tuple(Kc_sc_axes))

    if real_space_sum:
        N_images_right = np.ceil(Rc / Lmax).astype(int)
        input_realspace = np.zeros(N_images_right)
    else:
        N_images_right = 0
        input_realspace = []

    if verbose:
        print('\nEwald sum for Gaussians')
        print('alpha_screen [1/A] =', np.round(alpha_screen, 3))
        if real_space_sum:
            print('Real space cutoff Rc [A] =', np.round(Rc, 3),
                  '-> max images:', (2 * N_images_right + 1)**3)
        else:
            print('Real space sum will NOT be performed')
        print('Reciprocal space cutoff Kc [1/A] =', np.round(Kc, 3),
              '-> k-grid', 2 * Kc_sc_axes + 1, '=', tot_kvecs, 'vectors\n')

    return (np.asarray(Kc), np.asarray(Rc), input_realspace,
            np.expand_dims(alpha_screen, 0), input_recspace)


def set_ewald_parameters_gaussians_old(coords, Cell, sigmas, accuracy, Rc=0.0, gaussian_smearing=0.0, Kc_sc=0, verbose=False, q=None):
    """
    Function to set the ewald parameters for Ewald sum with Gaussians.

    Parameters
    ----------
    coords : TYPE
        DESCRIPTION.
    Cell : TYPE
        DESCRIPTION.
    accuracy : TYPE
        DESCRIPTION.
    sigmas : TYPE
        DESCRIPTION.
    Rc : TYPE, optional
        DESCRIPTION. The default is 0.0.
    gaussian_smearing : TYPE, optional
        DESCRIPTION. The default is 0.0.
    Kc_sc : TYPE, optional
        DESCRIPTION. The default is 0.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    q : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    input_realspace : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    input_recspace : TYPE
        DESCRIPTION.
    
    There are three parameters to set: real space cutoff (Rc), 
    reciprocal space cutoff (Kc) and the std of the screening
    Gaussian (or it 1/sqrt(2) inverse, α_screen), based on accuracy=ε.
    Call T = √-ln(ε).
    
    Real space:
        The real space kernel is [ erf (αij Rij) - erf(gij Rij) ] / Rij,
        (where 1/gij^2 = 1/αi^2 + 1/αj^2 + 1/α_screen^2)
        hence tails, at large R, as e^{- gij_min^2 R^2} / R^2 \sim e^{- gij_min^2 R^2}.
        Now, the worst case scenario is that αi = αj = α_min.
        Imposing that the error at large R be ε means
        (1-old)    exp[- gij_min^2 Rc^2] = ε -> 1/α_screen^2 = Rc^2/T - 2/α_min^2 = rad
        (1)        exp[- gij_min^2 Rc^2] / (Rc^2 gij_min) = ε
        Now, if rad is positive then α_screen = 1/√rad, otherwise there are no
        α_screen that can provide that accuracy, hence I'll do everything in recspace.
        Both equations come from an asymptotic expansion of the erf function,
        but in Eq. (1-old) the 1/R^2 is neglected. However, eq. (1) is more correct,
        so the code uses that one, to be solved iteratively for α_screen.
        
    Rec space:
        The rec space kernel is 4\pi/G^2 e^{- G^2 / 4 gij^2}, where 1/gij^2 = 1/αi^2 + 1/αj^2 + 1/α_screen^2.
        So the worst case scenario is when gij is when αi = αj = α_min,
        because then gij is maximum, 1/gij_max^2 = 2/α_max^2 + 1/α_screen^2
        Imposing that the error at large G be ε means
        (2-old)    exp[- Gc^2 / 4 gij_max^2 ] = ε -> Gc = √4 gij_max^2 T
        (2)   4pi/Gc^2 exp[- Gc^2 / 4 gij_max^2 ] = ε -> Gc
            
    The default values for Rc, Gc and α_screen are 0,0,0.
    
    Case 1: Rc rules
    If Rc is not set, then the code sets it to Lmax, where Lmax is the length
    of the largest lattice vector. So Rc = Lmax.
    Then, the code estimates α_screen from (1), iff rad > 0, otherwise the real 
    space sum is not performed (α_screen -> inf).
    Finally, the code estimates Gc from (2) and from knowing α_screen.
    If Rc is set, the logic is the same, only Rc is set to the value provided.
    
    Case 2: α_screen rules
    If α_screen is set (through gaussian_smearing), then the user wants to 
    split the workload into real and rec space sum. This can be beneficial
    in those cases in which the rec space alone is expensive. 
    In this case, Rc and Gc are computed again from (1) and (2) again.
    
    Case 3: Gc rules
    If Gc is set, meaning that the user sets the price to pay for the rec sum,
    then α_screen is computed from (2). Rc is then computed from (1) and knowledge
    of α_screen.
    
        
    
    Starting 

    """
    
    # N = len(np.argwhere(coords != np.inf)) // 3 # only where pos is not infinite
    # to_evA =  14.39964547842567 # 1/(4pieps0) * e^2 / A^2 to ev/A
    vol = np.linalg.det(Cell)
    Hk = 2*np.pi * np.linalg.inv(Cell).T
    b1, b2, b3 = np.linalg.inv(Cell) * 2*np.pi
    Lmax = np.max(np.linalg.norm(Cell, axis=1)) # max lattice vector
    # L = np.linalg.det(Cell)**(1/3) # or V^(1/3)?
    T = np.sqrt(-np.log(accuracy)) # √-ln(ε)
    # remove unwanted infinites in case of padding
    args_finite_num = ~np.isinf(sigmas)
    sigmas = np.array(sigmas)[args_finite_num]
    alphas = 1/(np.sqrt(2)* np.array(sigmas))
    alpha_min = alphas.min()       # minimum α -> slower real space decay
    alpha_max = alphas.max()       # maximum α -> slower rec space decay
    real_space_sum = True
    
    acc_is_set = Rc == 0.0 and Kc_sc == 0.0 and gaussian_smearing == 0.0
    Rc_is_set  = Rc != 0.0
    Gc_is_set  = Kc_sc != 0.0
    alpha_is_set = gaussian_smearing != 0.0
    
    
    # options = [acc_is_set, Rc_is_set, Gc_is_set, alpha_is_set]
    
    if acc_is_set or Rc_is_set: # then find α_screen from (1), then Gc from (2)
        #---- Real space cutoff ----
        if not Rc_is_set: # ...
            Rc = Lmax # fix it to Lmax
        # ... otherwise use the one provided
        
        #---- solving iteratively eq. (1) to get α_screen ----
        alpha_screen = 0.001
        for i in range(100):
            alpha_screen += 0.1
            one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
            gij = 1/np.sqrt(one_over_gij2)
            error = np.exp(-Rc**2 * gij**2) / (Rc**2 * gij) 
            # print(i, error)
            if (error <= accuracy): # reached solution
                break
        # print(i, error, alpha_min)
        # print(alpha_screen, 'hhh')
        if i == 99: # convergence not reached
            alpha_screen = np.inf  # or treat as very large
            # meaning we won't compute anything in real space
            real_space_sum = False
            if not Rc_is_set:    
                Rc = 0.0
            
        #---- You then compute Kc from Recipr space accuracy ----
        # # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
        # csi_rec = 1.0 / np.sqrt( 1.0/alpha_screen**2 + 2.0/(alpha_max**2) )
        # # solving for e^{-Gmax^2 / 4 csi^2} = acc_factor
        # Kc = 2.0 * csi_rec * T   # |k| cutoff
        Kc = 0.001
        # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
        one_over_csiij2 = 1/alpha_screen**2 + 2/alpha_max**2
        for i in range(100):
            error = 4 * np.pi / (vol *Kc**2) * np.exp(-Kc**2 / 4 * one_over_csiij2)
            Kc += 0.1
            # print(i, Kc,  error, accuracy)
            if (error <= accuracy): # reached solution
                break

    if Gc_is_set: # then find α_screen from (2), and Rc from (1)
        if alpha_is_set:
            alpha_screen = 1/(np.sqrt(2)*gaussian_smearing)
        else: # compute it from (2)
            Kc_sc = np.array([Kc_sc, Kc_sc, Kc_sc])
            Kc = Hk @ Kc_sc
            Kc = np.max(Kc)
            
            alpha_screen = 0.001
            error = 99.
            for i in range(100):
                # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
                one_over_csiij2 = 1/alpha_screen**2 + 2/alpha_max**2
                error = 4 * np.pi / (vol *Kc**2) * np.exp(-Kc**2 / 4 * one_over_csiij2)
                alpha_screen += 0.1
                # print(i, error, accuracy)
                if (error <= accuracy): # reached solution
                    break
            if i == 0: # convergence reached immediately, you're overshooting
                alpha_screen = np.inf  # or treat as very large
                # meaning we won't compute anything in real space
                real_space_sum = False
                
            # # solving for e^{-Gmax^2 / 4 csi^2} = acc_factor
            # rad = 4*T**2/Gc**2 - 2/alpha_max**2
            # # if rad <= 0: # then there are no solutions: you can't get this accuracy
                          
            # alpha_screen = 1/np.sqrt(rad)
            
            # # Now determine Rc if not determined already
            # print('AAAAA', Rc, rad)
            # # -gij^2 Rc^2 = ln(ε) -> Rc
            # one_over_gij2 = 1/alpha_screen**2 + 2/alpha_max**2
            # Rc = T * np.sqrt(one_over_gij2)
            # print('AAAAA', Rc)

        if not Rc_is_set: # then find it from (1)
            # # -gij^2 Rc^2 = ln(ε) -> Rc
            # one_over_gij2 = 1/alpha_screen**2 + 2/alpha_max**2
            # Rc = T * np.sqrt(one_over_gij2)
            #---- solving iteratively eq. (1) to get Rc ----
            error = 99.
            Rc = 0.001
            for i in range(100):
                Rc += 0.1
                one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
                gij = 1/np.sqrt(one_over_gij2)
                error = np.exp(-Rc**2 * gij**2) / (Rc**2 * gij) 
                if (error <= accuracy): # reached solution
                    break
            if i == 99: # convergence not reached
                Rc = 0.01  # or treat as very large
                # meaning we won't compute anything in real space
                real_space_sum = False
                
    if alpha_is_set:
        alpha_screen = 1/(np.sqrt(2)*gaussian_smearing)
        # let's get Rc
        if not Rc_is_set:
            # # -gij^2 Rc^2 = ln(ε) -> Rc
            # one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
            # Rc = T * np.sqrt(one_over_gij2)
            #---- solving iteratively eq. (1) to get Rc ----
            error = 99.
            Rc = 0.001
            for i in range(100):
                Rc += 0.1
                one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
                gij = 1/np.sqrt(one_over_gij2)
                error = np.exp(-Rc**2 * gij**2) / (Rc**2 * gij) 
                if (error <= accuracy): # reached solution
                    break
            if i == 99: # convergence not reached
                Rc = 0.01  # or treat as very large
                # meaning we won't compute anything in real space
                real_space_sum = False
        if not Gc_is_set:
            #---- solving iteratively eq. (2) to get Gc ----
            Kc = 0.001
            error = 99.
            for i in range(100):
                # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
                one_over_csiij2 = 1/alpha_screen**2 + 2/alpha_max**2
                error = 4 * np.pi / (vol *Kc**2) * np.exp(-Kc**2 / 4 * one_over_csiij2)
                Kc += 0.1
                # print(i, error, accuracy)
                if (error <= accuracy): # reached solution
                    break
            
    # we want to use scaled Kc
    Kc_sc = np.linalg.inv(Hk) @ [Kc, Kc, Kc]
    Kc_sc = np.ceil(Kc_sc).astype(int).max()

    # Kcx_sc = (np.ceil(Kc / np.linalg.norm(b1))).astype(int)
    # Kcy_sc = (np.ceil(Kc / np.linalg.norm(b2))).astype(int)
    # Kcz_sc = (np.ceil(Kc / np.linalg.norm(b3))).astype(int)
    # # Kc_sc = int(Kc_sc)
    # # Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc, Kc_sc, Kc_sc
    # Kc_sc = np.array([Kcx_sc, Kcy_sc, Kcz_sc])
    # # # Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]       
        
    
            
            
        
    # if Rc == 0.0:
    #     #---- Real space cutoff is fixed to L ----
    #     Rc = Lmax # by default we want the real space cutoff to be just one cell
    
    
    # if gaussian_smearing == 0.0:
    #     #---- You first compute alpha_screen from Real space accuracy ----
        
    #     # # compute α_screen from Rc (require positive radicand)
    #     # rad = (T/Rc)**2 - 2.0/(alpha_min**2)
    #     # if rad <= 0: # then there are no physical values of α_screen
    #     #     alpha_screen = np.inf  # or treat as very large
    #     #     # meaning we won't compute anything in real space
    #     #     real_space_sum = False
    #     # else:
    #     #     alpha_screen = 1.0 / np.sqrt(rad)
        
    #     # solving iteratively eq. (1) to get α_screen
    #     error = 99.
    #     alpha_screen = 0.001
    #     for i in range(100):
    #     # while error >= accuracy:
    #         alpha_screen += 0.1
    #         one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
    #         gij = 1/np.sqrt(one_over_gij2)
    #         error = np.exp(-Rc**2 * gij**2) / (Rc**2 * gij) 
    #         if (error <= accuracy): # reached solution
    #             break
    #     if i == 99: # convergence not reached
    #         alpha_screen = np.inf  # or treat as very large
    #         # meaning we won't compute anything in real space
    #         real_space_sum = False
            
    # else:
    #     alpha_screen = 1/(np.sqrt(2)*gaussian_smearing)
    #     if Rc == Lmax:
    #         # -gij^2 Rc^2 = ln(ε) -> Rc
    #         one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
    #         Rc = T * np.sqrt(one_over_gij2)

    # if Kc_sc == 0.0:
    #     #---- You then compute Kc from Recipr space accuracy ----
    #     # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
    #     csi_rec = 1.0 / np.sqrt( 1.0/alpha_screen**2 + 2.0/(alpha_max**2) )
    #     # solving for e^{-Gmax^2 / 4 csi^2} = acc_factor
    #     Kc = 2.0 * csi_rec * T   # |k| cutoff
        
    #     # we want to use scaled Kc
    #     Kcx_sc = (np.ceil(Kc / np.linalg.norm(b1))).astype(int)
    #     Kcy_sc = (np.ceil(Kc / np.linalg.norm(b2))).astype(int)
    #     Kcz_sc = (np.ceil(Kc / np.linalg.norm(b3))).astype(int)
    
    # else: # the user provided a Gc -> let's compute α_screen from (2)
        
    #     if len([Kc_sc]) == 1: # then use the cutoff for all directions
    #         Kc_sc = int(Kc_sc)
    #         Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc, Kc_sc, Kc_sc
    #         Kc_sc = np.array([Kcx_sc, Kcy_sc, Kcz_sc])
    #         Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
        
    #     # else: # then the user provided 3 cutoffs
    #     #     Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc
    #     #     Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
        

    #     # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
    #     # solving for e^{-Gmax^2 / 4 csi^2} = acc_factor
    #     Gc = np.max(Kc)
    #     rad = 4*T**2/Gc**2 - 2/alpha_max**2
    #     # if rad <= 0: # then there are no solutions: you can't get this accuracy
                      
    #     alpha_screen = 1/np.sqrt(rad)
        
    #     # Now determine Rc if not determined already
    #     print('AAAAA', Rc, rad)
    #     # -gij^2 Rc^2 = ln(ε) -> Rc
    #     one_over_gij2 = 1/alpha_screen**2 + 2/alpha_min**2
    #     Rc = T * np.sqrt(one_over_gij2)
    #     print('AAAAA', Rc)
    #     exit()
    #     if Rc >= 0.9999:
    #         real_space_sum = True
    
    tot_kvecs = (2*Kc_sc+1)*(2*Kc_sc+1)*(2*Kc_sc+1) // 2

    
    # Reciprocal space input
    input_recspace = np.zeros((Kc_sc, Kc_sc, Kc_sc))
    
    # Real space input: in principle you don't HAVE to do it unless it's necessary
    if real_space_sum:
        N_images_right = np.ceil( (Rc) / Lmax ).astype(int) # maximum amount of cell replicas on the right
        input_realspace = np.zeros(N_images_right)
    else:
        input_realspace = []
    # # in practice I want to do it, because I will retain the info on Rc
    # N_images_right = np.ceil( (Rc) / Lmax ).astype(int) # maximum amount of cell replicas on the right
    # input_realspace = np.zeros(N_images_right)
    
    if verbose:
        print('\nEwald sum for Gaussians')
        print('Alpha was set to [1/A]', np.round(alpha_screen,3))
        print('\t=> gaussian screening std of [A]', np.round(1/np.sqrt(2) * 1/(alpha_screen),3))
        if real_space_sum:
            print('Real space cutoff Rc [A]', np.round(Rc, 3))
            print('\t => max amount of images real space', (2*N_images_right+1)**3)
        else:
            print('The real space sum will NOT be performed')

        print('Reciprocal space cutoffs Kc [1/A]', np.round(Kc,3))
        print('\t=> [{},{},{}] k-vectors = {} in total \n'.format(2*Kc_sc+1, 2*Kc_sc+1, 2*Kc_sc+1, tot_kvecs))
    
    return np.asarray(Kc), np.asarray(Rc), input_realspace, np.expand_dims(alpha_screen,0), input_recspace


def set_ewald_parameters_gaussians_for_batch(Ns, batch_cell, batch_sigmas, accuracy, Rc=0.0, gaussian_smearing=0.0, Kc_sc=0, verbose=False, q=None):
    Ns, batch_cell, batch_sigmas = np.array(Ns), np.array(batch_cell), np.array(batch_sigmas)

    bsize = len(Ns)
    Kcx_sc_max, Kcy_sc_max, Kcz_sc_max = 0,0,0
    Nmax_images_right = 0
    alphas_screen = []
    rcuts = []
    Gcuts = []

    

    for b in range(bsize): # iterating over configurations in this batch
        N, cell, sigmas = Ns[b], batch_cell[b], batch_sigmas[b]
        # running set_ewald_params for this configuration
        Gcut, rcut, input_realspace, alpha_screen, input_recspace = set_ewald_parameters_gaussians(N, cell, sigmas, accuracy, Rc=Rc, gaussian_smearing=gaussian_smearing, Kc_sc=Kc_sc)
        # getting Num of images in real space and Num of G-vecs in rec space for this configuration
        Nimages_right = len(input_realspace)
        Kcs = len(input_recspace)
        alphas_screen.append(alpha_screen[0]) # α_screen belongs to the conf and must be stored
        rcuts.append(rcut)
        Gcuts.append(Gcut)
        if Nimages_right > Nmax_images_right:
            Nmax_images_right = Nimages_right
        if Kcs > Kcx_sc_max:
            Kcx_sc_max = Kcs
    
    # Final rec space input is the max number of G-vecs required for the batch
    Kcy_sc_max = Kcz_sc_max = Kcx_sc_max # for now 
    input_recspace = np.zeros((Kcx_sc_max, Kcy_sc_max, Kcz_sc_max))
    # Bookkeeping Num of G-vecs
    tot_kvecs = (2*Kcx_sc_max+1)*(2*Kcy_sc_max+1)*(2*Kcz_sc_max+1) // 2
    
    # Final real space input is the max number of images required for the batch
    input_realspace = np.zeros(Nmax_images_right)
    real_space_sum = Nmax_images_right > 0
    
    if verbose:
        print('\nEwald sum for Gaussians (batch mode)')
        print('Alphas are set to [1/A]', np.round(alphas_screen,5))
        print('\t=> gaussian screening std of [A]', np.round(1/np.sqrt(2) * 1/(alphas_screen),3))
        if real_space_sum:
            print('Real space cutoff Rc [A]', np.round(Rc, 3))
            print('\t => max amount of images real space', (2*Nmax_images_right+1)**3)
        else:
            print('The real space sum is NOT performed')
        
        print('Amount of k-vectors = [{},{},{}], {} in total \n'.format(2*Kcx_sc_max+1, 2*Kcy_sc_max+1, 2*Kcz_sc_max+1, tot_kvecs))
        
    return Gcuts, rcuts, input_realspace, np.array(alphas_screen), input_recspace


def create_masks(Nats, M, n=1):
    bsize = len(Nats)
    masks = np.zeros((bsize, M*n, M*n))
    for b in range(bsize):
        this_nats = Nats[b]
        
        bigger_mask = np.eye(M*n)
        
        # these lines work, Lagrange multiplier is not last row/column
        # smaller_mask = np.ones((this_nats*n+1,this_nats*n+1))#masks[b]
        # smaller_mask[0:this_nats*n, 0:this_nats*n] = 0
        # smaller_mask[this_nats*n,this_nats*n] = 0
        # bigger_mask[0:this_nats*n+1,0:this_nats*n+1] = smaller_mask
        
        # these lines work, Lagrange multiplier is at the end
        # these two leaves ones in places of padded charges
        bigger_mask[0:M+this_nats, 0:M+this_nats] = 0
        bigger_mask[this_nats:M,this_nats:M] = np.eye(M-this_nats)
        
        bigger_mask[-1, 0:this_nats] = 1
        bigger_mask[0:this_nats, -1] = 1
        bigger_mask[-1, -1] = 0
        
        masks[b] = bigger_mask

    return masks

@jax.jit
def Electrostatics_finite_systems(coords):
    xyzij = coords[:, jnp.newaxis] - coords[jnp.newaxis] 
    rij = jnp.linalg.norm(xyzij, axis=-1)
    rij = jnp.where(rij == 0, jnp.inf, rij)
    # # rij = jjnp.where(rij > Rc, jjnp.inf, rij)
    qq_kernel = 1/rij * CONV_FACT
    force_qq_kernel = 1/rij[:,:,jnp.newaxis]**3 * xyzij * CONV_FACT
    out = {'el_energy_kernel': jnp.nan_to_num(qq_kernel), 'el_force_kernel': jnp.nan_to_num(force_qq_kernel)}
    return out

# @jax.jit
def Electrostatics_finite_systems_gaussians(coords, alphas):
    xyzij = coords[:, jnp.newaxis] - coords[jnp.newaxis] 
    rij = jnp.linalg.norm(xyzij, axis=-1)
    rij = jnp.where(rij == 0, jnp.inf, rij)
    alphas_mat = 1/jnp.sqrt( 1/alphas[:,jnp.newaxis]**2 + 1/alphas[jnp.newaxis]**2)

    # # rij = jjnp.where(rij > Rc, jjnp.inf, rij)
    qq_kernel = erf(alphas_mat*rij)/rij * CONV_FACT
    force_qq_kernel = xyzij * D(rij[:,:,jnp.newaxis], alphas_mat[:,:,jnp.newaxis]) * CONV_FACT
    out = {'el_energy_kernel': jnp.nan_to_num(qq_kernel), 'el_force_kernel': jnp.nan_to_num(force_qq_kernel)}
    return out
