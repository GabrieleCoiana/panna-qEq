# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:08:32 2024

@author: Gabriele Coiana
"""
import numpy as np
import itertools
import time
from .pbc_jax import get_rij, get_rij_qq, get_Scells_images
import tracemalloc
# from scipy.special import erf, erfc
from jax.scipy.special import erf, erfc
# import scipy.constants as constants
import jax
# enable_64 = False
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
             alpha,
             realspace_cutoff,
             recspace_cutoff, # unused
             input_realspace=jnp.zeros(1),
             input_recspace=jnp.zeros((1,1,1)),
             range_separated_cutoff = 0.0,
             verbose = False,
             mode = 'kernel', 
             q=0
             ):
    '''
    '''    
    if verbose:
        print('Ewald sum')
        print('\tN of atoms', len(coords))
        print('\tN of images in real space', (input_realspace.size * 2 + 1)**3)
        print('\tN/2 -1 of g-vecs in rec space', input_recspace.shape)
    
    output = {}
    
    if mode == 'kernel': # compute only energy kernel
        # rec space Ewald summation
        rec_en_kernel = recewald_qq_jax_vect(input_recspace,
                                                alpha, 
                                                coords, 
                                                lattice_vectors, 
                                                compute_forces=True)        
    
        # real space Ewald summation
        real_en_kernel = realewald_qq_jax_vect(alpha, 
                                                realspace_cutoff, 
                                                input_realspace, 
                                                lattice_vectors, 
                                                coords, 
                                                compute_forces=True)

        output['el_energy_kernel'] = real_en_kernel + rec_en_kernel  
        # output['el_force_kernel'] =  real_force_kernel + rec_force_kernel
        
        
        # output['el_energy_kernel_real'] = real_en_kernel
        # output['el_energy_kernel_rec'] = rec_en_kernel
        # output['el_force_kernel_real'] = real_force_kernel
        # output['el_force_kernel_rec'] = rec_force_kernel
        
    
    else: # charges have been computed and we want the forces and stress
        real_forces, real_virial = real_forces_and_stress(alpha, 
                                                          realspace_cutoff, 
                                                          input_realspace, 
                                                          lattice_vectors, 
                                                          coords,
                                                          q)
        
        rec_forces, rec_virial = rec_forces_and_stress(input_recspace, 
                                                       alpha, 
                                                       coords, 
                                                       lattice_vectors, 
                                                       q)
        output['forces'] = real_forces + rec_forces 
        output['virial'] = real_virial + rec_virial  
    
    
    if range_separated_cutoff > 0.0: # never really tested
        input_realspace = jnp.zeros( 3)
        esr = compute_rs_1_over_rij(input_realspace, lattice_vectors, coords, range_separated_cutoff)
        output['el_energy_sr'] = esr
    
    # N = coords.shape[0]
    # output['linsys_bvector'] = jnp.zeros(N)
    # output['linsys_masks'] = mask
    
    
    
    return output


@jax.jit
def realewald_qq_jax_vect(alpha,
                          rcut,
                      input_realspace,  
                      lattice_vectors,
                      coords,
                      # Rsc=0.0,
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
    print('EWALDqq - Kernels: Real space sum, vectorized')
    
    Scells = get_Scells_images(input_realspace, lattice_vectors, coords)
    coords = jnp.where(jnp.isinf(coords), 1e19, coords)
    
    # Rc = 100
    # t1 = time.time()
    xyzij = coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:] 
    rij = jnp.linalg.norm(xyzij, axis=-1)
    
    # exclude self Rii = 0 and Rij > Rc
    rij = jnp.where(rij == 0, 1e19, rij)
    rij = jnp.where(rij > rcut, 1e19, rij)

    qq_kernel = jnp.sum(A(rij, alpha), axis=0) * CONV_FACT
    
    # if compute_forces:
    # force_qq_kernel = jnp.sum(B(rij[:,:,:,jnp.newaxis], alpha)*xyzij, axis=0) * CONV_FACT
    return qq_kernel#, jnp.nan_to_num(force_qq_kernel) # qq_kernel, force_qq_kernel #
    
    # WARNING: returning jnp.nan_to_num allows to use this function as is even in
    #       the case of atoms padding. However, it is unsafe in the case where, e.g.,
    #       you want to debug proper NaNs which might be happening if parameters
    #       like alpha get too large and give inf / inf forms.
    return jnp.nan_to_num(qq_kernel)



  
@jax.jit
def recewald_qq_jax_vect(input_recspace, 
                alpha,
                 coords,
                lattice_vectors,
                compute_forces=False):
    print('EWALDqq - Kernels: Reciprocal space sum, vectorized Stefanos version')
    comb = kvectors_for_ewald(input_recspace)
    # structure properties
    volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)
    # coords = jnp.where(jnp.isinf(coords), 1e19, coords)
    
    N = len(coords)
    C = (4.* jnp.pi / volume)

    k_vects = jnp.dot(comb , recip_lattice_vectors)
    k_norms = jnp.linalg.norm(k_vects, axis=1)
    

    # e^ikR
    # eikr = jnp.exp(1j * coords @ k_vects.T) 
    # sin = 2 * jnp.sin(coords @ k_vects.T)
    eikr = jnp.where(~jnp.isnan(jnp.exp(-1j * coords @ k_vects.T)), 
                      jnp.exp(-1j * coords @ k_vects.T), 
                      0)# (N, NG)
    
    # 4pi/V k^2 e^(-k^2/4alpha^2)
    ew_factor = jnp.sqrt(1/k_norms**2 * jnp.exp(-k_norms**2/(4*alpha**2)))
    Aik = eikr * ew_factor
    
    # qq
    qq_kernel = 2 *(Aik @ Aik.conj().T).real
    qq_kernel *= C * CONV_FACT 
    

    # self terms
    valid = jnp.all(jnp.isfinite(coords), axis=-1)        # (N,)
    self_qq = - 2* jnp.sqrt(alpha**2 / jnp.pi) * jnp.diag(valid) * CONV_FACT
    
    # if compute_forces:

    # # force
    # Bxik = Aik[jnp.newaxis] * 1j*k_vects.T[:,jnp.newaxis,:]
    # op = Bxik.conj() @ Aik.T
    # force_qq_kernel = ( op + op.conj() ).T.real
    # force_qq_kernel *= C * CONV_FACT
    
    return qq_kernel+self_qq#, jnp.nan_to_num(force_qq_kernel)
    
    # WARNING: see warning above
    return jnp.nan_to_num(qq_kernel+self_qq)
    
  
@jax.jit
def recewald_qq_jax_vect_gaussians(
                alphas,
                coords,
                lattice_vectors,
                input_recspace):
    print('Reciprocal space sum for Gaussians, Stefanos version')
    N = len(coords)
    # structure properties
    volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)
    C = (4.* jnp.pi / volume)
    alphas = alphas[:,jnp.newaxis]
    
    comb = kvectors_for_ewald(input_recspace)

    k_vects = jnp.dot(comb , recip_lattice_vectors)
    k_norms = jnp.linalg.norm(k_vects, axis=1)

    # e^ikR
    eikr = jnp.exp(1j * coords @ k_vects.T) 
    # sin = 2 * jnp.sin(coords @ k_vects.T)

    # 4pi/V k^2 e^(-k^2/4alpha^2)
    ew_factor = 1/k_norms * jnp.exp(-k_norms**2/(4*alphas**2))
    Aik = eikr * ew_factor
    
    # qq
    qq_kernel = 2 *(Aik @ Aik.conj().T).real
    qq_kernel *= C * CONV_FACT 
    
    # # force
    # ew_factor_force = ew_factor * 1j*k_vects.T
    # Fik = ew_factor_force[:,jnp.newaxis,:] * eikr[jnp.newaxis,:,:] 
    # force_qq_kernel = 2 * (Fik @ Aik.T.conj()).T.real
    

    # force
    Bxik = Aik[jnp.newaxis] * 1j*k_vects.T[:,jnp.newaxis,:]
    op = Bxik.conj() @ Aik.T
    force_qq_kernel = ( op + op.conj() ).T.real
    force_qq_kernel *= C * CONV_FACT
    
    # # self terms
    # self_qq = - 2* jnp.sqrt(alphas**2 / jnp.pi) * jnp.eye(N) * CONV_FACT

    return jnp.nan_to_num(qq_kernel), jnp.nan_to_num(force_qq_kernel)
    

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

@jax.jit
def real_forces_and_stress(alpha, rcut, input_realspace, lattice_vectors, coords, q):
    print('EWALDqq - Real forces and stress')
    coords = jnp.where(jnp.isinf(coords), 1e9, coords)
    Scells = get_Scells_images(input_realspace, lattice_vectors, coords)

    # Rc = 100
    # t1 = time.time()
    xyzij = coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:] # (Ncells, N, N, 3)
    rij = jnp.linalg.norm(xyzij, axis=-1)     # (Ncells, N, N)

    # exclude self Rii = 0 and Rij > Rc
    rij = jnp.where(rij < 1e-6, 1e18, rij)   # (Ncells, N, N)
    rij = jnp.where(rij > rcut, 1e18, rij) # (Ncells, N, N)
    
    # qiqj
    qiqj = q[None, :,None,None] * q[None, None,:,None] # (1,N,N,1,1)

    # F^α_{ij}
    force_qq_pairs = qiqj*B(rij[:,:,:,None], alpha)*xyzij # (Ncells, N, N, 3)
    # F^α_i = \sum_j F^α_{ij}
    forces_qq = jnp.sum(force_qq_pairs, axis=(0,2)) * CONV_FACT # (N, 3)
    # W_{α β} = \sum_{ij} F^α_{ij} r^β_{ij}
    virial_qq = .5 * jnp.einsum('Cijx, Cijy -> xy', force_qq_pairs, xyzij) * CONV_FACT # (3,3)

    return forces_qq, virial_qq

@jax.jit
def rec_forces_and_stress(input_recspace, 
                alpha,
                 coords,
                lattice_vectors, q):
    print('EWALDqq - Rec forces and stress')
    comb = kvectors_for_ewald(input_recspace)
    # structure properties
    volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)
    coords = jnp.where(jnp.isinf(coords), 1e9, coords)
    
    # G
    k_vects = jnp.dot(comb , recip_lattice_vectors)
    k_norms = jnp.linalg.norm(k_vects, axis=1)
    
    # e^iGR
    # eikr = jnp.exp(-1j * coords @ k_vects.T)  # (N, NG)
    eikr = jnp.where(~jnp.isnan(jnp.exp(-1j * coords @ k_vects.T)), 
                      jnp.exp(-1j * coords @ k_vects.T), 
                      0)# (N, NG)
    
    # 4pi/(V k^2) e^(-k^2/4η^2)
    ew_factor = 4*jnp.pi / volume \
        * 1/k_norms**2 * jnp.exp(-k_norms**2/(4*alpha**2)) # (NG)
    # S(G) structure factor
    S = q @ eikr # (NG) 
    S2 = S * S.conj() # (NG)
    
    
    
    # force
    # dS*(G) / d r_i^α = q_i (i Gα) e^{i G r_i}
    Bxik = q[:,None] * eikr * 1j*k_vects.T[:,None,:] # (3, N, NG)
    # dS*/dr S + c.c.
    Bxik = Bxik * S.conj() * ew_factor # (3, N, NG)
    Bxik = Bxik + Bxik.conj()   # (3, N, NG)
    forces_qq = jnp.sum(Bxik, axis=-1).real.T * CONV_FACT # (N,3)
    
    
    
    S2 = S2[:,None,None] # (NG, 1, 1)
    ew_factor = ew_factor[:,None,None] # (NG, 1, 1)
    
    # stress: δ_{αβ} - 2 Gα Gβ (1/4η^2 + 1/G^2)
    # stress factor: (1/4η^2 + 1/G^2)
    stress_factor = 1/k_norms**2 + 1/(4*alpha**2) # (NG)
    stress_factor = stress_factor[:,None,None]    # (NG, 1, 1)
    # outer product: δ_{αβ} - 2 Gα Gβ 
    GaGb = k_vects[:,:,None] * k_vects[:,None,:] # (NG, 3, 3)
    delta_ab = jnp.eye(3)[None] # (1, 3, 3)
    virial_qq = S2 * ew_factor * (delta_ab - 2 * GaGb * stress_factor)   # (NG, 3, 3)
    # \sum_G
    virial_qq = jnp.sum(virial_qq, axis=0).real * CONV_FACT # (3,3)
    
    return forces_qq, virial_qq




@jax.jit
def compute_rs_1_over_rij(input_realspace,
                          lattice_vectors,
                          coords,
                          Rsc):
    print('Range separated contribution')
    Scells = get_Scells_images(input_realspace, lattice_vectors, coords)


    xyzij = coords[jnp.newaxis,  :, jnp.newaxis] - Scells[:,jnp.newaxis,:]  # (Nimages, N, N, 3)
    rij = jnp.linalg.norm(xyzij, axis=-1) # (Nimages, N, N)
    
    # exclude all atoms Rij > Rsc
    rij = jnp.where(rij > Rsc, 0.0, rij) # (Nimages, N, N)
    
    # exclude self Rii = 0 and avoid 1/0
    rij = jnp.where(rij == 0, jnp.inf, rij) # (Nimages, N, N)
    
    return jnp.sum(1/rij, axis=0) # (N, N)

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
    
    displ = comb @ latvec # rows of latvec are lattice vectors # jnp.dot(latvec, comb.T).T
    displ = displ[:,jnp.newaxis,:]
    Scells = displ + R0
    return Scells


def erf_over_r(r, alpha):
    erf_over_r = erf(alpha*r) / r
    return erf_over_r

def erfc_over_r(r, alpha):
    erfc_over_r  = erfc(alpha*r)/r
    return erfc_over_r

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
    #convert to angstrom
    recip_lattice_vectors = 2 * np.pi * jnp.linalg.inv(lattice_vectors).T
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
@jax.jit
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






def set_ewald_parameters(N, Cell, accuracy, Rc=0.0, gaussian_smearing=0.0, Kc_sc=0, verbose=False, q=None):
    # N = len(np.argwhere(coords < 1e19)) // 3 # only where pos is not infinite
    to_evA =  14.39964547842567 # 1/(4pieps0) * e^2 / A^2 to ev/A
    b1, b2, b3 = jnp.linalg.inv(Cell) * 2*np.pi
    L = np.max(jnp.linalg.norm(Cell, axis=1)) # max lattice vector
    # L = np.linalg.det(Cell)**(1/3) # or V^(1/3)?

    if Rc == 0.0:
        #---- Real space cutoff is fixed to L ----
        Rc = L # by default we want the real space cutoff to be just one cell

    # Under Eq. 4 of J. Chem. Phys. 103, 3668–3679 (1995)
    if q is None or np.isnan(q):
        q = jnp.repeat(np.array([1,-1]), N//2) # fake +1e -1e molecule
    Q = jnp.sqrt(np.sum(q**2))

    if gaussian_smearing == 0.0:
        #---- You first compute alpha from Real space accuracy ----
        # Eq. 9 of J. Chem. Phys. 103, 3668–3679 (1995)
        exp = accuracy * np.sqrt(N * Rc * L**3) / (2 * Q**2) / to_evA
        alpha_scaled = np.sqrt(-np.log(exp) * L**2 / Rc**2)
        alpha = alpha_scaled/L
    else:
        alpha = 1/(np.sqrt(2)*gaussian_smearing)

    if Kc_sc == 0.0:
        #---- You then compute Kc from Recipr space accuracy ----
        # Eq. 11 of J. Chem. Phys. 103, 3668–3679 (1995)
        Kc = 1.
        # error = 2*Q**2*alpha_scaled/L**2 * 1/np.sqrt(np.pi * N) * 1/np.sqrt(Kc) * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
        prefactor = 1
        error_k = prefactor * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
        # error_check = lambda Kc: prefactor * np.exp(-Kc**2 / (4*alpha**2)) * to_evA > accuracy
        # jax.lax.while_loop(error_check, lambda x: x+0.1, Kc)
        while error_k > accuracy:
            Kc += 0.1
            error_k = prefactor * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
        # we want to use scaled Kc
        Kcx_sc = (np.ceil(Kc / np.linalg.norm(b1))).astype(int)
        Kcy_sc = (np.ceil(Kc / np.linalg.norm(b2))).astype(int)
        Kcz_sc = (np.ceil(Kc / np.linalg.norm(b3))).astype(int)
        Kc_sc = [Kcx_sc, Kcy_sc, Kcz_sc]
        
    elif len([Kc_sc]) == 1: # then use the cutoff for all directions
        Kc_sc = int(Kc_sc)
        Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc, Kc_sc, Kc_sc
        Kc_sc = np.array([Kcx_sc, Kcy_sc, Kcz_sc])
        Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
    
    else: # then the user provided 3 cutoffs
        Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc
        Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
    
    tot_kvecs = (2*Kcx_sc+1)*(2*Kcy_sc+1)*(2*Kcz_sc+1) // 2

    
    # Reciprocal space input
    input_recspace = np.zeros((Kcx_sc, Kcy_sc, Kcz_sc))
    
    # Real space input
    Lmax = np.max(np.linalg.norm(Cell, axis=1)) #lattice vectors must be read horizontally
    N_images_right = np.ceil( (Rc) / Lmax ).astype(int) # maximum amount of cell replicas on the right
    input_realspace = np.zeros(N_images_right)
    if verbose:
        print('\nEwald sum for point charges')
        print('Real space cutoff Rc [A]', np.round(Rc, 3))
        print('\t => max amount of images real space', (2*N_images_right+1)**3)
        print('Alpha was set to [1/A]', np.round(alpha,3))
        print('\t=> gaussian screening std of [A]', np.round(1/np.sqrt(2) * 1/(alpha),3))
        print('Reciprocal space cutoffs Kc [1/A]', np.round(Kc,3))
        print('\t=> [{},{},{}] k-vectors = {} in total \n'.format(2*Kcx_sc+1, 2*Kcy_sc+1, 2*Kcz_sc+1, tot_kvecs))
    
    
    return np.asarray(Kc), np.asarray(Rc), input_realspace, np.expand_dims(alpha,0), input_recspace



def set_ewald_parameters_for_batch(Ns, batch_Cell, accuracy, Rc=0.0, gaussian_smearing=0.0, Kc_sc=0.0, verbose=False, q=None):
    bsize = len(Ns)
    Kcx_sc_max, Kcy_sc_max, Kcz_sc_max = 0,0,0
    Nmax_images_right = 0
    alphas = []
    rcuts = []
    Gcuts = []
    
    for b in range(bsize):
        # coords = batch_coords[b]
        # remove padded elements (whose position is np.inf)
        # args_finite_num = ~np.isinf(coords)
        # coords = coords[args_finite_num].reshape((-1,3))
        Cell = batch_Cell[b]
        N = Ns[b] #len(np.argwhere(coords != np.inf)) // 3 # only where pos is not infinite
        to_evA = 14.39964547842567 # 1/(4pieps0) * e^2 / A^2 to ev/A  
        b1, b2, b3 = np.linalg.inv(Cell) * 2*np.pi
        L = np.max(np.linalg.norm(Cell, axis=1)) # max lattice vector
        # L = np.linalg.det(Cell)**(1/3) # or V^(1/3)?
        
        if Rc == 0.0:
            #---- Real space cutoff is fixed to L ----
            Rc_used = L # by default we want the real space cutoff to be just one cell
        else:
            Rc_used = float(Rc)
            
        # Under Eq. 4 of J. Chem. Phys. 103, 3668–3679 (1995)
        if q is None:
            q = np.repeat(np.array([1,-1]), N//2) # fake +1e -1e molecule
        Q = np.sqrt(np.sum(q**2))
    
        if gaussian_smearing == 0.0:
            #---- You first compute alpha from Real space accuracy ----
            # Eq. 9 of J. Chem. Phys. 103, 3668–3679 (1995)
            exp = accuracy * np.sqrt(N * Rc_used * L**3) / (2 * Q**2) / to_evA
            alpha_scaled = np.sqrt(-np.log(exp) * L**2 / Rc_used**2)
            alpha_used = alpha_scaled/L
            alphas.append(alpha_used)
            # print(L, alpha_used, accuracy, N, Rc_used, L, Q, 'mmm')
        # alpha_scaled = alpha/L
        else:
            alpha_used = 1/(np.sqrt(2)*gaussian_smearing)
            alphas.append(alpha_used)
        
        if Kc_sc == 0.0:
            #---- You then compute Kc from Recipr space accuracy ----
            # Eq. 11 of J. Chem. Phys. 103, 3668–3679 (1995)
            Kc = 1.
            # error = 2*Q**2*alpha_scaled/L**2 * 1/np.sqrt(np.pi * N) * 1/np.sqrt(Kc) * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
            prefactor = 1
            error_k = prefactor * np.exp(-Kc**2 / (4*alpha_used**2)) * to_evA
            
            # error_check = lambda Kc: prefactor * jnp.exp(-Kc**2 / (4*alpha**2)) * to_evA > accuracy
            # jax.lax.while_loop(error_check, lambda x: x+0.1, Kc)

            while error_k > accuracy:
                Kc += 0.1
                error_k = prefactor * np.exp(-Kc**2 / (4*alpha_used**2)) * to_evA
            # we want to use scaled Kc
            Kcx_sc = (np.ceil(Kc / np.linalg.norm(b1))).astype(int)
            Kcy_sc = (np.ceil(Kc / np.linalg.norm(b2))).astype(int)
            Kcz_sc = (np.ceil(Kc / np.linalg.norm(b3))).astype(int)
            # Kc_sc = np.array([Kcx_sc, Kcy_sc, Kcz_sc])

        elif len([Kc_sc]) == 1: # then use the scaled cutoff for all directions
            Kc_sc = int(Kc_sc) # force to int
            Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc, Kc_sc, Kc_sc
            Kc_sc = [Kcx_sc, Kcy_sc, Kcz_sc]
            Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
        
        else: # then the user provided 3 scaled cutoffs
            Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc
            Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
        
        if (Kcx_sc > Kcx_sc_max):
            Kcx_sc_max = Kcx_sc
        if (Kcy_sc > Kcy_sc_max):
            Kcy_sc_max = Kcy_sc
        if (Kcz_sc > Kcz_sc_max):
            Kcz_sc_max = Kcz_sc
        
        
        
        Lmax = np.max(np.linalg.norm(Cell, axis=1)) #lattice vectors must be read horizontally
        N_images_right = np.ceil( ( Rc_used) / Lmax ).astype(int) # maximum amount of cell replicas on the right
        # Nmax_images = 2*Nmax_images_right+1 # maximum amount of images
        if (N_images_right > Nmax_images_right):
            Nmax_images_right = N_images_right
        
            
        tot_kvecs = (2*Kcx_sc_max+1)*(2*Kcy_sc_max+1)*(2*Kcz_sc_max+1) // 2
        rcuts.append(Rc_used)
        Gcuts.append(Kc)
        
    # Reciprocal space input
    input_recspace = np.zeros((Kcx_sc_max, Kcy_sc_max, Kcz_sc_max))
    
    # Real space input
    input_realspace = np.zeros(Nmax_images_right)
    
    if verbose:
        print('\nEwald sum for point charges (batch mode)')
        print('Real space cutoff Rc [A]', np.round(Rc_used, 3))
        print('\t => max amount of images real space', (2*N_images_right+1)**3)
        print('Alphas are set to [1/A]', np.round(alphas,5))
        print('\t=> gaussian screening std of [A]', np.round(1/np.sqrt(2) * 1/(alphas),3))
        print('Amount of k-vectors = [{},{},{}], {} in total \n'.format(2*Kcx_sc_max+1, 2*Kcy_sc_max+1, 2*Kcz_sc_max+1, tot_kvecs))
    
    # print(rcuts, alphas, 'aaaaaaaaaaaaaaa') 
    # exit()
    # 
    return np.array(Gcuts), np.array(rcuts), input_realspace, np.array(alphas), input_recspace

# def set_ewald_parameters_gaussians(coords, Cell, accuracy, alphas, Rc=0.0, gaussian_smearing=0.0, Kc_sc=0, verbose=False, q=None):
#     # N = len(np.argwhere(coords != np.inf)) // 3 # only where pos is not infinite
#     # to_evA =  14.39964547842567 # 1/(4pieps0) * e^2 / A^2 to ev/A
#     b1, b2, b3 = np.linalg.inv(Cell) * 2*np.pi
#     Lmax = np.max(np.linalg.norm(Cell, axis=1)) # max lattice vector
#     # L = np.linalg.det(Cell)**(1/3) # or V^(1/3)?
#     T = np.sqrt(-np.log(accuracy)) # √-ln(ε)
#     alpha_min = alphas.min()       # minimum α -> slower real space decay
#     alpha_max = alphas.max()       # maximum α -> slower rec space decay

#     if Rc == 0.0:
#         #---- Real space cutoff is fixed to L ----
#         Rc = Lmax # by default we want the real space cutoff to be just one cell
    
    
#     if gaussian_smearing == 0.0:
#         #---- You first compute alpha_screen from Real space accuracy ----
#         # compute α_screen from Rc (require positive radicand)
#         rad = (T/Rc)**2 - 2.0/(alpha_min**2)

#         if rad <= 0: # then there are no physical values of α_screen
#             alpha_screen = np.inf  # or treat as very large
#             # meaning we won't compute anything in real space
#             real_space_sum = False
#         else:
#             alpha_screen = 1.0 / np.sqrt(rad)
#     else:
#         alpha_screen = 1/(np.sqrt(2)*gaussian_smearing)

#     if Kc_sc == 0.0:
#         #---- You then compute Kc from Recipr space accuracy ----
#         # 1/csi^2 = 2 / α_max^2 + 1/α_screen^2
#         csi_rec = 1.0 / np.sqrt( 1.0/alpha_screen**2 + 2.0/(alpha_max**2) )
#         # solving for e^{-Gmax^2 / 4 csi^2} = acc_factor
#         Kc = 2.0 * csi_rec * T   # |k| cutoff
        
#         # we want to use scaled Kc
#         Kcx_sc = (np.ceil(Kc / np.linalg.norm(b1))).astype(int)
#         Kcy_sc = (np.ceil(Kc / np.linalg.norm(b2))).astype(int)
#         Kcz_sc = (np.ceil(Kc / np.linalg.norm(b3))).astype(int)
        
#     elif len([Kc_sc]) == 1: # then use the cutoff for all directions
#         Kc_sc = int(Kc_sc)
#         Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc, Kc_sc, Kc_sc
#         Kc_sc = np.array([Kcx_sc, Kcy_sc, Kcz_sc])
#         Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
    
#     else: # then the user provided 3 cutoffs
#         Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc
#         Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
    
#     tot_kvecs = (2*Kcx_sc+1)*(2*Kcy_sc+1)*(2*Kcz_sc+1) // 2

    
#     # Reciprocal space input
#     input_recspace = np.zeros((Kcx_sc, Kcy_sc, Kcz_sc))
    
#     # Real space input
#     if real_space_sum:
#         N_images_right = np.ceil( (Rc) / Lmax ).astype(int) # maximum amount of cell replicas on the right
#         input_realspace = np.zeros(N_images_right)
#     else:
#         input_realspace = 0
#     if verbose:
#         print('Alpha was set to [1/A]', np.round(alpha_screen,3))
#         print('\t=> gaussian screening std of [A]', np.round(1/np.sqrt(2) * 1/(alpha_screen),3))
#         if real_space_sum:
#             print('\nReal space cutoff Rc [A]', np.round(Rc, 3))
#             print('\t => max amount of images real space', (2*N_images_right+1)**3)
#         else:
#             print('Therefore the real space sum is NOT computed')

#         print('Reciprocal space cutoffs Kc [2pi/L]', np.round(Kc,3))
#         print('\t=> [{},{},{}] k-vectors = {} in total \n'.format(2*Kcx_sc+1, 2*Kcy_sc+1, 2*Kcz_sc+1, tot_kvecs))
    
    
#     return input_realspace, np.expand_dims(alpha_screen,0), input_recspace


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
