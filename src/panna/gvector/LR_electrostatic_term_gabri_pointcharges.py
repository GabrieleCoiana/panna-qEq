# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:08:32 2024

@author: Gabriele Coiana
"""
import numpy as np
import itertools
from mendeleev import element
import time
from .pbc import replicas_max_idx, make_replicas, get_el_kernels, get_Scells
import tracemalloc
from scipy.special import erf, erfc
import scipy.constants as constants

e_charge = 1.60217662e-19 
epsilon_0 = 8.85418782e-12 #F/m unit
CONV_FACT = 1e10 * constants.e / (4 * np.pi * constants.epsilon_0)


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




def set_ewald_parameters(N, Cell, accuracy, Rc=None, alpha=None, Kc_sc=None, verbose=False, q=None):
    to_evA = 1 # 14.35 # 1/(4pieps0) * e^2 / A^2 to ev/A
    b1, b2, b3 = np.linalg.inv(Cell) * 2*np.pi
    L = np.max(np.linalg.norm(Cell, axis=1)) # max lattice vector
    # L = np.linalg.det(Cell)**(1/3) # or V^(1/3)?
    
    
    if Rc is None:
        #---- Real space cutoff is fixed to L ----
        Rc = L # by default we want the real space cutoff to be just one cell

    # Under Eq. 4 of J. Chem. Phys. 103, 3668–3679 (1995)
    if q is None:
        q = np.repeat([1,-1], N//2) # fake +1e -1e molecule
    Q = np.sqrt(np.sum(q**2))

    if alpha is None:
        #---- You first compute alpha from Real space accuracy ----
    
        # Eq. 9 of J. Chem. Phys. 103, 3668–3679 (1995)
        exp = accuracy * np.sqrt(N * Rc * L**3) / (2 * Q**2) / to_evA
        alpha_scaled = np.sqrt(-np.log(exp) * L**2 / Rc**2)
        alpha = alpha_scaled/L
    alpha_scaled = alpha/L

    if Kc_sc is None:
        #---- You then compute Kc from Recipr space accuracy ----
        # Eq. 11 of J. Chem. Phys. 103, 3668–3679 (1995)
        Kc = 1.
        # error = 2*Q**2*alpha_scaled/L**2 * 1/np.sqrt(np.pi * N) * 1/np.sqrt(Kc) * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
        prefactor = 1
        error = prefactor * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
        while error > accuracy:
            Kc += 0.1
            error = prefactor * np.exp(-Kc**2 / (4*alpha**2)) * to_evA
        # we want to use scaled Kc
        Kcx_sc = int(np.ceil(Kc / np.linalg.norm(b1)))
        Kcy_sc = int(np.ceil(Kc / np.linalg.norm(b2)))
        Kcz_sc = int(np.ceil(Kc / np.linalg.norm(b3)))
        Kc_sc = [Kcx_sc, Kcy_sc, Kcz_sc]
        
            
    elif type(Kc_sc) == int: # then use the cutoff for all directions
        Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc, Kc_sc, Kc_sc
        Kc_sc = [Kcx_sc, Kcy_sc, Kcz_sc]
        Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
    
    else: # then the user provided 3 cutoffs
        Kcx_sc, Kcy_sc, Kcz_sc = Kc_sc
        Kc = [b1 @ Kc_sc , b2 @ Kc_sc , b3 @ Kc_sc]
    
    

    if verbose:
        print('Real space cutoff Rc [A]', np.round(Rc, 3))
        print('Alpha was set to [1/A]', np.round(alpha,3))
        print('\t=> gaussian screening std of [A]', np.round(1/np.sqrt(2) * 1/(alpha),3))
        print('Reciprocal space cutoffs Kc [2pi/L]', np.round(Kc,3))
        print('\t=> [{},{},{}] k-vectors'.format(Kcx_sc, Kcy_sc, Kcz_sc))
    
    return Rc, alpha, Kc_sc



def Ewald_summation(key,
             coords, 
             lattice_vectors, 
             acc_factor, 
             species_sequence,
             PBC=True,
             kmax=None, 
             rs=None,
             compute_forces=False,
             alpha=None,
             Rc=None,
             verbose=True):
    '''
    filename:              example json
    acc_factor (float):    accuracy of the ewald methods required
    species_sequence:       sequence of species from json
    kmax(float):            cutoff for the k-space sum. Default to none and will be 
                            estimated according to references: Mol. Simul., 1:207–224, 1988. 
                            and Frankel, Understanding Molecular Simulation:
    compute_forces (bool): set to True if forces are required and False otherwise
    set gaussian width to covalent radius from mendeleev
    '''
    N = len(coords)
    
    
    
    #structure properties
    volume, recip_lattice_vectors = compute_structure_props(lattice_vectors)
    # Set the Ewald parameters
    Rc, alpha, Kc_sc = set_ewald_parameters(N, lattice_vectors, acc_factor, Rc=Rc, alpha=alpha, Kc_sc=kmax, verbose=verbose, q=None)

    output = {'key':key}
    
    if not PBC: 
        LR_terms = real_space_term(1/(np.sqrt(2) * alpha),  coords, compute_forces)
        output['el_energy_kernel'] = LR_terms[0] 
        if compute_forces:
            output['el_force_kernel'] = LR_terms[1] 
    
    else:
        # do an Ewald summation
        LR_terms_rec = recip_space_term(Kc_sc,alpha,volume, coords, lattice_vectors, recip_lattice_vectors, acc_factor,compute_forces)
        LR_terms_real = real_space_term(alpha, lattice_vectors, coords, Rc, compute_forces=compute_forces)
        E_self = - 2* np.sqrt(alpha**2 / np.pi) * np.eye(N) * CONV_FACT
        
        # print('LLLLLLL', LR_terms_rec[0] + LR_terms_real[0] + E_self)
        output['el_energy_kernel'] = LR_terms_rec[0] + LR_terms_real[0] + E_self
        # output['el_energy_kernel'] = [LR_terms_real[0] , LR_terms_rec[0] , E_self]
        if compute_forces:
            output['el_force_kernel'] = LR_terms_rec[1] + LR_terms_real[1]
            # output['el_force_kernel'] = [LR_terms_real[1] , LR_terms_rec[1] ]
    
    return output




def recip_space_term(Kc_sc, 
        alpha,
        volume, coords, lattice_vectors,
        recip_lattice_vectors,
        accuracy,
        compute_forces=False):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in unit of angstrom
        kmax : cutoff for the recip space sum
        rmax : cutoff for the real space sum
        gaussian_width: array of gaussian widths
        volume: unitcell volume
        coords : atomic coordinates 
        recip_lattice_vectors: reciprocal lattice vectors 

    Return:
        reciprocal space contribution to energy 
    '''
    
    print('Reciprocal space sum')
    # # Start tracing memory allocations
    # tracemalloc.start()
    
    rij = coords[:,np.newaxis] - coords
    
    kcx, kcy, kcz = Kc_sc[0], Kc_sc[1], Kc_sc[2]

    # k_max = int(np.ceil(Kc))
    # kmax = np.linalg.det(recip_lattice_vectors)**(1/3) * k_max

    l_list = range(-kcx, kcx + 1)
    m_list = range(-kcy, kcy + 1)
    n_list = range(-kcz, kcz + 1)
    replicas = [x for x in itertools.product(l_list, m_list, n_list)]
    # 0 0 0 is removed, g0ik = 0
    comb = np.array(replicas[:int(len(replicas)/2)])
    
    
    k_vects = comb @ recip_lattice_vectors
    k_norms = np.linalg.norm(k_vects, axis=1)
    
    
    
    
    # print(rij.shape)
    # print(k_vects.shape)
    # print('hhhhh')
    # exp = np.exp(1j * rij @ k_vects.T) / 1
    exp = 2 * np.cos(rij @ k_vects.T)
    ew_factor = 1/k_norms**2 * np.exp(-k_norms**2/(4*alpha**2)) * (4.* np.pi / volume)
    
    
    en_kernel = np.sum( exp * ew_factor, axis=2)
    en_kernel *= CONV_FACT 
    
    if compute_forces:
        force_tmp = 2. * np.sin(rij @ k_vects.T) * ew_factor
        force_kernel = np.matmul(force_tmp, k_vects)
        force_kernel *= CONV_FACT 
        return [en_kernel, force_kernel]
    
    # # Current and peak memory usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 1024 / 1024 /1024:.1f} GB")
    # print(f"Peak usage: {peak / 1024 / 1024/ 1024:.1f} GB")

    # tracemalloc.stop()
    return [en_kernel] 



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

    #convert to angstrom * eV so that q1q2/r is in eV when r is in angstrom and
    # q1, q2 are in electronic charge unit
    # L = np.max(np.linalg.norm(lattice_vectors, axis=1))
    
    # mult = int(np.ceil(Rc/L))
    # coords = get_Scells(lattice_vectors, coords, nrep=mult).reshape((-1,3))
    # lattice_vectors = lattice_vectors * mult
    Natoms = len(coords)
    

    if Natoms == 1:
        rij = np.zeros(3)
        rij_norm = np.linalg.norm(rij, axis=-1)
    else:
        # rij = coords[:,np.newaxis] - coords
        en_kernel, force_kernel = get_el_kernels(coords, lattice_vectors, Rc, alpha, erfc)


    V = en_kernel
    V *= CONV_FACT
    

    if compute_forces:
        
        F = force_kernel.T
        F *= CONV_FACT
        return [V, F]
    return [V]






def compute_structure_props(lattice_vectors):
    '''Calculuate the volume, reciprocal lattice vectors
    Args:
        lattice_vectors: lattice vectors
   
    Returns:
        reciprocal lattice vectors,volume
    '''
    #convert to angstrom
    recip_lattice_vectors = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    volume = np.abs(np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])))
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

def screening_energy_term(lattice_vectors, 
                        rij, gamma_all, 
                        pbc, rs, compute_forces=False):

    CONV_FACT = 1e10 * constants.e / (4 * np.pi * constants.epsilon_0)
   
    Natoms = len(gamma_all[0])
    V = np.zeros((Natoms,Natoms))

    if compute_forces:
        forces = np.zeros((Natoms,Natoms,3))
    if not pbc:
    #if False in pbc: 

        rij_norm = np.linalg.norm(rij, axis = -1)

        V = -2 * gamma_all / np.sqrt(np.pi) * (1.0 - gsc(rij_norm, rs))
        if compute_forces:
            # Here I use the 1-gausian function for the screening
            # prefer this function as its smoother
            forces_norm = -2. * gamma_all / np.sqrt(np.pi) * dgsc(rij_norm, rs)
            forces_norm = np.tile(forces_norm[:,:,np.newaxis], 3)
            forces = forces_norm * rij
            return [V * CONV_FACT, forces * CONV_FACT]
        return [V * CONV_FACT]


    max_indices = replicas_max_idx(lattice_vectors, rs)
    l_max, m_max, n_max = max_indices
   
    l_list = range(-l_max, l_max + 1)
    m_list = range(-m_max, m_max + 1)
    n_list = range(-n_max, n_max + 1)

    #loop over points in R space
    for l, m, n in itertools.product(l_list, m_list, n_list):
        R_vect = l * lattice_vectors[0] +\
                 m * lattice_vectors[1] +\
                 n * lattice_vectors[2]
        
        rij_pcell = rij + R_vect
        rij_pcell_norm = np.linalg.norm(rij_pcell, axis=-1)

        V -= 2 * gamma_all / np.sqrt(np.pi) * (1.0 - gsc(rij_pcell_norm, rs))
        if compute_forces:
            forces_norm = -2 * gamma_all / np.sqrt(np.pi) * dgsc(rij_pcell_norm, rs)
            forces_norm = np.tile(forces_norm[:,:,np.newaxis], 3)
            forces += forces_norm * rij_pcell
    if compute_forces:
        return [V * CONV_FACT, forces * CONV_FACT]
    return [V * CONV_FACT]






def recip_space_termaaa(Kc, 
        alpha,
        volume, coords,
        recip_lattice_vectors,
        accuracy,
        compute_forces=False):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in unit of angstrom
        kmax : cutoff for the recip space sum
        rmax : cutoff for the real space sum
        gaussian_width: array of gaussian widths
        volume: unitcell volume
        coords : atomic coordinates 
        recip_lattice_vectors: reciprocal lattice vectors 

    Return:
        reciprocal space contribution to energy 
    '''
    print('Reciprocal space sum')
    
    # # Start tracing memory allocations
    # tracemalloc.start()
    
    Natoms = len(coords)
    gaussian_width = np.tile(alpha, (Natoms, Natoms))

    # compute the norm of all distances
    # the force on k due to atoms j is  rk-ri
    if Natoms == 1:
        rij = np.zeros(3)
        gamma_all = 1./np.sqrt(2.)* alpha
    else:
        # print(coords.shape)
        rij = coords[:,np.newaxis] - coords
        # gamma_all = 1.0/np.sqrt(gaussian_width[:,np.newaxis]**2 + np.asarray(gaussian_width)**2)
        gamma_all = gaussian_width
    

    
    k_max = int(Kc) + (Kc % int(Kc) > 0)
    kmax = np.linalg.det(recip_lattice_vectors)**(1/3) * k_max

    # #Hartree energy in kspace
    # max_indices = replicas_max_idx(recip_lattice_vectors, kmax)
    # l_max, m_max, n_max = max_indices
    # print(l_max, m_max, n_max)

    
    l_list = range(-k_max, k_max + 1)
    m_list = range(-k_max, k_max + 1)
    n_list = range(-k_max, k_max + 1)


    replicas = [x for x in itertools.product(l_list, m_list, n_list)]
    # 0 0 0 is removed, g0ik = 0
    replicas = np.array(replicas[:int(len(replicas)/2)])
    

    k_vects = replicas @ recip_lattice_vectors
    k_norms = np.linalg.norm(k_vects, axis=-1)
    k_norms_in_ball = k_norms
    ks_in_ball = k_vects
    
    
    

    # restrict to ball
    idxes_in_ball = k_norms < kmax
    ks_in_ball =  k_vects[idxes_in_ball]
    k_norms_in_ball = k_norms[idxes_in_ball]
    # print(k_norms)
    # print(kmax)
    # print(alpha)
    


    gauss_term = (np.exp(-(k_norms_in_ball[:, np.newaxis, np.newaxis]**2 / (4.0 * gamma_all**2)))
            /k_norms_in_ball[:, np.newaxis, np.newaxis]**2).T

    # [i, j, 3] @ [n_k, 3].T = [i, j, n_k]
    # tmp = gauss_term.copy()
    _arg = rij @ ks_in_ball.T
    # tmp *= 2. * np.cos(_arg)
    gauss_term *= 2. * np.cos(_arg)
    

    # [i, j, n_k]
    # contract n_k
    # V = np.sum(tmp, axis=-1)
    V = np.sum(gauss_term, axis=-1)
    
    # # Current and peak memory usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 1024 / 1024 /1024:.1f} GB")
    # print(f"Peak usage: {peak / 1024 / 1024/ 1024:.1f} GB")
    # # del(tmp)


    if compute_forces:
        
        force_tmp = 2. * np.sin(_arg) * gauss_term
        #forces = force_tmp[:,:,np.newaxis,:] * ks_in_ball.T[np.newaxis,np.newaxis,:,:]
        forces = np.matmul(force_tmp, ks_in_ball)
        #forces = np.sum(forces, axis=-1)
        del(force_tmp)
        del(gauss_term)


    V *= CONV_FACT * (4.* np.pi / volume)
    
    if compute_forces:
        forces *= CONV_FACT * (4.* np.pi / volume)
        return [V, forces, rij, gamma_all]
    
    # # Current and peak memory usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage: {current / 1024 / 1024 /1024:.1f} GB")
    # print(f"Peak usage: {peak / 1024 / 1024/ 1024:.1f} GB")

    # tracemalloc.stop()
    return V
