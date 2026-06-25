import numpy as np
import json
import os
import itertools
from mendeleev import element
import time
from .pbc import replicas_max_idx

from scipy.special import erf
import scipy.constants as constants

e_charge = 1.60217662e-19 
epsilon_0 = 8.85418782e-12 #F/m unit

def Ewald_summation(key,
             coords, 
             lattice_vectors, 
             acc_factor, 
             species_sequence,
             PBC=None,
             kmax=None, 
             rs=None,
             compute_forces=False):
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
    covalent_radii = [element(x).covalent_radius for x in species_sequence]
    #convert to gaussian width and to angstrom unit 
    #multiply by  rt(2) because the gaussian is defined with
    gaussian_width = np.asarray(covalent_radii) * 0.01 * np.sqrt(2.0)
    #structure properties
    lattice_length = np.linalg.norm(lattice_vectors,axis=1)
    _rs = rs if rs else 0.0
    _PBC = PBC if PBC else False

    if _PBC:
        struct_prop = compute_structure_props(lattice_vectors)
        volume, recip_lattice_vectors = struct_prop

        gamma_max = 1/(np.sqrt(2.0)*np.min(gaussian_width))
        _kmax = kmax if kmax else 2.* gamma_max * np.sqrt(-np.log(acc_factor))

    output = {'key':key}
    if not _PBC: 

        LR_terms = real_space_term(gaussian_width, 
                                  coords, 
                                  compute_forces)
        E_sc = np.zeros(2)
        if _rs > 0:
            E_sc = screening_energy_term(lattice_vectors,
                  LR_terms[2], LR_terms[3], _PBC, 
                  _rs,
                  compute_forces=compute_forces)

        output['el_energy_kernel'] = LR_terms[0] + E_sc[0]
        if compute_forces:
            output['el_force_kernel'] = LR_terms[1] + E_sc[1]
        return output

    LR_terms = recip_space_term(_kmax,
                                gaussian_width,
                                volume, 
                                coords, 
                                recip_lattice_vectors, 
                                acc_factor,
                                compute_forces)
    E_sc = np.zeros(2)
    if _rs > 0:
        E_sc = screening_energy_term(lattice_vectors,
                  LR_terms[2], LR_terms[3], _PBC, 
                  _rs,
                  compute_forces=compute_forces)

    output['el_energy_kernel'] = LR_terms[0] + E_sc[0]
    if compute_forces:
        output['el_force_kernel'] = LR_terms[1] + E_sc[1]
    return output

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

def real_space_term(gaussian_width,
                      coords,
                      compute_forces=False):
    '''
    calculates the self interaction contribution to the electrostatic energy
    Args:
        All lengths are in angstrom
        gaussian_width: gaussian width, array of atomic gussian width
        coords : atomic coordinates
    Return:
        real space contribution to energy
    '''

    #convert to angstrom * eV so that q1q2/r is in eV when r is in angstrom and
    # q1, q2 are in electronic charge unit
    #1e10 comes from angstrom
    #CONV_FACT = 1e10 * e_charge / (4 * np.pi * epsilon_0)
    CONV_FACT = 1e10 * constants.e / (4 * np.pi * constants.epsilon_0)

    Natoms = len(gaussian_width)

    coords = np.asarray(coords)
    if Natoms == 1:
        rij = np.zeros(3)
        gamma_all = 1./np.sqrt(2.)* 1./gaussian_width
    else:
        rij = coords[:,np.newaxis] - coords
        gamma_all = 1.0/np.sqrt(gaussian_width[:,np.newaxis]**2 + np.asarray(gaussian_width)**2)

    #compute the norm of all distances
    rij_norm = np.linalg.norm(rij, axis=-1)
    #trick to set the i==j element of the matrix to zero
    rij_norm_inv = np.nan_to_num(1 / rij_norm, posinf=0, neginf=0)

    #erf_term = np.where(rij_norm>0, erf(gamma_all*rij_norm)/rij_norm, 0)
    erf_term = erf(gamma_all*rij_norm) * rij_norm_inv
    V = erf_term.copy()

    V *=CONV_FACT
    E_diag = CONV_FACT * 2 / np.sqrt(np.pi) * 1/(np.sqrt(2.0)*gaussian_width)
    diag_indexes = np.diag_indices(Natoms)
    V[diag_indexes] = E_diag

    if compute_forces:
        #rij_tmp = np.where(rij_norm>0, 1/rij_norm**2,0)
        
        rij_tmp = rij_norm_inv * rij_norm_inv
        F_erf_part = erf_term * rij_tmp
        F_gaussian_part = -2.0/np.sqrt(np.pi) * gamma_all * np.exp(-gamma_all**2*rij_norm**2) * rij_tmp
        F_norm = F_erf_part + F_gaussian_part
        F_norm = np.tile(F_norm[:,:,np.newaxis], 3)
        force = rij * F_norm
        return [V, CONV_FACT*force, rij, gamma_all]
    return [V, rij, gamma_all]

def recip_space_term(kmax, 
        gaussian_width,
        volume, coords,
        recip_lattice_vectors,
        acc_factor,
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

    #convert to angstrom * eV so that 1/r is in eV when r is in angstrom and
   
    #CONV_FACT = 1e10 * e_charge / (4 * np.pi * epsilon_0)
    CONV_FACT = 1e10 * constants.e / (4 * np.pi * constants.epsilon_0)
        
    Natoms = len(gaussian_width)

    #compute the norm of all distances
    # the force on k due to atoms j is
    # rk-ri
    coords = np.asarray(coords)
    if Natoms == 1:
        rij = np.zeros(3)
        gamma_all = 1./np.sqrt(2.)* 1./gaussian_width
    else:
        rij = coords[:,np.newaxis] - coords
        gamma_all = 1.0/np.sqrt(gaussian_width[:,np.newaxis]**2 + np.asarray(gaussian_width)**2)

   
    # refine kmax
    gamma_max = np.max(gamma_all)
    err = np.exp(-kmax**2/(4*gamma_max**2))
    while err > acc_factor:
        kmax += 0.1
        err = np.exp(-kmax**2/(4*gamma_max**2))
    kmax *= 1.00001

    #Hartree energy in kspace
    max_indices = replicas_max_idx(recip_lattice_vectors, kmax)
    l_max, m_max, n_max = max_indices
   
    l_list = range(-l_max, l_max + 1)
    m_list = range(-m_max, m_max + 1)
    n_list = range(-n_max, n_max + 1)

    replicas = [x for x in itertools.product(l_list, m_list, n_list)]

    if replicas[int(len(replicas)/2)] != (0, 0, 0):
        # this should never happen, the number of replicas is
        # always odd, replicas must be in lexichographic
        # order
        raise ValueError('something is wrong with replicas generator')

    # 0 0 0 is removed, g0ik = 0

    replicas = np.array(replicas[:int(len(replicas)/2)])

    k_vects = replicas @ recip_lattice_vectors
    k_norms = np.linalg.norm(k_vects, axis=-1)

    # restrict to ball
    idxes_in_ball = k_norms < kmax
    ks_in_ball =  k_vects[idxes_in_ball]
    k_norms_in_ball = k_norms[idxes_in_ball]

    # [i, j, n_k]
    #gauss_term = np.zeros((Natoms, Natoms, len(ks_in_ball)))

    gauss_term = (np.exp(-(k_norms_in_ball[:, np.newaxis, np.newaxis]**2 / (4.0 * gamma_all**2)))
            /k_norms_in_ball[:, np.newaxis, np.newaxis]**2).T

    # [i, j, 3] @ [n_k, 3].T = [i, j, n_k]
    tmp = gauss_term.copy()
    _arg = rij @ ks_in_ball.T
    tmp *= 2. * np.cos(_arg)

    # [i, j, n_k]
    # contract n_k
    V = np.sum(tmp, axis=-1)
    del(tmp)


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
    return [V, rij, gamma_all]
