""" WIP
module to compute long range interaction
"""
import itertools
import logging

import numpy as np
import scipy.constants as constants

from panna.gvector.pbc import replicas_max_idx

logger = logging.getLogger(__name__)

def gauss_charges(lattice_vectors, coords, charges, widths, kmax, acc_factor=1e-6):
    """ compute the energy due to gaussian-gaussain charge interaction
    ~ Ewald summation

    Parameters
    ----------
    lattice_vecotrs: np.array(3,3), Angsrom
        [a1, a2, a3]
    coords: np.array(nat,3), Angstrom
    charges: np.array(nat)
        charge in units of e
    widths: np.array(nat), Angsrom
        Size of the gaussian for each atom
    kmax: an inital hint for k, 1/Angsrom
        This number can be incresed by the routine.
        see accuracy facotr
    accuracy factor:
        if the first excluded term form the sum in the reciprocal
        space is > accuracy factor the function will increase kmax
        till it met the required accuracy.

    Returns
    -------
    energy:
        eV
    """

    alphas = 2/widths**2
    rij = coords[:, np.newaxis,:] - coords

    num = alphas[:, np.newaxis] + np.asarray(alphas)
    den = alphas[:, np.newaxis] * np.asarray(alphas)
    alpha_ij = num/den

    # simple estimate of the error
    # (~contribution of first excluded therm of the sum)
    alpha_max = np.max(alpha_ij)
    err_function = lambda kmax:np.exp(-kmax**2/4*alpha_max)
    print(f'initial error is {err_function(kmax):5.4} and kmax is {kmax:5.2f}')
    while err_function(kmax) > acc_factor:
        kmax += 0.1

    print(f'final   error is {err_function(kmax):5.4} and kmax is {kmax:5.2f}')

    # [b1, b2, b3]
    recip_lattice_vectors = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    volume = np.abs(np.linalg.det(lattice_vectors))

    # creation of all the idx replicas
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
    n_atoms = len(coords)
    tmp = np.zeros((n_atoms, n_atoms, len(ks_in_ball)))
    print(f'matrix size estimate... sorry, better impl. needed\n'
          f'{int((tmp.size * tmp.itemsize)/1024/1024)} Mb')

    tmp += (np.exp(-(k_norms_in_ball[:, np.newaxis, np.newaxis]**2/ 4.0) * alpha_ij)
            /k_norms_in_ball[:, np.newaxis, np.newaxis]**2).T

    # [i, j, 3] @ [n_k, 3].T = [i, j, n_k]
    tmp *= 2 * np.cos((rij @ ks_in_ball.T))

    # [i, j, n_k]
    # contract n_k
    elements = np.sum(tmp, axis=-1)
    del(tmp)

    # charges-charges interaction
    u1 = 2 * np.pi * np.sum(charges[:, np.newaxis]*charges * elements) / volume

    # self interaction
    u2 = 1/np.sqrt(2 * np.pi) * np.sum(charges**2 * np.sqrt(alphas))

    #gaussian untis to ev:
    conv_factor = 1e10 * constants.e / (4 * np.pi * constants.epsilon_0)

    return (u1 - u2) * conv_factor
