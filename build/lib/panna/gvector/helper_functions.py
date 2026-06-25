###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""
Common helper functions used to compute the gvectors
"""
import numpy as np


def _cutoff(r_ij, r_c):
    return 0.5 * (1.0 + np.cos(np.pi * r_ij / r_c))


def _dcutoff(r_ij, r_c):
    return 0.5 * np.pi * np.sin(np.pi * r_ij / r_c) / r_c


def G_radial(r_ij, R_s, eta_rad, Rc_rad):
    g_rad = np.exp(-eta_rad * (r_ij - R_s)**2) * _cutoff(r_ij, Rc_rad)
    return g_rad


def GdG_radial(r_ij, R_s, eta_rad, Rc_rad):
    Gauss = np.exp(-eta_rad * (r_ij - R_s)**2)
    g_rad = Gauss * _cutoff(r_ij, Rc_rad)
    dg_rad = (_dcutoff(r_ij, Rc_rad) + (2.0 * eta_rad * _cutoff(r_ij, Rc_rad)
                                        * (r_ij - R_s))) * Gauss / r_ij

    return g_rad, dg_rad


def G_angular_mBP(r_ij, r_ik, theta_jik, R_p, theta_s, eta_ang, zeta, Rc_ang):
    """
    Just the function to calculate G_radial element
    """
    eps = 1e-3
    G = np.exp(-eta_ang * (0.5 * (r_ij + r_ik) - R_p)**2)
    corr = np.cos(theta_jik) * np.cos(theta_s) + \
        np.sqrt(1 - np.cos(theta_jik)**2 + eps * np.sin(theta_s)**2) *\
        np.sin(theta_s)
    norm = (2 / (1 + np.sqrt(1 + eps * np.sin(theta_s)**2)))**zeta
    G = G * 2.0 * norm * (0.5 + corr * 0.5)**zeta
    G = G * _cutoff(r_ij, Rc_ang)
    G = G * _cutoff(r_ik, Rc_ang)

    return G

def GdG_angular_mBP(r_ij, r_ik, theta_jik, R_p, cs, ss, eta_ang, zeta, Rc_ang,
                    prefactor):
    """
    Calculates G angular mBP and pieces of its derivative wrt r
    Cosine centers are shifted by half, differently than mBP reference.
    The formula is also corrected with epsilon to fix for the discontinuous derivative
    See PANNA reference paper
    cs = cos(theta_s)
    ss = sin(theta_s)
    """
    # the correction to make second derivative continuous as before
    eps = 1e-3
    # radial center of gaussian
    r_cent = (r_ij + r_ik) * 0.5 - R_p
    # shortnames
    # cs=np.cos(theta_s) ;
    # ss=np.sin(theta_s) ;
    ct = np.cos(theta_jik)  #cos_theta_jik
    st = np.sqrt(1 - ct**2 + eps * ss**2)  #sin_theta_jik_approx
    # normalization of eps correction
    norm = 1.0 / (1. + np.sqrt(1. + eps * ss**2))
    # components of G = Gauss x CosTerm x CutTerms
    Gauss = 2.0 * np.exp(-eta_ang * r_cent**2)
    # cos (t1 - t2) = cost1 cost2 + sint1 sint2 but use sin1=sin_theta_approx
    # cos_approx = ct*cs + st*ss
    onepcos = 1.0 + ct * cs + st * ss
    CosTerm = (onepcos * norm)**zeta

    Cut1 = _cutoff(r_ij, Rc_ang)
    Cut2 = _cutoff(r_ik, Rc_ang)
    G = Gauss * CosTerm * prefactor
    #dG_cos_tmp = Gauss * dCosTerm * prefactor


    Cut1xCut2 = Cut1 * Cut2
    # derivatives
    # dGi/drj = A r_ij + B r_ik
    # because  dTheta/dr_ij form CosTerm mixes these two components
    # So A has contribution from all three components while B comes from CosTerm only

    #the commented dcos has division by zero when
    #theta_s = 0 or pi since the regularization has no effects. 

    #Safer to write this way so that if ss = 0 and theta_ijk = 180, 
    #onepcos = 0 + 1e-8
    #st = 0
    # Therefore, (cs * st - ss * ct) == 0, hence no divergence
    #dcos = zeta / onepcos * (cs  - ss * ct / st)
    dcos = zeta / (onepcos * st + 1e-8) * (cs * st - ss * ct)
    # A

    dG1 = -G/r_ij * \
        ( eta_ang*r_cent * Cut1xCut2 \
           + dcos * ct /r_ij * Cut1xCut2 \
           + _dcutoff(r_ij, Rc_ang)*Cut2 )

    # First one from derivative of Gauss,
    # Second term from dCosTerm and below with some cancellations:
    # dcos_apprx/dr_j_a = (-sintheta) * (cs - ss*ct /st) \
    # * ( r_ik_a/|r_ij||r_ik| - costheta r_j_a / |rij||rij| ) * (-1/sintheta)
    # Previously:
    # Last term from Cuts, with multiply/divide
    # Gauss * Costerm * Cut2 * Cut1/Cut1 * _dcutoff(r_ij, Rc_ang)/r_ij
    # Now:
    # Due to numerical error Cut1 can be 0, so we had to remove the
    # cut1 * cut2 multiplication form the G

    # B
    dG2 = (G * Cut1xCut2) / (r_ij * r_ik) * dcos
    # j--> k dGi/drk =  A'r_ik + B r_ij

    # A'
    dG3 = -G/r_ik * \
          ( eta_ang*r_cent * Cut1xCut2 \
           + dcos * ct /r_ik * Cut1xCut2 \
           + _dcutoff(r_ik, Rc_ang) * Cut1 )
    # B is symmetric

    # Fixing the G values
    G *= Cut1xCut2

    return (G, dG1, dG2, dG3)

def G_angular_BP(r_ij, r_ik, r_jk, theta_ijk, eta_ang, zeta, lamb, Rc_ang):
    G = np.exp(-eta_ang * (r_ij**2 + r_ik**2 + r_jk**2))
    G = G * _cutoff(r_ij, Rc_ang)
    G = G * _cutoff(r_ik, Rc_ang)
    G = G * _cutoff(r_jk, Rc_ang)
    G = G * 2.0 * (0.5 + lamb * np.cos(theta_ijk) * 0.5)**zeta
    return G
