###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import itertools

def orthogonal_vector(input_vector):
    """ calculate a random orthogonal unit vector to input_vector
        by solving the equation ax * x + ay * y + az * z = 0
        with input_vector = [ax, ay, az]
        candidate_vector = [x, y, z]

    Parameters
    ----------
    input_vector : 1D array
        a vector

    Returns
    -------
    1D array
        candidate_vector, random unit vector
    """
    candidate_vector = np.zeros(3)
    count = 0  # check for first non-zero element of input_vector
    for idx in range(3):

        if input_vector[idx] != 0 and count < 1:
            # choose only one of the components of candidate vector and the
            # other one is zero (this is sufficient !)
            candidate_vector[(idx + 1) % 3] = np.random.random(1)
            # candidate_vector[(idx + 2) % 3] = np.random.random(1)

            candidate_vector[idx] = -(candidate_vector[(idx + 1) % 3]
                                      * input_vector[(idx + 1) % 3]
                                      + candidate_vector[(idx + 2) % 3]
                                      * input_vector[(idx + 2) % 3]) \
                / input_vector[idx]
            count += 1

    candidate_vector = candidate_vector / np.linalg.norm(candidate_vector)

    return candidate_vector


def replicas_max_idx(lattice_vectors, Rc, pbc=[True, True, True]):
    """ calculate maximum number of cells needed with a given radial cutoff

    Parameters
    ----------
    lattice_vectors: lattice vectors as matrix (a1, a2, a3)
    Rc: maximum radial cutoff that you want to take in to account
    pbc: Normally pbc are recovered from lattice vector,
         if in the lattice_vectors a direction is set to zero
         then no pbc is applied in that direction.
         This argument allow you to turn off specific directions
         in the case where that specific direction has a lattice vector
         greater then zero.
         To achieve this pass an array of 3 logical value (one for each
         direction). False value turn off that specific direction.
         Default is true in every direction.
         eg. pbc = [True, False, True] => pbc along a1 and a3

    Returns
    -------
    max_indices: [lmax, mmax, nmax], numpy array
       integers for the number of replicas.
    """
    if not isinstance(lattice_vectors, np.ndarray):
        _lattice_vectors = np.asarray(lattice_vectors)
    else:
        _lattice_vectors = lattice_vectors.copy()

    lattice_vector_lengths = np.linalg.norm(_lattice_vectors, axis=1)
    lattice_vector_lengths_bool = lattice_vector_lengths > 1e-6
    max_indices = np.zeros(3, dtype=int)
    if not lattice_vector_lengths_bool.any():
        return max_indices

    lattice_vectors_idxs = np.where(lattice_vector_lengths_bool)[0]
    lattice_vectors_idxs_false = np.where(~lattice_vector_lengths_bool)[0]

    # define an index control
    lat_vec_idx_control = len(lattice_vectors_idxs)

    for idx in lattice_vectors_idxs_false:

        # when only one vector is has a nonzero length
        if lat_vec_idx_control == 1:

            j_idx = lattice_vectors_idxs[0]
            # orthogonal vector to _lattice_vectors[j_idx
            _lattice_vectors[idx] = orthogonal_vector(_lattice_vectors[j_idx])

        # when two vectors have nonzero length/given
        # compute the  cross_product  of the two given vectors

        if lat_vec_idx_control == 2:
            _lattice_vectors[idx] = np.cross(
                _lattice_vectors[(idx - lat_vec_idx_control) % 3],
                _lattice_vectors[(idx - lat_vec_idx_control + 1) % 3])
            _lattice_vectors[idx] = _lattice_vectors[idx] \
                / np.linalg.norm(_lattice_vectors[idx])
        lat_vec_idx_control += 1

    reciprocal_vectors = np.linalg.inv(_lattice_vectors)
    reciprocal_vectors_length = np.linalg.norm(reciprocal_vectors, axis=0)

    for idx in lattice_vectors_idxs:
        if pbc[idx]:
            b_length = reciprocal_vectors_length[idx]
            max_indices[idx] = int(np.ceil(Rc * b_length))

    return max_indices


def make_replicas(positions, lattice_vectors, Rc, pbc=[True, True, True]):
    """ Function computing directly all the replicas needed to contain
        all atom images within the cutoff.
        Also wraps original positions for convenience

    Parameters
    ----------
    lattice_vectors: lattice vectors as matrix (a1, a2, a3)
    Rc: maximum radial cutoff that you want to take in to account
    pbc: Normally pbc are recovered from lattice vector,
         if in the lattice_vectors a direction is set to zero
         then no pbc is applied in that direction.
         This argument allow you to turn off specific directions
         in the case where that specific direction has a lattice vector
         greater then zero.
         To achieve this pass an array of 3 logical value (one for each
         direction). False value turn off that specific direction.
         Default is true in every direction.
         eg. pbc = [True, False, True] => pbc along a1 and a3

    Returns
    -------
    positions: the wrapped atomic positions
    replicas: numpy array of displacement vectors for each replica.
    """
    lattice_vectors = np.asarray(lattice_vectors)
    # wrap the atoms in the unit cell
    if (lattice_vectors == 0).all():
        pass
    else:
        crystal_coord = positions @ np.linalg.inv(lattice_vectors)
        crystal_coord = crystal_coord % 1.0
        positions = crystal_coord @ lattice_vectors

    # compute how many cell replica we need in each direction
    max_indices = replicas_max_idx(lattice_vectors, Rc, pbc)
    l_max, m_max, n_max = max_indices
    l_list = range(-l_max, l_max + 1)
    m_list = range(-m_max, m_max + 1)
    n_list = range(-n_max, n_max + 1)
    # create a matrix with all the idx of the extra cell we need
    replicas = np.asarray(list(itertools.product(l_list, m_list, n_list)))

    # translation vectors needed to map each atom in the unit cell to its replica
    # @ means matrix multiplication
    replicas = replicas @ lattice_vectors

    # Keep only the replicas that are actually inside cutoff
    # Since we use all atoms in the tf code, this can be a good speedup
    corners = np.asarray(list(itertools.product([0,1],[0,1],[0,1]))) @ lattice_vectors
    keep_replicas = []
    for r in replicas:
        distances = np.sum((corners[:,np.newaxis,:] - \
                            corners[np.newaxis,:,:] - \
                            r[np.newaxis,np.newaxis,:])**2,axis=2)
        if np.min(distances)<Rc**2:
            keep_replicas.append(r)

    return positions, np.asarray(keep_replicas)



# def get_Rij(R0, latvec, nrep=3):
#     """
#     returns the R matrix
#     """
#     # N1N2N3 = N1 * N2 * N3
#     N = len(R0)
#     Scells = get_Scells(latvec, R0, nrep=nrep)
#     # print(Scells)
    
    
#     Rij = np.zeros((N,N))
#     xyzij = np.zeros((3,N,N))
#     for i in range(N):
#         atom_origin = R0[i]

#         for j in range(N):
#             dist_Scells_vec = Scells[:,j,:] - atom_origin
#             dist_Scells = np.linalg.norm(dist_Scells_vec, axis=1)
#             found_min_dist = np.argwhere(dist_Scells == np.min(dist_Scells, axis=0))
#             found_min_dist_0 = found_min_dist[0,0]
#             nearest_neigh = Scells[found_min_dist_0,j,:]

#             Rij[i,j] = np.linalg.norm(nearest_neigh - atom_origin)
#             xyzij[:,i,j] = nearest_neigh - atom_origin
            
            
#     return Rij, xyzij



def get_el_kernels(R0, latvec, cutoff, alpha, error_func):
    N = len(R0)
    Lmax = np.max(np.linalg.norm(latvec, axis=1)) #lattice vectors must be read horizontally
    max_rep_right = int( (Lmax + cutoff) // Lmax ) # maximum amount of cell replicas on the right
    max_rep = 2*max_rep_right+1
    Scells = get_Scells(latvec, R0, nrep=max_rep)
    
    energy_kernel = np.zeros((N,N))
    force_kernel = np.zeros((3,N,N))
    for i in range(N):
        ref_atm = np.expand_dims( R0[i], 0)
        for j in range(i+1):  
            
            dist_Scells_vec = Scells[:,j,:] - ref_atm
            dist_Scells = np.linalg.norm(dist_Scells_vec, axis=-1)
            
            condition = np.logical_and(dist_Scells < cutoff, dist_Scells > 1e-9)
            found_min_dist = np.argwhere(condition).flatten()
            # found_min_dist = np.argwhere(dist_Scells == dist_Scells.min()).flatten()[0]
            

            nearest_neigh = Scells[found_min_dist,j,:]
            neighbours_vecs = ref_atm - nearest_neigh 
            neighbours_dist = np.linalg.norm(neighbours_vecs, axis=1)
            
            
            erfc_over_rij  = error_func(alpha*neighbours_dist)/neighbours_dist
            erfc_over_rij3 = error_func(alpha*neighbours_dist)/neighbours_dist**3 
            gauss_term_ij = 2*alpha / np.sqrt(np.pi) * np.exp(-alpha**2*neighbours_dist**2) /neighbours_dist**2
            
            
            # sum_inverse_neighbours_dist = np.sum(1/neighbours_dist)
            # sum_erfcij = np.sum(erfcij)
            # sum_gauss_term_ij = np.sum(gauss_term_ij)
            
            force_ij = -(erfc_over_rij3+gauss_term_ij) @ neighbours_vecs
            e_ij = np.sum(erfc_over_rij)
            # avg_neighbours_vec = np.sum(neighbours_vecs/np.expand_dims(neighbours_dist,1), axis=0)
            
            # xyzij[:,i,j] = avg_neighbours_vec 
            # rij_inv[i,j] = sum_inverse_neighbours_dist
            # erf_matrix[i,j] = sum_erfcij
            # gauss_matrix[i,j] = sum_gauss_term_ij
            
            force_kernel[:,i,j] = force_ij 
            energy_kernel[i,j] = e_ij

    for i in range(3):
        diag = force_kernel[i].diagonal()      
        force_kernel[i] = force_kernel[i] - force_kernel[i].T
        np.fill_diagonal(force_kernel[i], diag)


    diag = energy_kernel.diagonal()
    energy_kernel = energy_kernel + energy_kernel.T
    np.fill_diagonal(energy_kernel, diag)

    return energy_kernel, force_kernel



def get_Rij_old(R0, latvec, nrep=3):
    """
    returns the R matrix
    """
    N = len(R0)
    Scells = get_Scells(latvec, R0, nrep=nrep)
    
    
    Rij = np.zeros((N,N))
    xyzij = np.zeros((3,N,N))
    for i in range(N):
        atom_origin = R0[i]

        for j in range(i):
            # print(i,j)
            dist_Scells_vec = Scells[:,j,:] - atom_origin
            dist_Scells = np.linalg.norm(dist_Scells_vec, axis=1)
            found_min_dist = np.argwhere(dist_Scells == np.min(dist_Scells, axis=0))
            found_min_dist_0 = found_min_dist[0,0]
            nearest_neigh = Scells[found_min_dist_0,j,:]
            
            
            Rij[i,j] = np.linalg.norm(nearest_neigh - atom_origin)
            xyzij[:,i,j] = nearest_neigh - atom_origin
            
    Rij = Rij + Rij.T
    xyzij[0] = xyzij[0] - xyzij[0].T
    xyzij[1] = xyzij[1] - xyzij[1].T
    xyzij[2] = xyzij[2] - xyzij[2].T

    return Rij, xyzij




def get_Scells(H, R0, nrep=3):
    N = len(R0)
    xl = nrep // 2 
    x1 = np.arange(-xl,-xl+nrep, dtype=int)
    comb = np.array(np.meshgrid(x1,x1,x1)).T.reshape(-1,3)
    displ = np.dot(H, comb.T).T
    
    Scells = np.zeros((nrep**3, N, 3))
    for c in range(nrep**3):
        Scells[c] = R0 + displ[c]     
    return Scells

