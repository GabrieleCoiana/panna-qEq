###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import jax.numpy as jnp
import jax
import itertools
from functools import partial


def get_indices_and_distances_of_pairs(pos, Scells, rcut, within_cutoff=False):
    """
    A simple function that takes reference cell's positions pos: (Nats, 3) and
    cell's images positions (Ncellimages, Nats, 3) and a short range cutoff rcut,
    and spits out pair relative positions xyzij: (Npairs, 3), 
    pair relative distances rij: (Npairs,) and indices indi,indj: (Npairs,) 
    of the pairs of atoms within the cutoff.
    It is the same as PANNA's inda, indb, nn_vecs, nn_r
    and the same as 
    from ase.neighbors import neighbor_list
    indi, indj, rij, xyzij = neighbor_list('ijdD', atoms, rcut)      
    
    Scells comes from panna.electrostatics.pbc_jax.get_Scells_images(), which 
    takes as input the number of ref. cell's images which you want to consider.
    Most of the times, you consider only the 3^3 nearest unit cells, but if the 
    cutoff is large you might need to consider 5^3 images, or more. 
    
    R = reference cell
    I = image cell
      ___  ___  ___
     | I ||   ||   |
     |___||___||___|
     |   || R ||   |
     |___||___||___|
     |   ||   ||   |
     |___||___||___|
     
    Scells is the positions of all atoms in all these 3^2 cells (2D example).
    
    The function checks that the rij where i belongs to R and j to any other cell
    is < rcut. If so, the pair is retained, otherwise it's thrown away.
    For this reason, this function is O(Nimages * Nats^2).
    In an MD simulation, this is not ideal. Usually neighbor list indices are 
    computed once in a while (e.g. skin), and the same indices can be reused!

    """
    # print('EWALDprep - Getting rijs list, vectorized version')
    xyzij = pos[None,  :, None] - Scells[:,None,:] # (Nimages, N, N, 3)
    rij = jnp.linalg.norm(xyzij, axis=-1)          # (Nimages, N, N)
    # remove rii numbers
    rij = jnp.where(rij < 1e-5, 1e18, rij) 
    
    
    mask = rij < rcut  
    if within_cutoff == False:        
        img_idx, indi, indj = jnp.where(mask, size=mask.size) #<-- this leaves the shapes unchanged
        rij = rij[img_idx, indi, indj]      # (Npairs,)
        xyzij = xyzij[img_idx, indi, indj]  # (Npairs, 3)
        
        n_valid = jnp.sum(mask)             # number of valid entries: rij < rcut
        valid = jnp.arange(mask.size) < n_valid     # True for real pairs only

        # populate the padded slots with the sentinel
        rij = jnp.where(valid, rij, 1e18)  # excluding rijs > rcut
        xyzij = jnp.where(valid[:, None], xyzij, 0.) # excluding xyzij > rcut
    else:
        img_idx, indi, indj = jnp.where(mask) # <-- this changes shape
        rij = rij[img_idx, indi, indj]
        xyzij = xyzij[img_idx, indi, indj]
    # rij = jnp.where(rij < rcut, rij, 1e19)
    # print(f'Nimages {len(Scells[:,0,0])}, Num of pair interactions {len(rij)}', )
    
    return indi, indj, rij, xyzij


def get_indices_and_distances_of_pairs_forloop(pos, Scells, rcut, within_cutoff=False):
    """
    with for loop to reduce memory usae
    """
    print('EWALDprep - Getting rijs list, for loop version')
    Nimages, N, _ = Scells.shape
    rijs = []
    xyzijs = []
    indis = []
    indjs = []
    for im in range(Nimages):
        print(im, rcut, 'aa')
        xyzij = xyzij = pos[:, None, :] - Scells[im][None, :, :] # (N, N, 3)
        rij = jnp.linalg.norm(xyzij, axis=-1)          # (N, N)
        # remove rii numbers
        rij = jnp.where(rij < 1e-5, 1e19, rij) 
        
        
        mask = rij < rcut  
        if within_cutoff == False:
            indi, indj = jnp.where(mask, size=mask.size) #<-- this leaves the shapes unchanged
            rij = rij[indi, indj]      # (Npairs,)
            xyzij = xyzij[indi, indj]  # (Npairs, 3)
            
            n_valid = jnp.sum(mask)             # number of valid entries: rij < rcut
            valid = jnp.arange(mask.size) < n_valid     # True for real pairs only

            # populate the padded slots with the sentinel
            rij = jnp.where(valid, rij, 1e18)  # excluding rijs > rcut
            xyzij = jnp.where(valid[:, None], xyzij, 0.) # excluding xyzij > rcut
        else:
            indi, indj = jnp.where(mask) # <-- this changes shape
            rij = rij[indi, indj]
            xyzij = xyzij[indi, indj]

        rijs.append(rij)
        xyzijs.append(xyzij)
        indis.append(indi)
        indjs.append(indj)
    
    indis = np.concatenate(indis)
    indjs = np.concatenate(indjs)
    rijs = np.concatenate(rijs)
    xyzijs = np.concatenate(xyzijs, axis=0)
    # print(f'Nimages {len(Scells[:,0,0])}, Num of pair interactions {len(rijs)}', )
    
    return indis, indjs, rijs, xyzijs

def estimate_maxNpairs_analytically(N, V, rcut, buffer_ratio=1.5):
    rho = N / V
    n_bar = rho * (4/3) * jnp.pi * rcut**3
    n_pairs = N * n_bar
    capacity = int(n_pairs * buffer_ratio) + 1
    print(f'capacity {capacity}')
    return capacity

def get_indices_and_distances_of_pairs_claude(pos, Scells, rcut, capacity, block_size=64):
    """
    Streams over blocks of reference-cell atoms i, so peak memory is
    O(Nimages * block_size * N) instead of O(Nimages * N * N).
    """
    float_dtype = pos.dtype                 # respects x64 automatically: float64 if enabled
    int_dtype   = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    N = pos.shape[0]
    n_blocks = (N + block_size - 1) // block_size
    N_pad = n_blocks * block_size
    pos_padded = jnp.pad(pos, ((0, N_pad - N), (0, 0)), constant_values=1e19)
    # padded atoms sit at 1e19 -> never within rcut of anything, safely excluded

    block_capacity = capacity // n_blocks + 1  # per-block pair budget

    def block_fn(carry, i_block):
        write_ptr, out_i, out_j, out_r, out_xyz = carry
        pos_i = jax.lax.dynamic_slice_in_dim(pos_padded, i_block * block_size,
                                              block_size, axis=0)  # (block, 3)

        xyzij = pos_i[None, :, None] - Scells[:, None, :]       # (Nimg, block, N, 3)
        rij = jnp.linalg.norm(xyzij, axis=-1)                   # (Nimg, block, N)
        rij = jnp.where(rij < 1e-5, 1e19, rij)
        mask = rij < rcut

        img_idx, li, lj = jnp.where(mask, size=block_capacity, fill_value=-1)
        valid = img_idx >= 0
        safe_img, safe_i, safe_j = img_idx.clip(0), li.clip(0), lj.clip(0)
        r_local   = jnp.where(valid, rij[safe_img, safe_i, safe_j], 1e19)
        xyz_local = jnp.where(valid[:, None], xyzij[safe_img, safe_i, safe_j], 0.0)
        gi_local  = jnp.where(valid, li + i_block * block_size, -1)  # global i index
        gj_local  = jnp.where(valid, lj, -1)
        
        # print(out_i.dtype, i_block.dtype, li.dtype)
        # print(out_r.dtype, r_local.dtype)
        # print(write_ptr.dtype if hasattr(write_ptr,'dtype') else type(write_ptr))
        
        n_local = jnp.sum(valid)
        out_i   = jax.lax.dynamic_update_slice_in_dim(out_i,   gi_local,  write_ptr, axis=0)
        out_j   = jax.lax.dynamic_update_slice_in_dim(out_j,   gj_local,  write_ptr, axis=0)
        out_r   = jax.lax.dynamic_update_slice_in_dim(out_r,   r_local,   write_ptr, axis=0)
        out_xyz = jax.lax.dynamic_update_slice_in_dim(out_xyz, xyz_local, write_ptr, axis=0)

        return (write_ptr + n_local, out_i, out_j, out_r, out_xyz), None

    init = (0,
            -jnp.ones(capacity, dtype=int_dtype),
            -jnp.ones(capacity, dtype=int_dtype),
            jnp.full(capacity, 1e19, dtype=float_dtype),
            jnp.zeros((capacity, 3), dtype=float_dtype))

    (_, indi, indj, rij_out, xyzij_out), _ = jax.lax.scan(block_fn, init, jnp.arange(n_blocks))
    valid = indi >= 0
    print(f'Num of pair interactions {len(rij_out)}', )
    
    return indi, indj, rij_out, xyzij_out#, valid




def load_example(folder):
    '''
    Returns:
    --------    
    spec_label: array of element labels per atom; new in y2ace  
    config_energy: array of total energy of configuration and units
    lattvecs: lattice vectors
    positions: position vectors of each atom
    forces: force vectors of each atom
    
    Parameters:
    -----------
    filedict: dictionary loaded from .example file

    '''
    import os, json
    
    files = os.listdir(folder)
    # nats = []

    for i, file in enumerate(files):
        with open(folder+file, 'r') as f:  
            filedict = json.load(f)

        # spec_label,config_energy,length_units,lattvecs,positions,forces = collect_from_json(example)
        # energy = config_energy[0]
        # nat = len(spec_label)
        
        
    length_units = filedict['unit_of_length']
    config_energy = filedict['energy']
    lattvecs = np.array(filedict['lattice_vectors'])
    species_data = filedict['atoms']
    spec_label = [sval[1] for sval in species_data] # species label list
    pos_forces = [] # parsing position, and forces  
    for i,sval in enumerate(species_data):
        pos_forces.append(sval[2:4])
    pos_forces = np.array(pos_forces) # pos_forces[atom][pos,force][x,y,z]
 	
    positions,forces = pos_forces[:,0],pos_forces[:,1]
    
    charges = {'Mg':1.4, 'O':-1.4}
    q0 = np.array([charges[a] for a in spec_label])

    return '', positions, lattvecs, 1e-06, spec_label, q0




# def get_Scells_images(latvec, R0, cutoff):
#     Lmax = jnp.max(jnp.linalg.norm(latvec, axis=1)) #lattice vectors must be read horizontally
#     max_rep_right = ( (Lmax + cutoff) // Lmax ).astype(int) # maximum amount of cell replicas on the right
#     Nmax_images = 2*max_rep_right+1
#     xl = max_rep_right
    
    
#     x1 = jnp.arange(-xl,-xl+Nmax_images, dtype=int)
#     comb = jnp.array(np.meshgrid(x1,x1,x1)).T.reshape(-1,3)
#     comb = jnp.vstack((comb, jnp.array([1e18, 1e18, 1e18])))
    
#     displ = jnp.dot(latvec, comb.T).T
#     displ = displ[:,jnp.newaxis,:]
#     Scells = displ + R0
#     return Scells, int(Nmax_images)**3


def get_Scells_images_oldversion(inp, latvec, R0):
    xl = jnp.shape(inp)[0]
    
    x1 = jnp.arange(-xl,xl+1, dtype=int)
    comb = jnp.array(jnp.meshgrid(x1,x1,x1)).T.reshape(-1,3)
    comb = jnp.vstack((comb, jnp.array([1e18, 1e18, 1e18])))
    
    displ = jnp.dot(latvec, comb.T).T
    displ = displ[:,jnp.newaxis,:]
    # displ2 = displ0[:,:,jnp.newaxis]
    Scells = displ + R0
    # Scells2 = displ2 + R0.T
    return Scells



def get_Scells_images(inp, latvec, R0):
    """
    

    Parameters
    ----------
    inp : TYPE
        DESCRIPTION.
    latvec : TYPE
        DESCRIPTION.
    R0 : TYPE
        DESCRIPTION.

    Returns
    -------
    Scells : TYPE
        DESCRIPTION.
        
    Remember this. The cell must be read in columns.

    """
    xl = jnp.shape(inp)[0]
    
    x1 = jnp.arange(-xl,xl+1, dtype=int)
    comb = jnp.array(jnp.meshgrid(x1,x1,x1)).T.reshape(-1,3) #(Ncells, 3)
    # latvec must contain a,b,c in COLUMNS
    displ = comb @ latvec # rows of latvec are lattice vectors
    displ = displ[:,jnp.newaxis,:]
    Scells = displ + R0
    return Scells



@partial(jax.jit, static_argnums=(2)) # @jax.jit
def get_rij(R0, Scells, Nmax_nn, latvec, cutoff):
    N = len(R0)
    # qq
    rij_qq = jnp.zeros((N,N,Nmax_nn)) 
    
    # qd
    xyzij_qd = jnp.zeros((N,N*3,Nmax_nn)) 
    rij_qd = jnp.zeros((N, N*3, Nmax_nn))
    
    # dd
    rij_dd_diag = jnp.ones((3*N, 3*N, Nmax_nn)) * jnp.inf
    rij_dd_full = jnp.zeros((3*N, 3*N, Nmax_nn))
    xyzij_dd_outer = jnp.zeros((3*N, 3*N, Nmax_nn)) 
    xyzij_dd_diag = jnp.zeros((3,3*N,3*N, Nmax_nn))
    xyzij_dd = jnp.zeros((3, 3*N, 3*N, Nmax_nn))
    xyzij_dd_derivatives = jnp.zeros((3,3*N,3*N, Nmax_nn))
    # xyzij_dd_outer_unitvec = jnp.zeros((3*N, 3*N, Nmax_nn)) 
    for i in range(N):
        ref_atm = R0[jnp.newaxis, i]
        for j in range(i+1):  
            dist_Scells_vec = Scells[:,j,:] - ref_atm
            dist_Scells = jnp.linalg.norm(dist_Scells_vec, axis=-1)
                        
            # the -0.1 is because we are working with float32
            condition = jnp.logical_and(dist_Scells < cutoff-0.1, dist_Scells > 1e-9)
            found_min_dist = jnp.argwhere(condition, size=Nmax_nn, fill_value=-1).flatten()


            nearest_neigh = Scells[found_min_dist,j,:]
            neighbours_vecs = ref_atm - nearest_neigh 
            neighbours_dist = jnp.linalg.norm(neighbours_vecs, axis=1)
            # num_nn = len(neighbours_dist)
            neighbours_vecs = jnp.where(jnp.abs(neighbours_vecs) < 1e09, neighbours_vecs, 0.)
            
            
            
            # qq
            rij_nn = neighbours_dist #jnp.pad(neighbours_dist, (0, Nmax_nn-num_nn), constant_values=jnp.inf)
            rij_qq = rij_qq.at[i,j].set(rij_nn)
            rij_qq = rij_qq.at[j,i].set(rij_nn)
            
            
            # qd
            xyzij_nn = neighbours_vecs.T #np.pad(neighbours_vecs.T, ((0,0),(0, Nmax_nn-num_nn)), constant_values=0)                         
            xyzij_qd = xyzij_qd.at[i,j*3:(j+1)*3].set(xyzij_nn)
            xyzij_qd = xyzij_qd.at[j,i*3:(i+1)*3].set(- xyzij_nn)
            rij_qd = rij_qd.at[i,j*3:(j+1)*3].set(rij_nn)
            rij_qd = rij_qd.at[j,i*3:(i+1)*3].set(rij_nn)
            
            
            # dd
            xyzij_nn_3by3 = xyzij_nn[jnp.newaxis,:,:] * xyzij_nn[:,jnp.newaxis,:]
            xyzij_dd_outer = xyzij_dd_outer.at[3*i:3*(i+1), 3*j:3*(j+1)].set(xyzij_nn_3by3)
            xyzij_dd_outer = xyzij_dd_outer.at[3*j:3*(j+1), 3*i:3*(i+1)].set(xyzij_nn_3by3) 
            
            # xyzij_nn_unit = xyzij_nn / rij_nn
            # xyzij_nn_unit_3by3 = xyzij_nn_unit[np.newaxis,:,:] * xyzij_nn_unit[:,np.newaxis,:]
            # xyzij_dd_outer_unitvec[3*i:3*(i+1), 3*j:3*(j+1)] = xyzij_nn_unit_3by3  
            # xyzij_dd_outer_unitvec[3*j:3*(j+1), 3*i:3*(i+1)] = xyzij_nn_unit_3by3  
            
            
            xyzij_dd = xyzij_dd.at[:, 3*i:3*(i+1), 3*j:3*(j+1)].set(xyzij_nn[:, jnp.newaxis, jnp.newaxis, :])
            xyzij_dd = xyzij_dd.at[:, 3*j:3*(j+1), 3*i:3*(i+1)].set(-xyzij_nn[:, jnp.newaxis, jnp.newaxis, :])   
            
            
            for x in range(3):
                rij_dd_diag = rij_dd_diag.at[3*i+x, 3*j+x].set(rij_nn)
                rij_dd_diag = rij_dd_diag.at[3*j+x, 3*i+x].set(rij_nn)
                
                xyzij_dd_diag = xyzij_dd_diag.at[:, 3*i+x, 3*j+x].set(xyzij_nn)
                xyzij_dd_diag = xyzij_dd_diag.at[:, 3*j+x, 3*i+x].set(xyzij_nn)

            rij_dd_full = rij_dd_full.at[3*i:3*(i+1), 3*j:3*(j+1)].set(rij_nn)
            rij_dd_full = rij_dd_full.at[3*j:3*(j+1), 3*i:3*(i+1)].set(rij_nn)
            
            
            derx = jnp.array([[2*xyzij_nn[0], xyzij_nn[1], xyzij_nn[2]],
                    [xyzij_nn[1], jnp.zeros(Nmax_nn), jnp.zeros(Nmax_nn)],
                    [xyzij_nn[2],jnp.zeros(Nmax_nn), jnp.zeros(Nmax_nn)]])
            dery = jnp.array([[jnp.zeros(Nmax_nn), xyzij_nn[0], jnp.zeros(Nmax_nn)],
                    [xyzij_nn[0], 2*xyzij_nn[1], xyzij_nn[2]],
                    [jnp.zeros(Nmax_nn),xyzij_nn[2], jnp.zeros(Nmax_nn)]])
            derz = jnp.array([[jnp.zeros(Nmax_nn), jnp.zeros(Nmax_nn), xyzij_nn[0]],
                    [jnp.zeros(Nmax_nn), jnp.zeros(Nmax_nn), xyzij_nn[1]],
                    [xyzij_nn[0],xyzij_nn[1], 2*xyzij_nn[2]]])
            
            xyzij_dd_derivatives = xyzij_dd_derivatives.at[0,3*i:3*(i+1), 3*j:3*(j+1)].set(derx)
            xyzij_dd_derivatives = xyzij_dd_derivatives.at[1,3*i:3*(i+1), 3*j:3*(j+1)].set(dery)
            xyzij_dd_derivatives = xyzij_dd_derivatives.at[2,3*i:3*(i+1), 3*j:3*(j+1)].set(derz)
            xyzij_dd_derivatives = xyzij_dd_derivatives.at[0,3*j:3*(j+1), 3*i:3*(i+1)].set(-derx)
            xyzij_dd_derivatives = xyzij_dd_derivatives.at[1,3*j:3*(j+1), 3*i:3*(i+1)].set(-dery)
            xyzij_dd_derivatives = xyzij_dd_derivatives.at[2,3*j:3*(j+1), 3*i:3*(i+1)].set(-derz)
            

 
    # for i in range(3):
    #     diag = force_kernel[i].diagonal()      
    #     force_kernel[i] = force_kernel[i] - force_kernel[i].T
    #     np.fill_diagonal(force_kernel[i], diag)

    # check
    # print(np.allclose(rij, np.linalg.norm(xyzij, axis=2)))

    return rij_qq, xyzij_qd, rij_qd, xyzij_dd_outer, rij_dd_diag, rij_dd_full, xyzij_dd, xyzij_dd_diag, xyzij_dd_derivatives #xyzij_dd_outer_unitvec


# @partial(jax.jit, static_argnums=(2)) # @jax.jit
def get_rij_qq(R0, Scells, Nmax_nn, latvec, cutoff):
    N = len(R0)
    # qq
    rij_qq = jnp.zeros((N,N,Nmax_nn)) 
    
    # qd
    xyzij_qd = jnp.zeros((N,N*3,Nmax_nn)) 
    rij_qd = jnp.zeros((N, N*3, Nmax_nn))
    
    for i in range(N):
        ref_atm = R0[jnp.newaxis, i]
        for j in range(i+1):  
            dist_Scells_vec = Scells[:,j,:] - ref_atm
            dist_Scells = jnp.linalg.norm(dist_Scells_vec, axis=-1)
                        
            # the -0.1 is because we are working with float32
            condition = jnp.logical_and(dist_Scells < cutoff-0.1, dist_Scells > 1e-9)
            found_min_dist = jnp.argwhere(condition, size=Nmax_nn, fill_value=-1).flatten()


            nearest_neigh = Scells[found_min_dist,j,:]
            neighbours_vecs = ref_atm - nearest_neigh 
            neighbours_dist = jnp.linalg.norm(neighbours_vecs, axis=1)
            # num_nn = len(neighbours_dist)
            neighbours_vecs = jnp.where(jnp.abs(neighbours_vecs) < 1e09, neighbours_vecs, 0.)
            
            
            
            # qq
            rij_nn = neighbours_dist #jnp.pad(neighbours_dist, (0, Nmax_nn-num_nn), constant_values=jnp.inf)
            rij_qq = rij_qq.at[i,j].set(rij_nn)
            rij_qq = rij_qq.at[j,i].set(rij_nn)
            
            
            # qd
            xyzij_nn = neighbours_vecs.T #np.pad(neighbours_vecs.T, ((0,0),(0, Nmax_nn-num_nn)), constant_values=0)                         
            xyzij_qd = xyzij_qd.at[i,j*3:(j+1)*3].set(xyzij_nn)
            xyzij_qd = xyzij_qd.at[j,i*3:(i+1)*3].set(- xyzij_nn)
            rij_qd = rij_qd.at[i,j*3:(j+1)*3].set(rij_nn)
            rij_qd = rij_qd.at[j,i*3:(i+1)*3].set(rij_nn)
            
            
            
 
    # for i in range(3):
    #     diag = force_kernel[i].diagonal()      
    #     force_kernel[i] = force_kernel[i] - force_kernel[i].T
    #     np.fill_diagonal(force_kernel[i], diag)

    # check
    # print(np.allclose(rij, np.linalg.norm(xyzij, axis=2)))

    return rij_qq, xyzij_qd, rij_qd





# def get_Rij_old(R0, latvec, nrep=3):
#     """
#     returns the R matrix
#     """
#     N = len(R0)
#     Scells = get_Scells(latvec, R0, nrep=nrep)
    
    
#     Rij = np.zeros((N,N))
#     xyzij = np.zeros((3,N,N))
#     for i in range(N):
#         atom_origin = R0[i]
#         for j in range(i):
            
#             dist_Scells_vec = Scells[:,j,:] - atom_origin
#             dist_Scells = np.linalg.norm(dist_Scells_vec, axis=1)
#             found_min_dist = np.argwhere(dist_Scells == np.min(dist_Scells, axis=0))
#             found_min_dist_0 = found_min_dist[0,0]
#             nearest_neigh = Scells[found_min_dist_0,j,:]
            
         
            
#             Rij[i,j] = np.linalg.norm(nearest_neigh - atom_origin)
#             xyzij[:,i,j] = nearest_neigh - atom_origin
            
            
#     Rij = Rij + Rij.T
#     xyzij[0] = xyzij[0] - xyzij[0].T
#     xyzij[1] = xyzij[1] - xyzij[1].T
#     xyzij[2] = xyzij[2] - xyzij[2].T

#     return Rij, xyzij


# def get_el_kernels(R0, latvec, cutoff, alpha, error_func):
    
#     rij = get_rij(R0, latvec, cutoff)
    
   
#     erfc_over_rij  = error_func(alpha*rij)/rij
#     # erfc_over_rij3 = error_func(alpha*rij)/rij**3 
#     # gauss_term_ij = 2*alpha / np.sqrt(np.pi) * np.exp(-alpha**2*neighbours_dist**2) /neighbours_dist**2

#             # force_ij = -(erfc_over_rij3+gauss_term_ij) @ neighbours_vecs
#     energy_kernel = np.sum(erfc_over_rij, axis=2)
            
#             # force_kernel[:,i,j] = force_ij 
#             # energy_kernel[i,j] = e_ij

#     # for i in range(3):
#     #     diag = force_kernel[i].diagonal()      
#     #     force_kernel[i] = force_kernel[i] - force_kernel[i].T
#     #     np.fill_diagonal(force_kernel[i], diag)


#     # diag = energy_kernel.diagonal()
#     # energy_kernel = energy_kernel + energy_kernel.T
#     # np.fill_diagonal(energy_kernel, 0)

#     return energy_kernel, np.array([0]) #force_kernel

# def get_Scells_and_max_num_of_nn(R0, latvec, cutoff, multiplier=2):
#     N = len(R0)
#     Lmax = jnp.max(jnp.linalg.norm(latvec, axis=1)) #lattice vectors must be read horizontally
#     max_rep_right = ( (Lmax + cutoff) // Lmax ).astype(int) # maximum amount of cell replicas on the right
#     max_rep = 2*max_rep_right+1
#     Scells = get_Scells(latvec, R0, nrep=max_rep)
#     # Nmax_nn = 700
    
#     # take NN of atom 0, and double this number
#     max_nn = 0
#     for j in range(N):
#         ref_atm = jnp.expand_dims( R0[0], 0)
#         dist_Scells_vec = Scells[:,j,:] - ref_atm
#         dist_Scells = jnp.linalg.norm(dist_Scells_vec, axis=-1)
#         condition = jnp.logical_and(dist_Scells < cutoff, dist_Scells > 1e-9)
#         found_min_dist = jnp.argwhere(condition).flatten()
        
#         if len(found_min_dist) > max_nn:
#             max_nn = len(found_min_dist)
    
#     Nmax_nn = int(max_nn* multiplier)      
    
#     # qq
#     rij_qq = jnp.zeros((N,N,Nmax_nn)) 
    
#     # qd
#     xyzij_qd = jnp.zeros((N,N*3,Nmax_nn)) 
#     rij_qd = jnp.zeros((N, N*3, Nmax_nn))
    
#     # dd
#     rij_dd_diag = jnp.ones((3*N, 3*N, Nmax_nn)) * jnp.inf
#     rij_dd = jnp.zeros((3*N, 3*N, Nmax_nn))
#     xyzij_dd = jnp.zeros((3*N, 3*N, Nmax_nn)) 
    
#     return Scells, Nmax_nn, rij_qq, xyzij_qd, rij_qd, rij_dd_diag, rij_dd, xyzij_dd

# def get_rij(R0, cutoff, inp):
#     Scells, Nmax_nn, rij_qq, xyzij_qd, rij_qd, rij_dd_diag, rij_dd, xyzij_dd = inp
#     N = len(R0)
#     Nc = len(Scells)

#     for i in range(N):
#         ref_atm = jnp.expand_dims( R0[i], 0)
#         for j in range(i+1):  
            
#             dist_Scells_vec = Scells[:,j,:] - ref_atm
#             dist_Scells = jnp.linalg.norm(dist_Scells_vec, axis=-1)
 
            
#             condition = jnp.logical_and(dist_Scells < cutoff, dist_Scells > 1e-9)
#             found_min_dist = jnp.argwhere(condition).flatten()#, size=Nmax_nn, fill_value=j).flatten()
            

            
#             nearest_neigh = Scells[found_min_dist,j,:]
#             neighbours_vecs = ref_atm - nearest_neigh 
#             neighbours_dist = jnp.linalg.norm(neighbours_vecs, axis=1)
#             num_nn = len(neighbours_dist)
            
            

#             # qq
#             rij_nn = jnp.pad(neighbours_dist, (0, Nmax_nn-num_nn), constant_values=jnp.inf)
#             rij_qq = rij_qq.at[i,j].set(rij_nn)
#             rij_qq = rij_qq.at[j,i].set(rij_nn)
            
#             # qd
#             xyzij_nn = jnp.pad(neighbours_vecs.T, ((0,0),(0, Nmax_nn-num_nn)), constant_values=0)             
#             xyzij_qd = xyzij_qd.at[i,j*3:(j+1)*3].set(xyzij_nn)
#             xyzij_qd = xyzij_qd.at[j,i*3:(i+1)*3].set(- xyzij_nn)
#             rij_qd = rij_qd.at[i,j*3:(j+1)*3].set(rij_nn)
#             rij_qd = rij_qd.at[j,i*3:(i+1)*3].set(rij_nn)
            
#             # dd
#             xyzij_nn_3by3 = xyzij_nn[jnp.newaxis,:,:] * xyzij_nn[:,jnp.newaxis,:]
#             xyzij_dd = xyzij_dd.at[3*i:3*(i+1), 3*j:3*(j+1)].set(xyzij_nn_3by3)  
#             xyzij_dd = xyzij_dd.at[3*j:3*(j+1), 3*i:3*(i+1)].set(xyzij_nn_3by3)  
#             for x in range(3):
#                 rij_dd_diag = rij_dd_diag.at[3*i+x, 3*j+x].set(rij_nn)
#                 rij_dd_diag = rij_dd_diag.at[3*j+x, 3*i+x].set(rij_nn)
#             rij_dd = rij_dd.at[3*i:3*(i+1), 3*j:3*(j+1)].set(rij_nn)
#             rij_dd = rij_dd.at[3*j:3*(j+1), 3*i:3*(i+1)].set(rij_nn)
 

           
            
#     # for i in range(3):
#     #     diag = force_kernel[i].diagonal()      
#     #     force_kernel[i] = force_kernel[i] - force_kernel[i].T
#     #     jnp.fill_diagonal(force_kernel[i], diag)

#     # check
#     # print(jnp.allclose(rij, jnp.linalg.norm(xyzij, axis=2)))

#     return rij_qq, xyzij_qd, rij_qd, xyzij_dd, rij_dd_diag, rij_dd



# def get_Scells(H, R0, nrep=3):
#     N = len(R0)
#     xl = nrep // 2 
#     x1 = jnp.arange(-xl,-xl+nrep, dtype=int)
#     comb = jnp.array(jnp.meshgrid(x1,x1,x1)).T.reshape(-1,3)
#     displ = jnp.dot(H, comb.T).T
    
#     Scells = jnp.zeros((nrep**3, N, 3))
#     for c in range(nrep**3):
#         Scells = Scells.at[c].set(R0 + displ[c])     
#     return Scells



def old_get_Scells(latvec, R0, cutoff):
    Lmax = jnp.max(jnp.linalg.norm(latvec, axis=1)) #lattice vectors must be read horizontally
    max_rep_right = ( (Lmax + cutoff) // Lmax ).astype(int) # maximum amount of cell replicas on the right
    max_rep = 2*max_rep_right+1
    xl = max_rep_right
    x1 = jnp.arange(-xl,-xl+max_rep, dtype=int)
    comb = jnp.array(np.meshgrid(x1,x1,x1)).T.reshape(-1,3)
    comb = jnp.vstack((comb, jnp.array([1e18, 1e18, 1e18])))
    
    return comb

def old_get_Scells_and_max_num_of_nn(R0, comb, latvec, cutoff, multiplier=2):
    N = len(R0)
    
    displ = jnp.dot(latvec, comb.T).T
    # # Scells = jnp.zeros((len(comb), N, 3))
    # # for c in range(len(comb)):
    # #     print(displ.shape, R0.shape)
    # #     Scells.at[c].set(1) #R0 + displ.at[c]
    displ = displ[:,jnp.newaxis,:]
    Scells = displ + R0 #this contains, at [-1, :, :], a fake infinitely distant supercell
    # take NN of atom 0, and double this number
    max_nn = 0
    for j in range(N):
        ref_atm = R0[jnp.newaxis, 0]
        dist_Scells_vec = Scells[:,j,:] - ref_atm
        dist_Scells = jnp.linalg.norm(dist_Scells_vec, axis=-1)
        
        condition = jnp.logical_and(dist_Scells < cutoff, dist_Scells > 1e-9)
        found_min_dist = jnp.argwhere(condition).flatten()
        

        if len(found_min_dist) > max_nn:
            max_nn = len(found_min_dist)
            
    Nmax_nn = int(max_nn* multiplier)      
    return Scells, Nmax_nn

def estimate_max_Nimages(R0, comb, latvec, cutoff, multiplier=2):
    N = len(R0)
    displ = jnp.dot(latvec, comb.T).T
    displ = displ[:,jnp.newaxis,:]
    Scells = displ + R0
    
    rij = jnp.linalg.norm( R0[:, jnp.newaxis, :] - R0, axis=-1)
    rij = rij + jnp.diagflat(jnp.inf * jnp.ones(N))    

    i,j = jnp.argwhere(rij == rij.min())[0]
    ref_atm = R0[i]


    dist_Scells_vec = Scells[:,j,:] - ref_atm
    dist_Scells = jnp.linalg.norm(dist_Scells_vec, axis=-1)

    condition = jnp.logical_and(dist_Scells < cutoff, dist_Scells > 1e-9)
    found_min_dist = jnp.argwhere(condition).flatten()
    max_nn = len(found_min_dist)
            
    Nmax_nn = int(max_nn * multiplier)      
    return Scells, Nmax_nn
