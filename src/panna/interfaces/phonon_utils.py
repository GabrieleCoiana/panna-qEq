# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:16:32 2024

@author: Gabriele Coiana
"""
import tensorflow as tf
import numpy as np
import os

def read_gvector_params(file):
    with open(file, 'r') as f:
        data = f.readlines()


    units = data[1].split()[2]
    species = data[2].split()[2]
    
    # radial
    nrs_rad = int(data[5].split()[2])
    rc_rad = float(data[6].split()[2])
    rs0_rad = float(data[7].split()[2])
    eta_rad = float(data[8].split()[2])
    
    rsst_rad = (rc_rad - rs0_rad) / nrs_rad
    rs_rad = np.arange(rs0_rad, rc_rad, rsst_rad)
    eta_rad = np.tile(np.asarray(eta_rad), nrs_rad)
    
    #angular
    nthetas = int(data[11].split()[2])
    zeta = float(data[12].split()[2])
    
    step_theta = np.pi / nthetas
    thetas = np.arange(0, np.pi, step_theta) + .5 * step_theta
    zeta = np.tile(np.asarray(zeta), nthetas)

    # radial_angular components
    nrs_ang = int(data[15].split()[2])
    rc_ang = float(data[16].split()[2])
    rs0_ang = float(data[17].split()[2])
    eta_ang = float(data[18].split()[2])
    
    rsst_ang = (rc_ang - rs0_ang) / nrs_ang
    rs_ang = np.arange(rs0_ang, rc_ang, rsst_ang)
    eta_ang = np.tile(np.asarray(eta_ang), nrs_ang)
    
    return units,species,rc_rad,rs_rad,eta_rad,thetas,zeta,rc_ang,rs_ang,eta_ang


def create_mBP_descriptors(units,species,rc_rad,rs_rad,eta_rad,thetas,zeta,rc_ang,rs_ang,eta_ang, compute_dgvect=False, verbose=False):
    from panna.gvector import GvectmBP
    gvect_func = GvectmBP(compute_dgvect=compute_dgvect,species=species)
    nsp = len(species.split(','))

    gvect_func.units = units
    gvect_func.update_parameter('Rc_rad', rc_rad)
    gvect_func.update_parameter('Rs_rad', rs_rad)
    gvect_func.update_parameter('eta_rad', eta_rad)
    gvect_func.update_parameter('Thetas', thetas)
    gvect_func.update_parameter('zeta', zeta)
    gvect_func.update_parameter('Rc_ang', rc_ang)
    gvect_func.update_parameter('Rs_ang', rs_ang)
    gvect_func.update_parameter('eta_ang', eta_ang)
    
    if(verbose):
        print("Number of species is: {}".format(nsp))
        print("Size of the radial part is: {}".format(len(rs_rad) * nsp))
        print("Size of the angular part for each species permutation is: {}".format(len(thetas)*len(rs_ang)))
        print("There are {} 3-body species permutations hence the total size of the angularpart is: {} ".format((nsp*nsp+nsp)//2 , len(thetas)*len(rs_ang) *(nsp*nsp+nsp)//2 ))
        print("The total size of the G-vector with these parameters is: {}".format(gvect_func.gsize))

    return gvect_func

def load_example(path, species):
    from panna.lib import ExampleJsonWrapper
    # from panna.lib.example_bin import load_example
    example = ExampleJsonWrapper(path, species)
    # example = load_example(path)
    return example

def save_descriptors_bin(example, gvect_np, outdir):
    from panna import gvector
    import os
    try:
        os.makedirs(outdir)
    except FileExistsError:
        print('Folder already exists, saving into {}'.format(outdir))

    # TODO maybe you dont need to transform example in example_dict
    example_dict = {
        'key': example.key,
        'lattice_vectors': example.angstrom_lattice_vectors,
        'species': example.species_indexes,
        'positions': example.angstrom_positions,
        'E': example.ev_energy,
        'Gvect': gvect_np}

    gvector.binary_encoder(example_dict, binary_out=outdir)
    return


def write_tfrecord(files, outdir):
    import tensorflow as tf
    from panna.lib.tfr_data_structure import tfr_writer, example_tf_packer
    from panna.lib.example_bin import load_example
    import os
    
    # example = load_example(files)
    # packed = example_tf_packer(example)

    data = [example_tf_packer(load_example(x)) for x in files]
    
    # keys = [x.replace('.bin','') for x in files]
    os.makedirs(outdir)
    with tf.io.TFRecordWriter(outdir+'/gabri.tfrecord') as record_writer:
        for entry in data:
            record_writer.write(entry.SerializeToString())
    return


def retrieve_ds_multiple_files(folder, prefix):
    files = [folder+f for f in os.listdir(folder) if prefix in f]
    return files

def write_dataset_tfr(ds_bytes, n_files, outdir, outfiles):
    from tensorflow.data.experimental import TFRecordWriter
    
    try:
        os.makedirs(outdir)
    except FileExistsError:
        print('Folder already exists, saving into {}'.format(outdir))
        
    len_ds = len(ds_bytes)
    for i in range(n_files):
        print('Saving into {}th file'.format(i))
        ds = ds_bytes.take(len_ds // n_files)
        writer = TFRecordWriter(outdir+outfiles+'_'+str(i)+'.tfrecord')
        writer.write(ds)
    return




def make_example(example_dict):
    from tensorflow.io import serialize_tensor
    from tensorflow.train import BytesList, FloatList, Int64List
    from tensorflow.train import Example, Features, Feature
    
    key = example_dict['key']
    # lvec = example_dict['lattice_vectors']
    species = example_dict['species']
    # pos = example_dict['positions']
    energy = example_dict['E']
    gvect = example_dict['Gvect']
    dgvect = example_dict['dGvect']
    forces = example_dict['forces']


    key_feature = Feature(bytes_list=BytesList(value=[key.encode()]))
    # lvec_feature = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(lvec).numpy()]))
    species_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(species).numpy()]))
    # pos_feature = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(pos).numpy()]))
    energy_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(energy).numpy()]))
    gvect_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(gvect).numpy()]))
    # gvect_feature = Feature(float_list=FloatList(value=gvect.flatten()))
    gsize_feature = Feature(int64_list=Int64List(value=[gvect.size]))
    forces_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(forces).numpy()]))
    dgvect_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(dgvect).numpy()]))



    
    features = Features(feature={
        # 'key': key_feature,
        # 'lvec': lvec_feature,
        # 'species': species_feature,
        # 'pos': pos_feature,
        'energy': energy_feature,
        'gvect': gvect_feature,
        'gsize': gsize_feature,
        'forces': forces_feature,
        'dgvect': dgvect_feature
    })
    
    example = Example(features=features)
    
    return example.SerializeToString()


def parse_example(ex_tf, gsize, natoms):
    # parse tf example
    from tensorflow.io import FixedLenFeature, VarLenFeature, FixedLenSequenceFeature

    feature_description = {
        # 'key': FixedLenFeature([], tf.string),
        # 'lvec': FixedLenFeature([], tf.string),
        # 'species': FixedLenFeature([], tf.string),
        # 'pos': FixedLenFeature([], tf.string),
        'energy': FixedLenFeature([], tf.string),
        'gvect': FixedLenFeature([], tf.string),
        'gsize': FixedLenFeature([], tf.int64),
        'forces': FixedLenFeature([], tf.string),
        'dgvect': FixedLenFeature([], tf.string),
    }

    ex_tf_parsed = tf.io.parse_single_example(ex_tf, feature_description)
    
    # key = tf.cast(ex_tf_parsed['key'], tf.string) 
    # lvec = tf.io.parse_tensor(ex_tf_parsed['lvec'], out_type=tf.float64)
    # species = tf.io.parse_tensor(ex_tf_parsed['species'], out_type=tf.float64)
    # pos = tf.io.parse_tensor(ex_tf_parsed['pos'], out_type=tf.float64)
    energy = tf.io.parse_tensor(ex_tf_parsed['energy'], out_type=tf.float64)
    # gsize = tf.cast(ex_tf_parsed['gsize'], tf.int64)
    gvect = tf.io.parse_tensor(ex_tf_parsed['gvect'], out_type=tf.float64)
    # gvect = tf.cast(ex_tf_parsed['gvect'], tf.float64)
    forces = tf.io.parse_tensor(ex_tf_parsed['forces'], out_type=tf.float64)
    dgvect = tf.io.parse_tensor(ex_tf_parsed['dgvect'], out_type=tf.float64)



    
    return (tf.reshape(gvect, [gsize]), tf.reshape(dgvect, [natoms, gsize // natoms, natoms, 3]) ), ( tf.reshape(energy, [1]), tf.reshape(forces, [natoms*3]) )
    # return tf.reshape(gvect, [gsize]), ( tf.reshape(energy, [1]), tf.reshape(forces, [natoms*3]) )


def parse_dataset(raw_dataset, gsize, nat, batch_size, cache=True, ds_size=None):
    parsed_ds = raw_dataset.map(lambda ex: parse_example(ex, gsize, nat), num_parallel_calls=tf.data.AUTOTUNE)
    if ds_size is None:
        ds_size = sum(parsed_ds.map(lambda x,y: 1).as_numpy_iterator()) # the stupidest way to know size of dataset, see below
    # ds_size = len(list(parsed_ds)) # this is very inefficient but .cardinality() gives -2
    if cache:
        parsed_ds = parsed_ds.cache()
    parsed_ds = parsed_ds.shuffle(ds_size)
    parsed_ds = parsed_ds.batch(batch_size)
    parsed_ds = parsed_ds.prefetch(tf.data.AUTOTUNE)
    return parsed_ds, ds_size


   
    
   
    
def make_example_nodgvec(example_dict):
    from tensorflow.io import serialize_tensor
    from tensorflow.train import BytesList, FloatList, Int64List
    from tensorflow.train import Example, Features, Feature
    
    key = example_dict['key']
    # lvec = example_dict['lattice_vectors']
    species = example_dict['species']
    # pos = example_dict['positions']
    energy = example_dict['E']
    gvect = example_dict['Gvect']
    forces = example_dict['forces']



    key_feature = Feature(bytes_list=BytesList(value=[key.encode()]))
    # lvec_feature = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(lvec).numpy()]))
    species_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(species).numpy()]))
    # pos_feature = Feature(bytes_list=BytesList(value=[tf.io.serialize_tensor(pos).numpy()]))
    energy_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(energy).numpy()]))
    gvect_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(gvect).numpy()]))
    # gvect_feature = Feature(float_list=FloatList(value=gvect.flatten()))
    gsize_feature = Feature(int64_list=Int64List(value=[gvect.size]))
    forces_feature = Feature(bytes_list=BytesList(value=[serialize_tensor(forces).numpy()]))



    
    features = Features(feature={
        # 'key': key_feature,
        # 'lvec': lvec_feature,
        # 'species': species_feature,
        # 'pos': pos_feature,
        'energy': energy_feature,
        'gvect': gvect_feature,
        'gsize': gsize_feature,
        'forces': forces_feature
    })
    
    example = Example(features=features)
    
    return example.SerializeToString()


def parse_example_nodgvec(ex_tf, gsize, natoms):
    # parse tf example
    from tensorflow.io import FixedLenFeature, VarLenFeature, FixedLenSequenceFeature

    feature_description = {
        # 'key': FixedLenFeature([], tf.string),
        # 'lvec': FixedLenFeature([], tf.string),
        # 'species': FixedLenFeature([], tf.string),
        # 'pos': FixedLenFeature([], tf.string),
        'energy': FixedLenFeature([], tf.string),
        'gvect': FixedLenFeature([], tf.string),
        'gsize': FixedLenFeature([], tf.int64),
        'forces': FixedLenFeature([], tf.string)
    }

    ex_tf_parsed = tf.io.parse_single_example(ex_tf, feature_description)
    
    # key = tf.cast(ex_tf_parsed['key'], tf.string) 
    # lvec = tf.io.parse_tensor(ex_tf_parsed['lvec'], out_type=tf.float64)
    # species = tf.io.parse_tensor(ex_tf_parsed['species'], out_type=tf.float64)
    # pos = tf.io.parse_tensor(ex_tf_parsed['pos'], out_type=tf.float64)
    energy = tf.io.parse_tensor(ex_tf_parsed['energy'], out_type=tf.float64)
    # gsize = tf.cast(ex_tf_parsed['gsize'], tf.int64)
    gvect = tf.io.parse_tensor(ex_tf_parsed['gvect'], out_type=tf.float64)
    # gvect = tf.cast(ex_tf_parsed['gvect'], tf.float64)
    forces = tf.io.parse_tensor(ex_tf_parsed['forces'], out_type=tf.float64)




    
    return (tf.reshape(gvect, [gsize]) ) ,  (tf.reshape(energy, [1]), tf.reshape(forces, [natoms*3]))
    # return tf.reshape(gvect, [gsize]), ( tf.reshape(energy, [1]), tf.reshape(forces, [natoms*3]) )



def parse_dataset_nodgvec(raw_dataset, gsize, nat, batch_size, cache=True, ds_size=None):
    parsed_ds = raw_dataset.map(lambda ex: parse_example_nodgvec(ex, gsize, nat), num_parallel_calls=tf.data.AUTOTUNE)
    if ds_size is None:
        ds_size = sum(parsed_ds.map(lambda x,y: 1).as_numpy_iterator()) # the stupidest way to know size of dataset, see below
    # ds_size = len(list(parsed_ds)) # this is very inefficient but .cardinality() gives -2
    if cache:
        parsed_ds = parsed_ds.cache()
    parsed_ds = parsed_ds.shuffle(ds_size)
    parsed_ds = parsed_ds.batch(batch_size)
    parsed_ds = parsed_ds.prefetch(tf.data.AUTOTUNE)
    return parsed_ds, ds_size



def get_force_constants(sc, natuc, tol=1e-06, verbose=False, disp=0.001, dx_and_sx=False, save_forces_to_file=False, sortedd=False, jaxMD=False):
    R0 = sc.positions
    SCell = sc.cell.array
    natsc = len(R0)
    Nc = natsc // natuc
    forces = np.zeros((natuc*3, natsc*3))
    
    # the standard is that you repeat the unit cell by tiling: Mg O -> Mg O Mg O Mg O ...
    # in case atoms are ordered like Mg Mg Mg ... O O O instead of Mg O Mg O ...
    if sortedd:
        indices_of_ucatoms = np.arange(0, natsc, Nc)
    else:
        indices_of_ucatoms = np.arange(natuc)

    #initial forces should be 0
    if jaxMD == False:
        init_forces = sc.get_forces()
    else:
        force_fn, neighbors, neighbors_LR = jaxMD
        init_forces = np.array(force_fn(R0, SCell, neighbors, neighbors_LR))
    if verbose:
        print('Max initial force')
        print(np.max(init_forces, axis=0))
        print('Will displace atoms of [A] ', disp)
    assert abs(init_forces).max() <= tol

    for i, index in enumerate(indices_of_ucatoms):
        for x in range(3):
            if verbose:
                print('Displacing atom {} in direction {}'.format(i,x))
            # Rd = R0.copy() # this doesnt work
            
            # new_positions = R0.copy() # this doesnt work
            R0[index,x] += disp # displace to the right
            sc.set_positions(R0)
            if jaxMD == False: # compute forces with ASE
                forcedx = sc.get_forces()
            else: # compute forces with jaxMD
                force_fn, neighbors, neighbors_LR = jaxMD
                forcedx = np.array(force_fn(R0, SCell, neighbors, neighbors_LR))
            forces[i*3+x] = forcedx.flatten()
            R0[index,x] -= disp # remove displacement
            
            if dx_and_sx:
                if verbose:
                    print('Displacing atom {} in direction -{}'.format(i,x))
                # Rd = R0.copy() # this doesnt work

                R0[index,x] -= disp # displace to the left
                sc.set_positions(R0)
                if jaxMD == False: # compute forces with ASE
                    forcesx = sc.get_forces()
                else: # compute forces with jaxMD
                    force_fn, neighbors = jaxMD
                    forcesx = np.array(force_fn(R0, SCell, neighbors, neighbors_LR))
                R0[index,x] += disp # remove displacement
                
                # the formula is F = 1/2 * (Fdx - Fsx)
                forces[i*3+x] = (.5*(forcedx - forcesx)).flatten()
                
    K = -forces / disp  
    if save_forces_to_file:
        np.savetxt('K.dat', K)
    return K


def get_eigvals_eigvec(D):
    eigvals, eigvecs = np.linalg.eigh(D)
    idx = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx], eigvecs[:,idx]
    omegas = np.sqrt(eigvals, dtype=complex)/(2*np.pi)
    omegas = omegas.real - omegas.imag
    return eigvals, eigvecs, omegas

def get_eigvals(D):
    eigvals = np.linalg.eigvalsh(D)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    omegas = np.sqrt(eigvals, dtype=complex)/(2*np.pi)
    omegas = omegas.real - omegas.imag
    return eigvals, omegas

def get_D(k, K,N1,N2,N3,R0,masses,tot_atoms_uc, H):
    """
    New version of get_D, two times faster than get_D_old. Highly vectorised.
    
    builds the dynamical matrix
    expects forces in 1/amu * ev / A**2
    returns Thz**2
    """
    N1N2N3 = N1 * N2 * N3
    N = N1N2N3*tot_atoms_uc
    
    Scells, Rnew = R_matrix(N1,N2,N3,tot_atoms_uc, R0, H)
    #prepare kdotr
    exp_coeff = np.zeros((tot_atoms_uc*3,N*3), dtype=complex)
    for i in range(tot_atoms_uc):
        R = Rnew[:,i*3:i*3+3]
        kdotR = np.exp(1j*np.dot(R,k))
        exp_coeff[i*3,:] = kdotR
        exp_coeff[i*3+1,:] = kdotR
        exp_coeff[i*3+2,:] = kdotR
    
    mass_coeff = 1/np.sqrt(np.outer(masses, np.repeat(masses, N1N2N3)))
    Dall = mass_coeff*K*exp_coeff
    
    # Da = np.reshape(Dall, (tot_atoms_uc*3,tot_atoms_uc*3,N1N2N3))
    # D = np.sum(Da, axis=2)
    # Da = np.reshape(Dall, (N1N2N3, tot_atoms_uc*3, tot_atoms_uc*3))
    # D = np.sum(Da, axis=0)
    # D = np.zeros((tot_atoms_uc*3, tot_atoms_uc*3), dtype=complex)
    # for i in range(8):
    #     D += Dall[:,i*12:(i+1)*12]
    
    Da = np.reshape(Dall, (tot_atoms_uc*3,N,3))
    D = np.zeros((tot_atoms_uc*3, tot_atoms_uc*3), dtype=complex)
    for j in range(tot_atoms_uc):
        # print(j,N,tot_atoms_uc*3)
        D[:, j*3:j*3+3] = np.sum(Da[:,j:N:tot_atoms_uc,:], axis=1)
    
    # Da = np.reshape(Dall, (tot_atoms_uc*3,N,3))
    # D = np.zeros((tot_atoms_uc*3, tot_atoms_uc*3), dtype=complex)
    # for j in range(tot_atoms_uc*3):
    #     # print(j,N,tot_atoms_uc*3)
    #     D[:, j:j+1] = np.sum(Dall[:,j::tot_atoms_uc*3], axis=1).reshape((12,1))
        
    return (D+D.conj().transpose())/2*0.964*10**(4)


def get_Scells(H, R0):
    N = len(R0)
    x1 = np.arange(-1,2)
    comb = np.array(np.meshgrid(x1,x1,x1)).T.reshape(-1,3)
    displ = np.dot(H, comb.T).T
    Scells = np.zeros((27, N, 3))
    for c in range(27):
        Scells[c] = R0 + displ[c]     
    return Scells

def R_matrix(N1,N2,N3,tot_atoms_uc,R0, H):
    """
    returns the R matrix
    """
    N1N2N3 = N1 * N2 * N3
    N = len(R0[:,0])
    Scells = get_Scells(H, R0)
    
    
    R_matrix_old = np.zeros((tot_atoms_uc*3,N*3))
    R_matrix_new = np.zeros((N*3,tot_atoms_uc*3))
    count = 0
    for i in range(tot_atoms_uc):
        atom_origin = R0[i]
        row = []
        for j in range(N):
            dist_Scells_vec = Scells[:,j,:] - atom_origin
            dist_Scells = np.linalg.norm(dist_Scells_vec, axis=1)
            found_min_dist = np.argwhere(dist_Scells == np.min(dist_Scells, axis=0))
            found_min_dist_0 = found_min_dist[0]
            row.append((Scells[found_min_dist_0,j,:] - atom_origin).flatten())
        row = np.array(row, dtype=object)
        repeated_row = np.tile(row.flatten(),(3,1))
        R_matrix_old[count:count+3,:] = repeated_row
        # R_matrix_new[count_new] = 
        count =  count + 3
        R_matrix_new[:,i*3:i*3+3] = np.repeat(row.flatten().reshape(N,3),3,0)
        
    return Scells, R_matrix_new# *0.529177249

def R_matrix_multiplicity(N1,N2,N3,tot_atoms_uc,R0, H, tolerance=1.01, sortedd=False):
    """
    returns the R matrix with multiplicities!
    """
    N1N2N3 = N1 * N2 * N3
    N = len(R0[:,0])
    Scells = get_Scells(H, R0)
    # Scells2 = get_Scells2(H, S0)    
    
    R_matrix_new = []

    count = 0
    for i in range(tot_atoms_uc):
        if sortedd:
            atom_origin = R0[i*N1N2N3]
        else:
            atom_origin = R0[i]
        row = []
        for j in range(N):
            dist_Scells_vec = Scells[:,j,:] - atom_origin
            dist_Scells = np.linalg.norm(dist_Scells_vec, axis=1)
            min_dist = np.min(dist_Scells, axis=0)
            found_min_dist = np.argwhere( dist_Scells <= tolerance*min_dist)
            row.append(Scells[found_min_dist,j,:] - atom_origin)
        row = np.array(row, dtype=object)
        R_matrix_new.append(row)
    
    return R_matrix_new# *0.529177249

def get_D_multiplicity(k, K,N1,N2,N3,R0,masses,tot_atoms_uc, H, sortedd=False):
    """
    New version of get_D, with multiplicity. Eigenvectors should be fine now.
    Its a total delirium.
    
    builds the dynamical matrix
    expects forces in 1/amu * ev / A**2
    returns Thz**2
    """
    N1N2N3 = N1 * N2 * N3
    N = N1N2N3*tot_atoms_uc
    
    Rnew = R_matrix_multiplicity(N1,N2,N3,tot_atoms_uc, R0,  H, sortedd=sortedd)
    #prepare kdotr
    exp_coeff = np.zeros((tot_atoms_uc*3,N*3), dtype=complex)
    for i in range(tot_atoms_uc):
        R = Rnew[i]
        for j in range(N):
            R_this_atom = R[j][:,0,:]
            multiplicity = len(R_this_atom)
            exp_phases = np.exp(1j*np.dot(R_this_atom, k).astype(float))
            exp_phase = np.sum(exp_phases)/multiplicity
            # exp_phase = exp_phases[0]
            
            exp_coeff[i*3,j*3:j*3+3] = exp_phase
            exp_coeff[i*3+1,j*3:j*3+3] = exp_phase
            exp_coeff[i*3+2,j*3:j*3+3] = exp_phase
    
    # I need to know whether atoms are sorted this way: Mg Mg Mg, ... O, O, O ...
    if sortedd: # ase.build.sort(atoms)
        mass_coeff = 1/np.sqrt(np.outer(masses, np.repeat(masses, N1N2N3)))
        Dall = K*exp_coeff*mass_coeff
        Da = np.reshape(Dall, (tot_atoms_uc*3,N,3))
        D = np.zeros((tot_atoms_uc*3, tot_atoms_uc*3), dtype=complex)
        for j in range(tot_atoms_uc):
            D[:, j*3:j*3+3] = np.sum(Da[:,j*N1N2N3:(j+1)*N1N2N3,:], axis=1)
    
    else: # standard way
        # ... or this way: Mg O Mg O Mg O ...
        mass_coeff = 1/np.sqrt(np.outer(masses, np.tile(masses, N1N2N3)))
        Dall = mass_coeff*K*exp_coeff
        Da = np.reshape(Dall, (tot_atoms_uc*3,N,3))
        D = np.zeros((tot_atoms_uc*3, tot_atoms_uc*3), dtype=complex)
        for j in range(tot_atoms_uc):
            # print(j,N,tot_atoms_uc*3)
            D[:, j*3:j*3+3] = np.sum(Da[:,j:N:tot_atoms_uc,:], axis=1)
        
    return (D+D.conj().transpose())/2*0.964*10**(4) 




# def get_commensurate_kpath(sc, N1,N2,N3, N1_,N2_,N3_, kinput_scaled, labels):
    
#     # =============================================================================
#     #   Brillouin zone 
#     if type(sc) == str:
#         # then youre reading structure from file (VASP format)
#         conv = np.genfromtxt(sc, skip_header=1, max_rows=1, usecols=(0))
#         # IMPORTANT: you read a, b, c in the columns!
#         h = np.genfromtxt(sc,skip_header=2, max_rows=3, usecols=(0,1,2)).T * conv
#         # S = np.genfromtxt(sc, skip_header=8)
#     else:
#         conv = 1
#         h = sc.get_cell().T
        
    
#     Hk = 2*np.pi * np.linalg.inv(h)
    
    
#     kinput_real = kinput_scaled @ Hk
    
#     #Compute all possible k points
#     x1 = np.arange(0,N1_+1)/N1_
#     x2 = np.arange(0,N2_+1)/N2_
#     x3 = np.arange(0,N3_+1)/N3_
#     comb = np.array(np.meshgrid(x1,x2,x3)).T.reshape(-1,3)
    
#     allkpoints_scaled = comb
#     # allkpoints = np.multiply(b_mod,comb)
#     # =============================================================================
    
    
#     # =============================================================================
#     # Compute directions <x y z>
#     directions_scaled = []
#     directions = []
#     for i in range(0,len(kinput_scaled)-1):
#         if(kinput_scaled[i] not in allkpoints_scaled):
#             print('The k point '+str(kinput_scaled[i])+' provided is not compatible with the supercell. \nTry again :)')
#         num = 1/np.max(np.abs(kinput_scaled[i]-kinput_scaled[i+1]))
#         mod = np.linalg.norm(kinput_scaled[i]-kinput_scaled[i+1])
#         directions_scaled.append((kinput_scaled[i]-kinput_scaled[i+1])*num)
#         directions.append((kinput_scaled[i]-kinput_scaled[i+1])/mod)
#     # =============================================================================
    
#     # =============================================================================
#     # Remove kpoints already input
#     indexes = []
#     for i in range(len(allkpoints_scaled)):
#         for kpoint in kinput_scaled:
#             if(np.array_equal(allkpoints_scaled[i],kpoint)):
#                 indexes.append(i)
#     allkpoints_scaled_no = np.delete(allkpoints_scaled,indexes,axis=0)
#     # =============================================================================
    
#     # =============================================================================
#     # Look for other k points in the same path
#     kdef = kinput_real
#     kdef_scaled = kinput_scaled
#     x_labels = labels
#     count = 0
#     for index in range(0,len(directions)): #iterating over directions
#         count = count+1
#         direction = directions[index]
#         if (all(direction <= np.array([0,0,0]))):
#             for kpoint in allkpoints_scaled_no:
#                 diff = kinput_scaled[index]-kpoint
#                 mod = np.linalg.norm(diff)
#                 actual_direction = diff/mod
#                 if(np.array_equal(np.round(actual_direction,9),np.round(direction,9)) and all(np.abs(diff)<=np.abs(kinput_scaled[index+1]-kinput_scaled[index]))):
# #                    kdef = np.insert(kdef,count,np.dot(h,kpoint),axis=0)   
#                     kdef_scaled = np.insert(kdef_scaled,count,kpoint,axis=0)
#                     x_labels = np.insert(x_labels,count,' ')
#                     count = count+1 
#         else:
#             for kpoint in allkpoints_scaled_no[::-1]:
#                 diff = kinput_scaled[index]-kpoint
#                 mod = np.linalg.norm(diff)
#                 actual_direction = diff/mod
#                 if(np.array_equal(np.round(actual_direction,9),np.round(direction,9)) and all(np.abs(diff)<=np.abs(kinput_scaled[index+1]-kinput_scaled[index]))):
# #                    kdef = np.insert(kdef,count,np.dot(h,kpoint),axis=0)
#                     kdef_scaled = np.insert(kdef_scaled,count,kpoint,axis=0)
#                     x_labels = np.insert(x_labels,count,' ')
#                     count = count+1 

#     # =============================================================================
    
#     kdef_scaled = np.array(kdef_scaled)
#     kdef = kdef_scaled @ Hk#np.dot(Hk,kdef_scaled.T).T
#     Nqpoints = len(kdef)
    
#     # =============================================================================
#     #  fake k points only for plotting, projection
#     directionsdef = []
#     for i in range(0,len(kdef_scaled)-1):
#         directionsdef.append((kdef[i]-kdef[i+1]))
#     kk = np.zeros(len(kdef_scaled))
#     for i in range(1,len(kk)):
#         kk[i] = kk[i-1]+(np.linalg.norm(directionsdef[i-1]))
#     # =============================================================================
    
#     print('AAAAAAAAAA')
#     print(np.round(Hk, 3))
#     return Nqpoints, kdef, kdef_scaled, kk, x_labels, Hk


def get_commensurate_kpath(sc, N1,N2,N3, N1_,N2_,N3_, kinput_scaled, labels):
    # =============================================================================
    #   Brillouin zone 
    if type(sc) == str:
        # then youre reading structure from file (VASP format)
        conv = np.genfromtxt(sc, skip_header=1, max_rows=1, usecols=(0))
        h = np.genfromtxt(sc,skip_header=2, max_rows=3, usecols=(0,1,2)).T * conv
        # S = np.genfromtxt(sc, skip_header=8)
    else:
        conv = 1
        h = sc.get_cell().T
        
        
    
    # #your default units are Bohrs, but phonopy's are A
    # if (conv==1.0 or conv==1):
    #     conv_factor = 1.88973
    # else:
    #     conv_factor = 1
    conv_factor = 1
        
    a1,a2,a3 = h[:,0]*conv_factor/N1,h[:,1]*conv_factor/N2,h[:,2]*conv_factor/N3    
    c = 2*np.pi
    
    V = np.dot(a1,np.cross(a2,a3))
    b1 = c*np.cross(a2,a3)/V
    b2 = c*np.cross(a3,a1)/V
    b3 = c*np.cross(a1,a2)/V
    Hk = np.vstack((b1,b2,b3)).T
    
    b1_mod = np.linalg.norm(b1)
    b2_mod = np.linalg.norm(b2)
    b3_mod = np.linalg.norm(b3)
    b_mod = [b1_mod,b2_mod,b3_mod]
    kinput_real = np.multiply(b_mod,np.array(kinput_scaled))
    #Compute all possible k points
    x1 = np.arange(0,N1_+1)/N1_
    x2 = np.arange(0,N2_+1)/N2_
    x3 = np.arange(0,N3_+1)/N3_
    comb = np.array(np.meshgrid(x1,x2,x3)).T.reshape(-1,3)
    #a1 = np.multiply(comb,b1)
    #a2 = np.multiply(comb,b2)
    #a3 = np.multiply(comb,b3)
    #kpoints = a1+a2+a3
    allkpoints_scaled = comb
    # allkpoints = np.multiply(b_mod,comb)
    # =============================================================================
    
    
    # =============================================================================
    # Compute directions <x y z>
    directions_scaled = []
    directions = []
    for i in range(0,len(kinput_scaled)-1):
        if(kinput_scaled[i] not in allkpoints_scaled):
            print('The k point '+str(kinput_scaled[i])+' provided is not compatible with the supercell. \nTry again :)')
        num = 1/np.max(np.abs(kinput_scaled[i]-kinput_scaled[i+1]))
        mod = np.linalg.norm(kinput_scaled[i]-kinput_scaled[i+1])
        directions_scaled.append((kinput_scaled[i]-kinput_scaled[i+1])*num)
        directions.append((kinput_scaled[i]-kinput_scaled[i+1])/mod)
    # =============================================================================
    
    # =============================================================================
    # Remove kpoints already input
    indexes = []
    for i in range(len(allkpoints_scaled)):
        for kpoint in kinput_scaled:
            if(np.array_equal(allkpoints_scaled[i],kpoint)):
                indexes.append(i)
    allkpoints_scaled_no = np.delete(allkpoints_scaled,indexes,axis=0)
    # =============================================================================
    
    # =============================================================================
    # Look for other k points in the same path
    kdef = kinput_real
    kdef_scaled = kinput_scaled
    x_labels = labels
    count = 0
    for index in range(0,len(directions)): #iterating over directions
        count = count+1
        direction = directions[index]
        if (all(direction <= np.array([0,0,0]))):
            for kpoint in allkpoints_scaled_no:
                diff = kinput_scaled[index]-kpoint
                mod = np.linalg.norm(diff)
                actual_direction = diff/mod
                if(np.array_equal(np.round(actual_direction,9),np.round(direction,9)) and all(np.abs(diff)<=np.abs(kinput_scaled[index+1]-kinput_scaled[index]))):
#                    kdef = np.insert(kdef,count,np.dot(h,kpoint),axis=0)   
                    kdef_scaled = np.insert(kdef_scaled,count,kpoint,axis=0)
                    x_labels = np.insert(x_labels,count,' ')
                    count = count+1 
        else:
            for kpoint in allkpoints_scaled_no[::-1]:
                diff = kinput_scaled[index]-kpoint
                mod = np.linalg.norm(diff)
                actual_direction = diff/mod
                if(np.array_equal(np.round(actual_direction,9),np.round(direction,9)) and all(np.abs(diff)<=np.abs(kinput_scaled[index+1]-kinput_scaled[index]))):
#                    kdef = np.insert(kdef,count,np.dot(h,kpoint),axis=0)
                    kdef_scaled = np.insert(kdef_scaled,count,kpoint,axis=0)
                    x_labels = np.insert(x_labels,count,' ')
                    count = count+1 

    # =============================================================================
    
    kdef_scaled = np.array(kdef_scaled)
    kdef = np.dot(Hk,kdef_scaled.T).T
    Nqpoints = len(kdef)
    
    # =============================================================================
    #  fake k points only for plotting, projection
    directionsdef = []
    for i in range(0,len(kdef_scaled)-1):
        directionsdef.append((kdef[i]-kdef[i+1]))
    kk = np.zeros(len(kdef_scaled))
    for i in range(1,len(kk)):
        kk[i] = kk[i-1]+(np.linalg.norm(directionsdef[i-1]))
    # =============================================================================
    
    return Nqpoints, kdef, kdef_scaled, kk, x_labels, Hk

from math import gcd
from functools import reduce

def get_commensurate_kpath_v2(sc, N1, N2, N3, N1_, N2_, N3_, kinput_scaled, labels, grid_tol=1e-3):
    # --- Read cell ---
    if isinstance(sc, str):
        conv = np.genfromtxt(sc, skip_header=1, max_rows=1, usecols=(0))
        h = np.genfromtxt(sc, skip_header=2, max_rows=3, usecols=(0, 1, 2)).T * conv
    else:
        h = sc.get_cell()

    # --- Reciprocal lattice vectors of the supercell (columns of Hk) ---
    a1, a2, a3 = h[:, 0] / N1, h[:, 1] / N2, h[:, 2] / N3
    c = 2 * np.pi
    V = np.dot(a1, np.cross(a2, a3))
    b1 = c * np.cross(a2, a3) / V
    b2 = c * np.cross(a3, a1) / V
    b3 = c * np.cross(a1, a2) / V
    Hk = np.vstack((b1, b2, b3)).T   # k_cart = Hk @ k_scaled

    kinput_scaled = np.asarray(kinput_scaled, dtype=float)
    Ns = np.array([N1_, N2_, N3_])

    def on_grid(kpt):
        vals = kpt * Ns
        return np.all(np.abs(vals - np.round(vals)) < grid_tol * Ns)

    kdef_scaled = []
    x_labels_out = []
    kk_out = []
    cumulative_length = 0.0

    # Always include the first waypoint, snapped to nearest grid point if close
    first = kinput_scaled[0]
    if on_grid(first):
        first = np.round(first * Ns) / Ns  # snap to exact grid value
    kdef_scaled.append(first.copy())
    x_labels_out.append(labels[0])
    kk_out.append(0.0)

    prev_cart = Hk @ kinput_scaled[0]

    n_waypoints = len(kinput_scaled)

    for i in range(n_waypoints - 1):
        k_start = kinput_scaled[i]
        k_end = kinput_scaled[i + 1]
        diff = k_end - k_start

        steps_candidates = []
        for comp, N in zip(diff, Ns):
            if abs(comp) > 1e-12:
                steps_candidates.append(int(round(abs(comp) * N)))

        if not steps_candidates:
            continue

        n_steps = reduce(lambda a, b: a * b // gcd(a, b), steps_candidates)
        n_steps = max(1, n_steps)

        is_last_segment = (i == n_waypoints - 2)

        for n in range(1, n_steps + 1):
            t = n / n_steps
            kpt = k_start + t * diff
            kpt_cart = Hk @ kpt
            cumulative_length += np.linalg.norm(kpt_cart - prev_cart)
            prev_cart = kpt_cart

            is_segment_end = (n == n_steps)
            force_keep = is_segment_end and is_last_segment  # always keep final waypoint

            if on_grid(kpt) or force_keep:
                kpt_out = kpt.copy()
                if on_grid(kpt):
                    kpt_out = np.round(kpt * Ns) / Ns  # snap to exact grid value
                kdef_scaled.append(kpt_out)
                kk_out.append(cumulative_length)
                x_labels_out.append(labels[i + 1] if is_segment_end else ' ')

    kdef_scaled = np.array(kdef_scaled)
    kdef = (Hk @ kdef_scaled.T).T
    kk = np.array(kk_out)
    x_labels_out = np.array(x_labels_out)
    Nqpoints = len(kdef)

    return Nqpoints, kdef, kdef_scaled, kk, x_labels_out, Hk

def get_commensurate_kpath_claude(sc, N1, N2, N3, N1_, N2_, N3_, kinput_scaled, labels):
    # --- Read cell ---
    if type(sc) == str:
        conv = np.genfromtxt(sc, skip_header=1, max_rows=1, usecols=(0))
        h = np.genfromtxt(sc, skip_header=2, max_rows=3, usecols=(0, 1, 2)).T * conv
    else:
        h = sc.get_cell()

    # --- Reciprocal lattice vectors of the supercell ---
    a1, a2, a3 = h[:, 0] / N1, h[:, 1] / N2, h[:, 2] / N3
    c = 2 * np.pi
    V = np.dot(a1, np.cross(a2, a3))
    b1 = c * np.cross(a2, a3) / V
    b2 = c * np.cross(a3, a1) / V
    b3 = c * np.cross(a1, a2) / V
    Hk = np.vstack((b1, b2, b3)).T

    # --- All commensurate k-points in the supercell BZ ---
    x1, x2, x3 = np.arange(0, N1_ + 1) / N1_, np.arange(0, N2_ + 1) / N2_, np.arange(0, N3_ + 1) / N3_
    allkpoints_scaled = np.array(np.meshgrid(x1, x2, x3)).T.reshape(-1, 3)

    def is_commensurate(kpt):
        return any(np.allclose(kpt, ak) for ak in allkpoints_scaled)

    def find_intermediates(k_start, k_end):
        """Find commensurate k-points strictly between k_start and k_end along the segment."""
        direction = k_end - k_start
        intermediates = []
        for kpt in allkpoints_scaled:
            diff = kpt - k_start
            # Must be collinear with segment
            if not np.allclose(np.cross(diff, direction), 0):
                continue
            # Compute parametric t; must be strictly in (0, 1)
            nz = np.abs(direction) > 1e-10
            if not np.any(nz):
                continue
            t_vals = diff[nz] / direction[nz]
            t = t_vals[0]
            if np.allclose(t_vals, t) and 1e-10 < t < 1 - 1e-10:
                intermediates.append((t, kpt.copy()))
        intermediates.sort(key=lambda x: x[0])
        return [kpt for _, kpt in intermediates]

    # --- Build full path, segment by segment through all kinput waypoints ---
    # kdef_scaled: all points on path (commensurate only, but path shaped by incommensurate waypoints)
    # kk: cumulative path length including incommensurate segments
    kdef_scaled = []   # commensurate k-points to return
    x_labels_out = []  # labels aligned to kdef_scaled
    kk_out = []        # cumulative path coordinate for each commensurate point

    cumulative_length = 0.0
    prev_kpt_cart = np.dot(Hk, kinput_scaled[0])  # Cartesian of last processed point

    for i in range(len(kinput_scaled) - 1):
        k_start, k_end = kinput_scaled[i], kinput_scaled[i + 1]
        k_start_cart = np.dot(Hk, k_start)
        k_end_cart   = np.dot(Hk, k_end)

        # Emit k_start if commensurate (skip on i>0 to avoid duplicates)
        if i == 0:
            if is_commensurate(k_start):
                kdef_scaled.append(k_start.copy())
                x_labels_out.append(labels[i])
                kk_out.append(cumulative_length)
            else:
                print(f'Warning: segment start {k_start} is incommensurate — used for path geometry only.')

        # Advance cumulative length to k_start (already done for i==0, needed after incommensurate waypoints)
        # (prev_kpt_cart tracks where we physically are on the path)

        # Find commensurate intermediates strictly inside this segment
        intermediates = find_intermediates(k_start, k_end)
        for kpt in intermediates:
            kpt_cart = np.dot(Hk, kpt)
            cumulative_length += np.linalg.norm(kpt_cart - prev_kpt_cart)
            prev_kpt_cart = kpt_cart
            kdef_scaled.append(kpt.copy())
            x_labels_out.append(' ')
            kk_out.append(cumulative_length)

        # Advance cumulative length to k_end (whether commensurate or not)
        cumulative_length += np.linalg.norm(k_end_cart - prev_kpt_cart)
        prev_kpt_cart = k_end_cart

        # Emit k_end only if commensurate
        if is_commensurate(k_end):
            kdef_scaled.append(k_end.copy())
            x_labels_out.append(labels[i + 1])
            kk_out.append(cumulative_length)
        else:
            print(f'Warning: k-point {k_end} is incommensurate — used for path geometry only.')

    kdef_scaled = np.array(kdef_scaled)
    kdef = np.dot(Hk, kdef_scaled.T).T
    kk = np.array(kk_out)
    x_labels_out = np.array(x_labels_out)
    Nqpoints = len(kdef)

    return Nqpoints, kdef, kdef_scaled, kk, x_labels_out, Hk


def get_kpath_many(kinput, Hk, npoints=20):
    ks_scal = np.array([0,0,0])
    distances, dist = [], 0
    for i in range(len(kinput)-1):
        this_k = kinput[i]
        next_k = kinput[i+1]
        diff = next_k - this_k
        increment = diff/npoints
        kpoints_in_between = np.outer(increment, np.arange(npoints+1)).T + this_k
        distance = np.linalg.norm(np.dot(Hk,(kpoints_in_between-this_k).T).T, axis=1) + dist
        dist = distance[-1]
        distances.append(distance[0:-1])
        ks_scal = np.vstack((ks_scal,kpoints_in_between[:-1,:]))
        
    ks_scal = np.vstack((ks_scal, kinput[-1]))
    ks_scal = ks_scal[1::,:]
    distances = np.array(distances).flatten()
    distances = np.concatenate((distances, [distance[-1]]))

    ks = np.dot(Hk, ks_scal.T).T # ks_scal @ Hk #
    return ks_scal, ks, distances

def get_path(sc, N1, N2, N3, kinput_scaled, labels, interp=0, mode='interp'):
    if mode == 'interp':
        Nqs_path, kps, kps_sc, xcom, xlabels, Hk = get_commensurate_kpath(sc, N1, N2, N3, N1, N2, N3, kinput_scaled, labels)
        kps_sc, kps, x = get_kpath_many(kps_sc, Hk, npoints=interp+1)
    else:
        Nqs_path, kps, kps_sc, xcom, xlabels, Hk = get_commensurate_kpath_claude(sc, N1, N2, N3, N1, N2, N3, kinput_scaled, labels)
        x = xcom
    # path = sc.cell.bandpath('GXWGL', npoints=5) # #GKMG # , special_points={'Q': [0.5,0.5,1]}
    # kps_sc = path.kpts
    # kps = path.cartesian_kpts()
    # Hk = np.array(sc.cell.reciprocal()) * 2*np.pi * n
    # kps = (Hk @ kps_sc.T).T
    return kps_sc, kps, x, xcom, xlabels

def get_phonons_increasing_scells(uc, 
                                  configfile, 
                                  scells, 
                                  kinput_scaled, 
                                  labels, 
                                  frelax=False, 
                                  asr=False, 
                                  tol=1e-06, 
                                  disp=0.001, 
                                  dx_and_sx=False, 
                                  verbose=False, 
                                  save_forces_to_file=False, 
                                  save_freqs_eigvecs=False, 
                                  appendix='', 
                                  connect_bands=False,
                                  jaxMDpotfile=False):
    from ase.build.supercells import make_supercell
    from ase.optimize import BFGS
    from panna.interfaces.jax_ASE import PANNAJAXCalculator_new
    # from panna.interfaces.old_jax_ASE import PANNAJAXCalculatorLR
    natuc = uc.get_global_number_of_atoms()
    xs_list = []
    freqs_list = []
    evecs_list = []
    print('Phonon calculation with supercells of increasing dimension')
    for i, ii in enumerate(scells):
        # re-instantiate pcalc: this resets the NN list creation
        pcalc = PANNAJAXCalculator_new(config=configfile, verbose=verbose)
        # pcalc = PANNAJAXCalculatorLR(config=configfile, verbose=verbose)
        print(f'---- Creating supercell {ii} x {ii} x {ii} ----')
        # creating supercell
        N1, N2, N3 = ii,ii,ii
        sc = make_supercell(uc, [[N1, 0, 0], [0, N2, 0], [0, 0, N3]])
        SCell = sc.cell.array
        R0 = sc.get_positions()
        species = sc.get_chemical_symbols()
        sc.calc = pcalc
        # sc = sort(sc)
        masses = uc.get_masses()
        
        if jaxMDpotfile == False:
            jaxMD = False
        else:
            from panna.interfaces.jaxMD import PANNAJAXMdCalculator
            import jax_md
            import jax
            displacement_fn, shift_fn = jax_md.space.periodic_general(SCell, fractional_coordinates=False)

            pcalcj = PANNAJAXMdCalculator(jaxMDpotfile, SCell, species, rcut_LR=10, ewald_acc_factor=1e-07)
            neighbor_fn, neighbor_fn_LR, energy_fn, force_fn = pcalcj.initialize()
            # First neigh allocation and call for jit compilation
            neighbors = neighbor_fn.allocate(R0)
            if neighbor_fn_LR != 0:
                neighbors_LR = neighbor_fn_LR.allocate(R0)
            else:
                neighbors_LR = 0

            jaxMD = (force_fn, neighbors, neighbors_LR)
            
        if frelax:
            if jaxMDpotfile == False:
                print('Relaxing supercell')
                relax = BFGS(sc)
                relax.run(fmax=1e-09, steps=50)
                # Erelax = sc.get_potential_energy() #* 1/Ha_to_ev
                Frelax = sc.get_forces() #/ CONV_FACT_f 
                # R0relax = sc.get_positions()
                # Qrelax = sc.get_charges()
                # Drelax = sc.get_dipole_moment() #* Ang_to_bohr
                print('Max force component', abs(Frelax).max(), 'ev/A')
            else:
                step = 0.002
                Ftol = 1e-08
                # force_fn, neighbors, shift_fn = jaxMD
                init_fn, apply_fn = jax_md.minimize.fire_descent(force_fn, shift_fn, step)
                apply_fn = jax.jit(apply_fn)
                state = init_fn(R0, box=SCell, neighbors=neighbors)
                
                for i in range(2000):
                    print(i)
                    neighbors = neighbors.update(state.position)
                    state = apply_fn(state, box=SCell, neighbors=neighbors)
                    
                    # Ei = float(energy_fn(state.position, SCell, neighbors))
                    Fi = np.array(force_fn(state.position, SCell, neighbors))
                    Fmaxi = np.max(abs(Fi))
                    
                    if Fmaxi < Ftol:
                        print('Ftol reached, ending relaxation.')
                        break
                print('Max force component', Fmaxi, 'ev/A')
                R = np.array(state.position)
                
                # update ase positions
                sc.set_positions(state.position)#state.tolist())
                
                jaxMD = (force_fn, neighbors)
        
        
        inputs_phonons = (natuc, N1, N2, N3, kinput_scaled, labels, 0, masses)
        x, xcom, xlabels, kps_sc, kps, frequencies_list, eigvecs_list, Ds_list, K = get_phonons(sc, inputs_phonons, asr=asr, dx_and_sx=dx_and_sx, save_freqs_eigvecs=save_freqs_eigvecs, appendix=appendix, tol=tol, verbose=verbose, connect_bands=connect_bands, mode='comm', jaxMD=jaxMD)
        #frequencies_list = np.clip(frequencies_list, 0, None ) # aaaa
        
        xs_list.append(x)
        freqs_list.append(frequencies_list)
        evecs_list.append(eigvecs_list)
        
        print(f'---- End supercell {ii} x {ii} x {ii} ----')
    
    # From Claude
    # Collect all (x, freq_row) pairs into a dict keyed by x value
    # Later supercells overwrite earlier ones for duplicate x values (last wins)
    # # Build unified x and frequencies across all supercells
    # # xs_list and freqs_list have len = number of supercells
    # # freqs_list[i] has shape (Nkps_i, ndofs)
    unified = {}
    unified_evecs = {}
    
    for xs, freqs, evecs in zip(xs_list, freqs_list, evecs_list):
        for x_val, freq_row, evec_mat in zip(xs, freqs.T, evecs):
            key = round(float(x_val), 10)
            unified[key] = np.array(freq_row).flatten()
            unified_evecs[key] = np.array(evec_mat)  # shape (Ndofs, Ndofs)
    
    x_sorted = sorted(unified.keys())
    x_unified = np.array(x_sorted)
    freqs_unified = np.array([unified[x] for x in x_sorted])        # shape (Nkps_total, Ndofs)
    evecs_unified = np.array([unified_evecs[x] for x in x_sorted])  # shape (Nkps_total, Ndofs, Ndofs)
    
    if save_freqs_eigvecs:
        save_freqs_and_eigvecs(x_unified, xcom, xlabels, freqs_unified.T, evecs_unified, appendix)
    
    
    return x_unified, freqs_unified.T, evecs_unified, xcom, xlabels#xs_list, freqs_list, evecs_list, xcom, xlabels # 


def get_phonons(sc, inputs_phonons, 
                asr=False, 
                tol=1e-06, 
                disp=0.001, 
                dx_and_sx=False, 
                verbose=False, 
                save_forces_to_file=False, 
                save_freqs_eigvecs=False, 
                appendix='', 
                connect_bands=False, 
                mode='interp',
                sortedd=False,
                jaxMD=False):
    
    natuc, N1, N2, N3, kinput_scaled, labels, interp, masses = inputs_phonons
    
    # # # the standard is that you repeat the unit cell by tiling: Mg O -> Mg O Mg O Mg O ...
    # # # in case atoms are ordered like Mg Mg Mg ... O O O instead of Mg O Mg O ...
    # # _, indices_of_ucatoms = np.unique(sc.get_chemical_symbols(), return_index=True) 
    # # sortedd = indices_of_ucatoms.max() > natuc
    # sortedd=sortedd
    # if verbose:
    #     print('Atoms are sorted ', sc.get_chemical_symbols())
        
    # getting force constants
    K = get_force_constants(sc, natuc, tol=tol, disp=disp, dx_and_sx=dx_and_sx, verbose=verbose, save_forces_to_file=False, sortedd=sortedd, jaxMD=jaxMD)
    
    if asr:
        meanKx = np.mean(K[:,0::3], axis=1)
        meanKy = np.mean(K[:,1::3], axis=1)
        meanKz = np.mean(K[:,2::3], axis=1)
    
        K[:,0::3] -= meanKx[:,None]
        K[:,1::3] -= meanKy[:,None]
        K[:,2::3] -= meanKz[:,None]
        
    kps_sc, kps, x, xcom, xlabels = get_path(sc, N1, N2, N3, kinput_scaled, labels, interp=0+interp, mode=mode)
        
    R0 = sc.get_positions()
    SCell = sc.get_cell().T
    masses_uc = np.repeat(masses, 3)
    
    frequencies_list = np.zeros((natuc*3,len(kps)))
    eigvecs_list = np.zeros((len(kps), natuc*3, natuc*3), dtype=complex)
    Ds_list = np.zeros((len(kps), natuc*3, natuc*3), dtype=complex)
    for i in range(len(kps)):
        k = kps[i]
        print('\t processing k-point:', i, np.round(kps_sc[i], 3))

        # 0K dispersion
        D = get_D_multiplicity(k, K,N1,N2,N3,R0,masses_uc, natuc, SCell, sortedd=sortedd)
        eigvals, eigvecs, omegas = get_eigvals_eigvec(D)
        eigvecs_list[i] = eigvecs
        Ds_list[i] = D
        frequencies_list[:,i] = omegas 
        
    
    if save_freqs_eigvecs:
        save_freqs_and_eigvecs(x, xcom, xlabels, frequencies_list, eigvecs_list, appendix)
    
    if connect_bands:
        indexes_freqs, frequencies_list, eigvecs_list = get_real_branches(frequencies_list, eigvecs_list, kps)#, new_indexes=indici[5])

    
    return x, xcom, xlabels, kps_sc, kps, frequencies_list, eigvecs_list, Ds_list, K


def save_freqs_and_eigvecs(x, kk, labels, frequencies_list, eigvecs_list, appendix):
    print('Saving frequencies and eigenvectors...')
    ndofs = len(frequencies_list[:,0])
    with open('frequencies'+appendix, 'w') as f:
        f.write('# Phonon frequencies. (Nkps x natuc*3+1) matrix. The first column is x, the other columns are the frequencies.\n')
        f.write('# commensurate path: {} \n'.format(','.join(list(map(str, np.around(kk,3).tolist())))))
        f.write('# labels: {} \n'.format(','.join(list(labels))))
    with open('frequencies'+appendix, 'ab') as f:
        np.savetxt(f, np.hstack((np.expand_dims(x,1),frequencies_list.T)))
        
    with open('eigvecs'+appendix, 'w') as f:
        f.write('# Phonon eigenvectors along the path given by the user in phonon_calc.py.It is a (Nkpoints*natuc*3 x natuc*3*2) matrix. Real and imaginary parts are in alternating columns.\n\
                # Example of loading with Python:\n\
                #   import numpy as np\n\
                #   matrix = np.loadtxt(`band_eigvecs`, comments=[`#`])\n\
                #   eigvecs = matrix[:,0::2] + 1j* matrix[:,1::2]\n\
                #   eigvecs = eigvecs.reshape((Nkps,natuc*3,natuc*3))\n')

    with open('eigvecs'+appendix, 'ab') as f:
        np.savetxt(f, eigvecs_list.reshape((-1,ndofs)).view(float))
    return





def get_real_branches(freqs, eigvecs, ks, new_indexes=None):
    """
    This code finds the phonon bands based on the eigenvectors. 
    It is equivalent to Phonopy's band connection = .true.
    It is quite hard to read, to be honest I think I don't know myself!
    """
    num_ks = len(ks)
    num_branches = len(freqs[:,0])    
    
    if new_indexes is None:    
        new_indexes = range(num_branches) #you have to start from something, you assume the first kpoint is in order
    all_indexes = [new_indexes]
    
    for k in range(0,num_ks-1):
        kpoint = ks[k]
        next_kpoint = ks[k+1]

        this_eigvecs_T = eigvecs[k].T
        norm_this_eigvecs = np.linalg.norm(this_eigvecs_T,axis=1)
        this_eigvecs_norm = this_eigvecs_T.T /norm_this_eigvecs
        
        next_eigvecs_T = eigvecs[k+1].T
        norm_next_eigvecs = np.linalg.norm(next_eigvecs_T,axis=1)
        next_eigvecs_norm = next_eigvecs_T.T /norm_next_eigvecs
        
        indexes = []
        
        # for l in new_indexes:
        #     print(l)
        #     eig  = this_eigvecs_norm[:,l]
        #     # print(np.round(eig,2))
        #     dot = np.abs(np.dot(next_eigvecs_norm.T, eig))
        #     print(np.round(dot,2))
        #     index = int(np.argwhere(dot == dot.max()))
        #     if (index in indexes): #case of degeneracy
        #         dot[index] = 0
        #         index = int(np.argwhere(dot==dot.max()))
        #     indexes.append(index)

        this_eigvecs_reshaped = this_eigvecs_norm[:, new_indexes]
        
        dot = np.abs(np.dot(this_eigvecs_reshaped.conj().T, next_eigvecs_norm)) #* (np.dot(next_eigvecs_norm.T, this_eigvecs_reshaped)).conj()
        # print()
        # print(k)
        # # print(np.round(this_eigvecs_reshaped[:,0],2).reshape((5,3)))
        # for a in range(15):
        #     # print(np.round(next_eigvecs_norm[:,a].conj().reshape((5,3)),2))
        #     # print(np.round(this_eigvecs_reshaped[:,0].reshape(5,3), 2))
        #     print(np.round(np.dot(next_eigvecs_norm[:,a].conj(), this_eigvecs_reshaped[:,1]), 2))
        # # print(np.round(next_eigvecs_norm[:,:],2))
        # print(np.round(dot,2))
        for i in range(num_branches):
            row = dot[i,:]
            max_index, max_row = np.argwhere(row == max(row))[0,0], max(row)
            for j in range(num_branches):
                if(i==j):
                    continue
                this_row = dot[j,:]
                max_index_this_row, max_this_row = np.argwhere(this_row == max(this_row))[0,0], max(this_row)
                if(max_index_this_row == max_index):
                    if(max_row > max_this_row):
                        dot[j,max_index_this_row] = 0
                    else:
                        dot[i,max_index] = 0
        # print(np.round(dot,2))                   
        maxima = np.max(dot, axis=1)
        for l in range(num_branches):
            index = np.argwhere(dot[l,:] == maxima[l])[0,0]
            indexes.append(index)
            new_indexes = indexes #update new indexes
        # # uncomment this in the G-M path for BaTiO3 in the relax configuration to avoid band crossing at q = 2
        # if (k==1):
        #     indexes[7] = 7
        #     indexes[9] = 8
        # # uncomment this in the G-L path for BaTiO3 in the relax configuration to avoid band crossing at q = 3
        # if (k==2):
        #     indexes[3] = 4
        #     indexes[9] = 6
        all_indexes.append(indexes)    
        # print(new_indexes)
        # print()
    
    
    freqs_sorted = np.copy(freqs)
    eigvecs_sorted = np.copy(eigvecs)
    for j in range(1,num_ks):
        freqs_sorted[:,j] = freqs[all_indexes[j],j]
        eigvecs_sorted[j] = eigvecs[j][:,all_indexes[j]]
    
    return np.array(all_indexes), freqs_sorted, np.array(eigvecs_sorted)


import matplotlib.pyplot as plt
def plot_phonon_mode(
    k_cart,
    n,
    omega,
    eigvec,
    Ruc,
    Cell,
    masses=None,
    symbols=None,
    title="",
    ampl=1.0,
    repeat=(1, 1, 1),
    k_scale=1.0,
    show_imag=False,
):

    k_cart = np.asarray(k_cart, dtype=float)
    Ruc = np.asarray(Ruc, dtype=float)
    Cell = np.asarray(Cell, dtype=float)

    natoms = len(Ruc)

    eigvec = np.asarray(eigvec)
    if eigvec.ndim == 1:
        eigvec = eigvec.reshape(natoms, 3)

    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        eigvec = eigvec / np.sqrt(masses)[:, None]

    eigvec = ampl * eigvec

    print("# ===============================")
    print(f"Kpoint cartesian {np.round(k_cart, 4)} Mode {n}; frequency: {omega:.4f} THz\n")
    print("Real eigenvector:")
    print(np.round(np.real(eigvec), 4))
    print("Imaginary eigenvector:")
    print(np.round(np.imag(eigvec), 4))
    print("# ===============================")

    # Repeated atoms
    positions = []
    atom_indices = []

    for i in range(repeat[0]):
        for j in range(repeat[1]):
            for l in range(repeat[2]):
                shift = i * Cell[0] + j * Cell[1] + l * Cell[2]
                positions.append(Ruc + shift)
                atom_indices.extend(range(natoms))

    positions = np.vstack(positions)
    atom_indices = np.array(atom_indices)

    eig_real = np.real(eigvec)[atom_indices]
    eig_imag = np.imag(eigvec)[atom_indices]
    # Choose a global phase so the largest eigenvector component is real and positive
    imax = np.unravel_index(np.argmax(np.abs(eigvec)), eigvec.shape)
    phase = np.exp(-1j * np.angle(eigvec[imax]))
    
    eig_projected = np.real(eigvec * phase)
    eig_plot = eig_projected[atom_indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Atoms
    if symbols is None:
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            s=120, alpha=0.6, label="atoms"
        )
    else:
        symbols = np.asarray(symbols)
        for sym in np.unique(symbols):
            mask = symbols[atom_indices] == sym
            ax.scatter(
                positions[mask, 0],
                positions[mask, 1],
                positions[mask, 2],
                s=120,
                alpha=0.6,
                label=sym,
            )

    # Eigenvector arrows
    ax.quiver(
        positions[:, 0], positions[:, 1], positions[:, 2],
        eig_plot[:, 0], eig_plot[:, 1], eig_plot[:, 2],
        color="black",
        label="Re(eigenvector)",
    )

    if show_imag and np.any(np.abs(eig_imag) > 1e-12):
        ax.quiver(
            positions[:, 0], positions[:, 1], positions[:, 2],
            eig_imag[:, 0], eig_imag[:, 1], eig_imag[:, 2],
            color="red",
            alpha=0.6,
            label="Im(eigenvector)",
        )

    # Draw true repeated-cell parallelepiped
    full_cell = np.array([
        repeat[0] * Cell[0],
        repeat[1] * Cell[1],
        repeat[2] * Cell[2],
    ])

    corners_frac = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ])

    corners = corners_frac @ full_cell

    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]

    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="black",
            alpha=0.45,
            linewidth=1,
        )

    # Cartesian k-vector: use directly, no reciprocal conversion
    if np.linalg.norm(k_cart) > 1e-12:
        origin = np.mean(positions, axis=0)

        # scale relative to atom/cell size, but direction is exactly k_cart
        cell_length = np.mean(np.linalg.norm(Cell, axis=1))
        k_arrow = k_cart / np.linalg.norm(k_cart) * k_scale * cell_length

        ax.quiver(
            origin[0], origin[1], origin[2],
            k_arrow[0], k_arrow[1], k_arrow[2],
            color="red",
            linewidth=3,
            label="k-vector",
        )

        ax.text(
            origin[0] + k_arrow[0],
            origin[1] + k_arrow[1],
            origin[2] + k_arrow[2],
            "k",
            color="red",
            fontsize=14,
        )

    # Clean axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_title(
        f"{title}\n"
        f"k_cart = {np.round(k_cart, 4)}, mode = {n}, frequency = {omega:.4f} THz"
    )

    # Use actual plotted geometry for limits, not a huge cube
    all_points = np.vstack([
        positions,
        positions + eig_plot,
        corners,
    ])

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    padding = 0.08 * np.max(maxs - mins)
    if padding == 0:
        padding = 1.0

    ax.set_xlim(mins[0] - padding, maxs[0] + padding)
    ax.set_ylim(mins[1] - padding, maxs[1] + padding)
    ax.set_zlim(mins[2] - padding, maxs[2] + padding)

    # Matplotlib 3D aspect ratio: preserve true cell shape
    ranges = maxs - mins
    ax.set_box_aspect(ranges)

    ax.legend()
    ax.view_init(elev=15, azim=-35)

    plt.tight_layout()
    plt.show()

