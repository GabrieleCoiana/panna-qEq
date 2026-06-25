"""Utilities to handling the input system
"""

import tensorflow as tf
import tensorflow.io as tfio
from panna.gvector.pbc import make_replicas
from panna.lib import ExampleJsonWrapper
import numpy as np
import random
import os
import shlex
from functools import partial
try:
    import h5py
except ImportError:
    pass

class ParseFn():
    """Parse TFExample records and perform simple data augmentation.

    Parameters
    ----------
        g_size: int
            size of the g_vector
        zeros:
            array of zero's one value per specie.
        n_species: int
            number of species
        forces: boolean
            recover forces AND dg/dx
        energy_rescale: float
            scale the energy

    """
    def __init__(self,
                 g_size: int,
                 n_species: int,
                 forces: bool = False,
                 sparse_dgvect: bool = False,
                 energy_rescale: float = 1.0,
                 long_range_el: bool = False,
                 names: bool = False):
        self._g_size = g_size
        self._n_species = n_species
        self._forces = forces
        self._sparse_dgvect = sparse_dgvect
        self._energy_rescale = energy_rescale
        self._long_range_el = long_range_el
        self._names = names

    @property
    def feature_description(self):
        """Features of the example."""
        feat = {}
        feat["energy"] = tfio.FixedLenFeature([], dtype=tf.float32)
        feat["species"] = tfio.FixedLenSequenceFeature([],
            dtype=tf.int64,
            allow_missing=True,
            default_value=self._n_species)
        feat["gvects"] = tfio.FixedLenSequenceFeature(
            shape=[self._g_size],
            dtype=tf.float32,
            allow_missing=True)
        if self._names:
            feat["name"] = tfio.FixedLenFeature([], dtype=tf.dtypes.string,
                                                default_value=b'N.A.')
        if self._forces:
            if self._sparse_dgvect:
                feat["dgvect_size"] = tfio.FixedLenFeature([], dtype=tf.int64)
                feat["dgvect_values"] = tfio.FixedLenSequenceFeature([],
                    dtype=tf.float32,
                    allow_missing=True)
                feat["dgvect_indices1"] = tfio.FixedLenSequenceFeature([],
                    dtype=tf.float32,
                    allow_missing=True)
                feat["dgvect_indices2"] = tfio.FixedLenSequenceFeature([],
                    dtype=tf.float32,
                    allow_missing=True)
            else:
                feat["dgvects"] = tfio.FixedLenSequenceFeature([],dtype=tf.float32,
                                                               allow_missing=True)
            feat["forces"] = tfio.FixedLenSequenceFeature([],
                dtype=tf.float32,
                allow_missing=True)
        if self._long_range_el:
            feat['el_energy_kernel'] = tfio.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True, default_value=0.0)
            if self._forces:
                feat['el_force_kernel'] = tfio.FixedLenSequenceFeature(
                    [], dtype=tf.float32, allow_missing=True, default_value=0.0)
            feat['atomic_charges'] = tfio.FixedLenSequenceFeature(
                [], dtype=tf.float32, allow_missing=True, default_value=0.0)
            feat['total_charge'] = tfio.FixedLenFeature([], dtype=tf.float32)



        return feat

    def _post_processing(self, example):
        example['energy'] = example['energy'] * self._energy_rescale
        return example

    def __call__(self, serialized):
        """Return a sample ready to be batched.

        Return
        ------
            species_tensor: Sparse Tensor, (n_atoms) value in range(n_species)
            g_vectors_tensor: Sparse Tensor, (n_atoms, g_size)
            energy: true energy value corrected with the zeros
        """
        examples = tfio.parse_example(serialized, features=self.feature_description)
        examples = self._post_processing(examples)
        return examples


# Some consts
sym2num = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 
'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 
'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 
'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 
'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 
'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 
'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 
'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 
'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 
'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 
'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 
'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 
'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 
'Lv': 116, 'Ts': 117, 'Og': 118}
bohr2A = np.float32(0.52917721067)
Ha2eV = np.float32(2*13.6056980659)

# Loading directly a batch of data as a list of pairs
def load_json(data_dir, Rcut, species_list, get_forces, shuffle):
    all_examples = []
    all_targets = []
    if type(data_dir)!=np.ndarray:
        data_dir = [data_dir]
    for i,d in enumerate(data_dir):
        if d != b'':
            for file in os.listdir(d):
                if os.path.splitext(file)[-1] == b'.example':
                    all_examples.append(file)
                    all_targets.append(i)
    if len(all_examples) == 0:
        raise ValueError('No example file found.')
    # Species list decoded because it comes as b'' string
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    nex = len(all_examples)
    if shuffle:
        # Co-shuffling files and indices
        inds = np.arange(nex)
        random.shuffle(inds)
        all_examples = [all_examples[i] for i in inds]
        all_targets = [all_targets[i] for i in inds]
    for i in range(0,nex):
        target = all_targets[i]
        example = ExampleJsonWrapper(os.path.join(data_dir[target],all_examples[i]),species_list)
        positions, replicas = make_replicas(example.angstrom_positions, \
                                      example.angstrom_lattice_vectors, Rcut)
        nats = len(example.angstrom_positions)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,-1,3))
        rij_vec = posj+rep-posi
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))
        # Mask with the cutoff
        radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
        indall = np.where(radial_mask)
        inda = indall[0]
        indb = indall[1]
        yield {'nats': nats,
               'nn_r': rij[indall],
               'nn_vecs': rij_vec[indall],
               'species': example.species_indexes,
               'sp_a': example.species_indexes[inda],
               'sp_b': example.species_indexes[indb],
               'inda': inda,
               'indb': indb,
               'energy': example.ev_energy,
               'name': all_examples[i][:-8],
               'forces': example.forces,
               'targets': [target]*nats,
               'conf_targets': target,
               }

# Loading from .xyz file(s)
def load_xyz(data_dir, Rcut, species_list, get_forces, shuffle):
    all_examples = []
    all_names = []
    if type(data_dir)==np.ndarray:
        raise "xyz format works with a single .xyz or a directory"
    if data_dir[-4:]==b'.xyz':
        files = [data_dir.decode('utf-8')]
    else:
        files = []
        for file in os.listdir(d):
            if os.path.splitext(file)[-1] == '.xyz':
                files.append(file)
    # Reading all data at once.. for large datasets we can make another version
    for file in files:
        basename = os.path.basename(file)[:-4].replace(" ","_")+'_'
        with open(file, 'r') as f:
            lines = f.readlines()
            i = 0
            j = 1
            while i<len(lines):
                n = int(lines[i])
                all_examples.append(lines[i:i+n+2])
                all_names.append(basename+str(j))
                i += n+2
                j += 1
    if len(all_examples) == 0:
        raise ValueError('No example found.')
    # Species list decoded because it comes as b'' string
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    nex = len(all_examples)
    if shuffle:
        # Co-shuffling files and indices
        inds = np.arange(nex)
        random.shuffle(inds)
        all_examples = [all_examples[i] for i in inds]
        all_names = [all_names[i] for i in inds]
    for i in range(0,nex):
        nats = int(all_examples[i][0])
        # Split on spaces preserving chunks in quotes
        descr = shlex.split(all_examples[i][1])
        lattice = np.zeros((3,3))
        for d in descr:
            name, prop = d.split('=')
            if name.lower()=='lattice':
                lattice = np.asarray([float(x) for x in prop.strip('"').split()]).reshape((3,3))
            elif name.lower()=='energy':
                energy = float(prop)
            elif name.lower()=='properties':
                chunks = prop.split(":")
                k = 0
                for j in range(0,len(chunks),3):
                    t = chunks[j]
                    if t=='species':
                        spcol = k
                    elif t=='pos':
                        poscol = k
                    elif t=='forces':
                        forcol = k
                    k += int(chunks[j+2])
        positions = []
        species = []
        forces = []
        for l in all_examples[i][2:]:
            vals = l.split()
            positions.append([float(v) for v in vals[poscol:poscol+3]])
            species.append(species_list.index(vals[spcol]))
            forces.append([float(v) for v in vals[forcol:forcol+3]])
        species = np.asarray(species)
        positions, replicas = make_replicas(positions, lattice, Rcut)
        posi = np.reshape(positions,(nats,1,1,3))
        posj = np.reshape(positions,(1,nats,1,3))
        rep = np.reshape(replicas,(1,1,-1,3))
        rij_vec = posj+rep-posi
        rij = np.sqrt(np.sum(rij_vec**2, axis=3))
        # Mask with the cutoff
        radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
        indall = np.where(radial_mask)
        inda = indall[0]
        indb = indall[1]
        yield {'nats': nats,
               'nn_r': rij[indall],
               'nn_vecs': rij_vec[indall],
               'species': species,
               'sp_a': species[inda],
               'sp_b': species[indb],
               'inda': inda,
               'indb': indb,
               'energy': energy,
               'name': all_names[i],
               'forces': forces,
               }

# Loading from spice hdf5 format
def load_spice(data_dir, Rcut, species_list, get_forces, shuffle):
    # Expecting data in Ha, bohr, with no PBC
    Ffact = -Ha2eV/bohr2A
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    # Creating conversion from atomic number to progressive index
    num2ord = {sym2num[s]:i for i,s in enumerate(species_list)}
    data = h5py.File(data_dir, 'r')
    molist = [(mo, i) for mo in data.keys() if len(data[mo])>0 \
                      for i in range(len(data[mo]['formation_energy']))]
    random.shuffle(molist)
    nex = len(molist)
    for i in range(0,nex):
        mol, ind = molist[i]
        spnums = list(data[mol]['atomic_numbers'])
        nats = len(spnums)
        species = np.asarray([num2ord[n] for n in spnums])
        posi = np.reshape(data[mol]['conformations'][ind]*bohr2A,(nats,1,3))
        posj = np.reshape(data[mol]['conformations'][ind]*bohr2A,(1,nats,3))
        rij_vec = posj-posi
        rij = np.sqrt(np.sum(rij_vec**2, axis=2))
        # Mask with the cutoff
        radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
        indall = np.where(radial_mask)
        inda = indall[0]
        indb = indall[1]
        yield {'nats': nats,
               'nn_r': rij[indall],
               'nn_vecs': rij_vec[indall],
               'species': species,
               'sp_a': species[inda],
               'sp_b': species[indb],
               'inda': inda,
               'indb': indb,
               'energy': data[mol]['formation_energy'][ind]*Ha2eV,
               'name': mol+str(ind),
               'forces': data[mol]['dft_total_gradient'][ind] * Ffact,
               }

# Loading data in numpy format
def load_npy(data_dir, Rcut, species_list, get_forces, shuffle):
    species_list = [s.decode('utf-8') for s in species_list]
    nsp = len(species_list)
    # Creating conversion from atomic number to progressive index
    num2ord = {sym2num[s]:i for i,s in enumerate(species_list)}
    all_files = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[-1] == b'.npy':
            all_files.append(file)    
    if len(all_files) == 0:
        raise ValueError('No data file found in {}.'.format(data_dir))
    # Shuffling files, loading one at a time
    random.shuffle(all_files)
    for f in all_files:
        data = np.load(data_dir+b'/'+f, allow_pickle=True)
        nex = len(data)-1
        indices = np.arange(1,nex+1)
        np.random.shuffle(indices)
        for i in range(0,nex):
            conf = data[indices[i]]
            spnums = list(conf[2])
            nats = conf[1]
            species = np.asarray([num2ord[n] for n in spnums])
            positions, replicas = make_replicas(conf[5], conf[3], Rcut)
            posi = np.reshape(positions,(nats,1,1,3))
            posj = np.reshape(positions,(1,nats,1,3))
            rep = np.reshape(replicas,(1,1,-1,3))
            rij_vec = posj+rep-posi
            rij = np.sqrt(np.sum(rij_vec**2, axis=3))
            # Mask with the cutoff
            radial_mask = np.logical_and(rij < Rcut, rij > 1e-8)
            indall = np.where(radial_mask)
            inda = indall[0]
            indb = indall[1]
            name = conf[4]
            if len(conf)>8:
                name = name+'_'+conf[8]
            yield {'nats': nats,
               'nn_r': rij[indall],
               'nn_vecs': rij_vec[indall],
               'species': species,
               'sp_a': species[inda],
               'sp_b': species[indb],
               'inda': inda,
               'indb': indb,
               'energy': conf[6],
               'name': name,
               'forces': conf[7],
               }