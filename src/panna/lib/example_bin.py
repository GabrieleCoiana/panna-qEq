"""Wrapper for the binary example format."""

import os

import numpy as np


class Example(object):
    """ Example class, a class to handle a binary PANNA example.

    Parameters
    ----------
        g_vectors: (n_atoms, g_size)
            list of g's, one row one atom in the example
        species_vector: (n_atoms)
            list of idx, one per atom in the example
        true_energy: (1)
            value of the true energy
        d_g_vectors: (n_atoms, g_size, n_atoms * 3), opt
            derivative of the gvect
        forces: (n_atoms * 3), opt
            forces acting on the atoms
        per_atom_quantity: (n_atoms), opt
            a per atom quantity that can be learn (eg charge)
        atomic_species: list of strings,
                        with atomic species name
        el_energy_kernel: (n_atoms, n_atoms), opt
            electrostatics energy kernel
        el_force_kernel: (n_atoms, n_atoms, 3), opt
            electrostatics force kernel
        atomic_charges: (n_atoms), opt
            atomic charges
        total_charge: (1), opt
            system total charge
        name: arbitrary name
    """
    def __init__(self,
                 g_vectors,
                 species_vector,
                 true_energy,
                 d_g_available=False,
                 d_g_vectors=np.empty(0),
                 d_g_vector_values=np.empty(0),
                 d_g_vector_indices1=np.empty(0),
                 d_g_vector_indices2=np.empty(0),
                 forces=np.empty(0),
                 per_atom_quantity=np.empty(0),
                 atomic_species=None,
                 el_energy_kernel=np.empty(0),
                 el_force_kernel=np.empty(0),
                 atomic_charges=np.empty(0),
                 total_charge=0.0,
                 name=None):

        self._g_vectors = np.asarray(g_vectors)
        self._species_vector = np.asarray(species_vector)
        self._true_energy = true_energy
        self._n_species = max(set(species_vector)) + 1

        self._d_g_available = d_g_available
        self._d_g_vectors = np.asarray(d_g_vectors)
        self._d_g_vector_values = np.asarray(d_g_vector_values)
        self._d_g_vector_indices1 = np.asarray(d_g_vector_indices1)
        self._d_g_vector_indices2 = np.asarray(d_g_vector_indices2)
        self._forces = np.asarray(forces)

        self._per_atom_quantity = per_atom_quantity

        self._atomic_species = atomic_species
        self._el_energy_kernel = np.asarray(el_energy_kernel)
        self._el_force_kernel = np.asarray(el_force_kernel)
        self._atomic_charges = np.asarray(atomic_charges)
        self._total_charge = total_charge

        self._name = name or 'NA'

    @property
    def true_energy(self):
        return self._true_energy

    @property
    def n_atoms(self):
        return len(self._species_vector)

    @property
    def atoms_per_species(self):
        unique, counts = np.unique(self._species_vector, return_counts=True)
        return counts

    @property
    def species_vector(self):
        return self._species_vector

    @property
    def gvects(self):
        return self._g_vectors

    @property
    def g_size(self):
        return np.shape(self._g_vectors)[1]

    @property
    def dgvects(self):
        if not self._d_g_available:
            raise ValueError("g derivative Not available")

        if self._d_g_vectors.size > 0:
            return self._d_g_vectors

        g_size = self.g_size
        self._d_g_vectors = np.zeros((self.n_atoms, g_size, self.n_atoms * 3))
        dgv = self.dgvect_values
        dgi1 = self.dgvect_indices1.astype(np.int32)
        dgi2 = self.dgvect_indices2.astype(np.int32)
        i1, i2 = np.divmod(dgi1, g_size)
        self._d_g_vectors[i1, i2, dgi2] = dgv
        return self._d_g_vectors

    @property
    def dgvect_values(self):
        if not self._d_g_available:
            raise ValueError("sparse g derivative values Not available")
        return self._d_g_vector_values

    @property
    def dgvect_indices1(self):
        if not self._d_g_available:
            raise ValueError("sparse g derivative index1 Not available")
        return self._d_g_vector_indices1

    @property
    def dgvect_indices2(self):
        if not self._d_g_available:
            raise ValueError("sparse g derivative index2 Not available")

        return self._d_g_vector_indices2

    @property
    def forces(self):
        if self._forces.size > 0:
            return self._forces
        raise ValueError("forces Not available")

    @property
    def per_atom_quantity(self):
        if self._per_atom_quantity.size > 0:
            return self._per_atom_quantity
        raise ValueError("per atom quantity Not available")

    @property
    def name(self):
        filename, _dummy_extension = os.path.splitext(self._name)
        return filename

    @property
    def el_energy_kernel(self):
        if self._el_energy_kernel.size > 0:
            return self._el_energy_kernel
        raise ValueError("electrostatic energy kernel Not available")
    @property
    def el_force_kernel(self):
        if self._el_force_kernel.size > 0:
            return self._el_force_kernel
        raise ValueError("electrostatic force kernel Not available")
    @property
    def total_charge(self):
        return self._total_charge
    @property
    def atomic_charges(self):
        return self._atomic_charges

def load_example(file_name):
    """Load binary example.

    Parameters
    ----------
      file name: a file name to load

    Return
    ------
      Example object for internal computation
    """
    key = os.path.split(file_name)[-1]
    with open(file_name, "rb") as binary_file:
        bin_version = int.from_bytes(binary_file.read(4),
                                     byteorder='little',
                                     signed=False)
        if bin_version != 0:
            raise NotImplementedError()
        # converting to int to avoid handling little/big endian
        flags = int.from_bytes(binary_file.read(2), byteorder='little', signed=False)
        derivative_flag = flags & 0b00000001
        force_flag = flags & 0b00000010
        per_atom_quantities_flag = flags & 0b00000100
        sparse_derivative_flag = flags & 0b00001000
        el_flag = flags & 0b00010000 
        n_atoms = int.from_bytes(binary_file.read(4), byteorder='little', signed=False)
        g_size = int.from_bytes(binary_file.read(4), byteorder='little', signed=False)
        payload = binary_file.read()

    # assuming machine that created the binary file is little endian
    data = np.frombuffer(payload, dtype='<f4')

    energy = np.reshape(data[0], [1])[0]

    spec_tensor_bytes = n_atoms
    gvect_tensor_bytes = n_atoms * g_size
    prev_bytes = 1

    spec_tensor = data[prev_bytes:prev_bytes + spec_tensor_bytes]
    spec_tensor = np.int64(np.reshape(spec_tensor, [n_atoms]))

    prev_bytes += spec_tensor_bytes
    gvect_tensor = data[prev_bytes:prev_bytes + gvect_tensor_bytes]
    gvect_tensor = np.reshape(gvect_tensor, [n_atoms, g_size])
    prev_bytes += gvect_tensor_bytes

    if derivative_flag:
        if data.size == prev_bytes:
            raise ValueError('Derivatives requested but not ' 'present in the file')
        if not sparse_derivative_flag:
            dgvec_tensor_bytes = n_atoms**2 * g_size * 3
            dgvect_tensor = data[prev_bytes:prev_bytes + dgvec_tensor_bytes]
            dgvect_tensor = np.reshape(dgvect_tensor, [n_atoms, g_size, n_atoms * 3])
            prev_bytes += dgvec_tensor_bytes
            dgvect_tensor_values = np.empty(0)
            dgvect_tensor_indices1 = np.empty(0)
            dgvect_tensor_indices2 = np.empty(0)
        else:
            dgvect_elements = int(data[prev_bytes])
            prev_bytes += 1
            dgvect_tensor_values = data[prev_bytes:prev_bytes + dgvect_elements]
            prev_bytes += dgvect_elements
            dgvect_tensor_indices = data[prev_bytes:prev_bytes + 2 * dgvect_elements]
            dgvect_tensor_indices = np.reshape(dgvect_tensor_indices,
                                               [dgvect_elements, 2])
            dgvect_tensor_indices1 = dgvect_tensor_indices[:, 0]
            dgvect_tensor_indices2 = dgvect_tensor_indices[:, 1]
            prev_bytes += 2 * dgvect_elements
            dgvect_tensor = np.empty(0)
    else:
        dgvect_tensor = np.empty(0)
        dgvect_tensor_values = np.empty(0)
        dgvect_tensor_indices1 = np.empty(0)
        dgvect_tensor_indices2 = np.empty(0)

    if force_flag:
        if data.size == prev_bytes:
            raise ValueError('Forces requested but not present in the file')
        forces_bytes = n_atoms * 3
        forces = data[prev_bytes:prev_bytes + forces_bytes]
        prev_bytes += forces_bytes
    else:
        forces = np.empty(0)

    if per_atom_quantities_flag:
        per_atom_quantity = data[prev_bytes:prev_bytes + n_atoms]
    else:
        per_atom_quantity = np.empty(0)

    if el_flag:
        el_energy_kernel_bytes = n_atoms * n_atoms
        total_charge_bytes = 1
        atomic_charges_bytes = n_atoms

        el_energy_kernel = data[prev_bytes:prev_bytes + el_energy_kernel_bytes]
        el_energy_kernel = np.reshape(el_energy_kernel,[n_atoms, n_atoms])
        prev_bytes += el_energy_kernel_bytes

        if  derivative_flag:
            el_force_kernel_bytes = n_atoms * n_atoms * 3
            el_force_kernel = data[prev_bytes:prev_bytes + el_force_kernel_bytes]
            el_force_kernel = np.reshape(el_force_kernel, [n_atoms, n_atoms, 3])
            prev_bytes += el_force_kernel_bytes
        else:
            el_force_kernel=np.empty(0)

        atomic_charges = data[prev_bytes:prev_bytes+atomic_charges_bytes]
        prev_bytes += atomic_charges_bytes

        total_charge = data[prev_bytes:prev_bytes+total_charge_bytes]


    else:
        el_energy_kernel=np.empty(0)
        el_force_kernel=np.empty(0)
        total_charge = 0.0
        atomic_charges = np.empty(0)
    return Example(name=key,
                   true_energy=energy,
                   species_vector=spec_tensor,
                   g_vectors=gvect_tensor,
                   d_g_available=derivative_flag,
                   d_g_vectors=dgvect_tensor,
                   d_g_vector_values=dgvect_tensor_values,
                   d_g_vector_indices1=dgvect_tensor_indices1,
                   d_g_vector_indices2=dgvect_tensor_indices2,
                   forces=forces,
                   per_atom_quantity=per_atom_quantity,
                   el_energy_kernel=el_energy_kernel,
                   el_force_kernel=el_force_kernel,
                   atomic_charges=atomic_charges,
                   total_charge=total_charge)
