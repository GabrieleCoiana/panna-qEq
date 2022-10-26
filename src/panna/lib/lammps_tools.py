""" Collection of tools to interact with lammps
"""
import numpy as np


def convert_lattice_to_lmp(lattice_vectors):
    """ convert 3 lattice vectors to lammps format.

    This procedure is defined in lammps documentation
    lammps.sandia.gov/doc/Howto_triclinic.html

    Parameters
    ----------
    lattice_vectors: 3x3 array, original lattice vectors in rows
                     [a1, a2, a3]

    Returns
    -------
    converted lattice matrix row wise:
    [[xhi - xlo, 0, 0],
     [xy, yhi - ylo, 0],
     [xz, yz, zhi - zlo]]

    """
    lattice_vectors = np.asarray(lattice_vectors)
    a_norm = np.linalg.norm(lattice_vectors[0])
    a_versor = lattice_vectors[0] / np.linalg.norm(lattice_vectors[0])

    a_x = a_norm
    b_x = np.dot(lattice_vectors[1], a_versor)
    b_y = np.linalg.norm(np.cross(a_versor, lattice_vectors[1]))
    c_x = np.dot(lattice_vectors[2], a_versor)
    c_y = (np.dot(lattice_vectors[1], lattice_vectors[2]) - b_x * c_x) / b_y
    c_z = np.sqrt(np.linalg.norm(lattice_vectors[2])**2 - c_x**2 - c_y**2)

    return np.asarray([[a_x, 0, 0], [b_x, b_y, 0], [c_x, c_y, c_z]])


def convert_vectors_to_lmp(lattice_vectors, vectors):
    """Convert one or more vectors to lmp with the required rotation/inversion

    This procedure is defined in lammps documentation
    lammps.sandia.gov/doc/Howto_triclinic.html

    Parameters
    ----------
    lattice_vectors: 3x3 array, original lattice vectors in rows
                     [a1, a2, a3]
    vectors: numpy array ?, 3
             vector(s) to be converted
    Return
    ------
    converted vector(s)
    """
    lattice_vectors = np.asarray(lattice_vectors)
    vectors = np.asarray(vectors, dtype=np.float)

    volume = np.abs(np.linalg.det(lattice_vectors))
    lmp_lattice = convert_lattice_to_lmp(lattice_vectors)

    part_1 = np.transpose(lmp_lattice)
    part_2 = np.cross([lattice_vectors[1], lattice_vectors[2], lattice_vectors[0]],
                      [lattice_vectors[2], lattice_vectors[0], lattice_vectors[1]])

    transform_matrix = np.dot(part_1, part_2) / volume

    return np.dot(vectors, transform_matrix.T)


def convert_vectors_from_lmp(lattice_vectors, vectors):
    """Convert one or more vectors to lmp with the required rotation/inversion

    This procedure is defined in lammps documentation
    lammps.sandia.gov/doc/Howto_triclinic.html

    Parameters
    ----------
    lattice_vectors: 3x3 array, original lattice vectors in rows
                     [a1, a2, a3]
    vectors: numpy array ?, 3
             vector(s) to be converted
    Return
    ------
    converted vector(s)
    """

    lattice_lammps = convert_lattice_to_lmp(lattice_vectors)

    volume = np.abs(np.linalg.det(lattice_vectors))

    bxc = np.cross(lattice_vectors[1], lattice_vectors[2])
    cxa = np.cross(lattice_vectors[2], lattice_vectors[0])
    axb = np.cross(lattice_vectors[0], lattice_vectors[1])

    converter = lattice_lammps.T @ np.array([bxc, cxa, axb]) / volume

    return vectors @ np.linalg.inv(converter).T
