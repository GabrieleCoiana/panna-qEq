###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import json
import logging
import os
from string import Template

import numpy as np

from .atomic_property import recover_property
from .lammps_tools import convert_lattice_to_lmp, convert_vectors_to_lmp
from .units_converter import distances_to_angstrom, energy_to_ev

logger = logging.getLogger(__name__)


class ExampleJsonWrapper():
    """A wrapper around PANNA json files

    Parameters
    ----------
    file_name: file to inherit form
    atomic_sequence: converter from string to integer for species
                      infer if not passed

    """

    # TODO it would be nice to parse the atom key separately and have a setter
    # for the unit of length such as later one can simply set the units and the
    # whole class will act consequently
    def __init__(self, file_name, atomic_sequence=None,
                 species_gaussian_width=None,
                 atomic_hardness=None):

        with open(file_name) as file_stream:
            example = json.load(file_stream)

        self._example = example
        self._file = file_name

        if atomic_sequence:
            self._atomic_sequence = atomic_sequence
        else:
            species = []
            for atom in self._example['atoms']:
                species.append(atom[1])
            self._atomic_sequence = list(set(species))
        if species_gaussian_width:
            self._species_gaussian_width = species_gaussian_width
        else:
            self._species_gaussian_width = []

    @property
    def unit_of_length(self):
        """unit of length in the example"""
        unit_of_length = self._example.get('unit_of_length', None)
        if unit_of_length is None:
            logger.warning(f'unit_of_length unknown, assumed angstrom {self.key}')
            unit_of_length = 'angstrom'
        return unit_of_length

    @property
    def unit_of_energy(self):
        """unit of energy in the example"""
        try:
            unit_of_energy = self._example.get('energy', [])[1]
        except IndexError:
            logger.warning(f'unit of energy unknown, assumed eV {self.key}')
            unit_of_energy = 'eV'
        return unit_of_energy

    @property
    def lattice_vectors(self):
        """lattice vectors in the example or 3x3 zero matrix
        """
        lattice_vectors = self._example.get('lattice_vectors',
                                            np.zeros(9).reshape(3, 3))
        return np.asarray(lattice_vectors).astype(float)

    @property
    def angstrom_lattice_vectors(self):
        """lattice vectors in angstrom"""
        return np.multiply(self.lattice_vectors,
                           distances_to_angstrom(self.unit_of_length))

    @property
    def lmp_lattice_vectors(self):
        """return lammps matrix for lattice vectors, units as specified
           in unit of length
           matrix:
           [[xhi - xlo, 0, 0],
            [xy, yhi - ylo, 0],
            [xz, yz, zhi - zlo]]
        """
        return convert_lattice_to_lmp(self.angstrom_lattice_vectors)

    @property
    def lmp_atomic_positions(self):
        """return atomic positions rotated/translated as required by lammps
           in unit of length
        """
        return convert_vectors_to_lmp(self.angstrom_lattice_vectors,
                                      self.angstrom_positions)

    @property
    def atomic_position_unit(self):
        """return atomic position unit string"""
        atomic_position_unit = self._example.get('atomic_position_unit', None)
        if atomic_position_unit is None:
            # legacy key in dictionary
            atomic_position_unit = self._example.get('atomic_coordinates', None)
        return atomic_position_unit

    @property
    def positions(self):
        """return atomic positions in cartesian coordinate"""
        pos = []
        for atom in self._example['atoms']:

            if self.atomic_position_unit == 'cartesian':
                pos.append(np.asarray(atom[2]).astype(float))

            elif self.atomic_position_unit == 'crystal':
                pos.append(
                    np.dot(np.transpose(self.lattice_vectors),
                           np.asarray(atom[2]).astype(float)))
            else:
                raise ValueError('atomic_position_unit unknown')
        return np.asarray(pos)

    @property
    def angstrom_positions(self):
        """return atomic positions in angstrom"""
        return np.multiply(self.positions, distances_to_angstrom(self.unit_of_length))

    @property
    def forces(self):
        """ return forces in eV/Angstrom
        """
        forces = []
        for atom in self._example['atoms']:
            try:
                force = np.asarray(atom[3]).astype(float) * energy_to_ev(
                    self.unit_of_energy) / distances_to_angstrom(self.unit_of_length)
                forces.append(force)
            except IndexError:
                raise ValueError('forces were requested but are not available')
        return np.asarray(forces)

    @property
    def species_indexes(self):
        """ return list of integers to better manage species type
        """
        idxes = []
        for atom in self._example['atoms']:
            try:
                idxes.append(self.species_str_2idx[atom[1]])
            except KeyError:
                raise KeyError(f'Species {atom[1]} Not found')
        return np.asarray(idxes)

    @property
    def species_str(self):
        """ return list of strings
        """
        species = []
        for atom in self._example['atoms']:
            species.append(atom[1])
        return species

    @property
    def energy(self):
        """ return the energy """
        # TODO: hotfix
        energy = self._example['energy'][0]
        if isinstance(energy, str):
            energy = energy.strip('\"')
        return float(energy)

    @property
    def per_atom_energy(self):
        """ return the per atom energy,
        if not available raise value error
        """
        try:
            pa_energy = self._example['per_atom_energy']
        except KeyError:
            raise ValueError('per atom energy not available')
        return np.asarray(pa_energy, dtype=float)

    @property
    def atomic_charges(self):
        """ recover the per atom charge
        """
        charges = []
        for atom in self._example['atoms']:
            if len(atom) > 4:
                charges.append(atom[4])
            else:
                charges.append(0.0)
        return np.asarray(charges, dtype=float)

    @property
    def ev_energy(self):
        """ return the energy in eV"""
        return self.energy * energy_to_ev(self.unit_of_energy)

    @property
    def per_atom_ev_energy(self):
        """ return the per atom energy in eV,
        if not available raise value error
        """
        return self.per_atom_energy * energy_to_ev(self.unit_of_energy)

    @property
    def key(self):
        """ a unique key, for now simple file name"""
        return os.path.basename(self._file).split('.')[0]

    @property
    def number_of_atoms(self):
        """number of atoms"""
        return len(self._example['atoms'])

    @property
    def species(self):
        """list of species in the examples

        converter idx to string
        """
        return self._atomic_sequence

    @property
    def species_str_2idx(self):
        """converter form string to idx"""
        return {species: idx for idx, species in enumerate(self._atomic_sequence)}

    @property
    def number_of_species(self):
        """number of species"""
        return len(self.species)
    @property
    def tmp_species_gaussian_width(self):
        """ converter string atomic type to value
        """
        return {
            x: y
            for x, y in zip(self._atomic_sequence, self._species_gaussian_width)}

    @property
    def per_atom_gaussian_width(self):
        """ recover the per atom gaussian width
        """
        alpha = []

        for count, atom in enumerate(self._example['atoms']):
            specie = atom[1]
            alpha.append(self.tmp_species_gaussian_width[specie])
        return np.asarray(alpha, dtype=np.float)
    @property
    def tmp_atomic_hardness(self):
        """ converter string atomic type to value
        """
        return {
            x: y
            for x, y in zip(self._atomic_sequence, self._atomic_hardness)}

    @property
    def per_atom_hardness(self):
        """ recover the per atom hardness
        """
        hardness = []
        for atom in self._example['atoms']:
            specie = atom[1]
            hardness.append(self.tmp_atomic_hardness[specie])
        return np.asarray(hardness, dtype=np.float)

    @property
    def total_charge(self):
        """ return the total charge """
        try:
            q = self._example['total_charge']
        except:
            q = 0.0

        if isinstance(q, str):
            q = q.strip('\"')
        return float(q)


    def to_lammps(self):
        """return a string ready to be written as pos lammps file,
        units are the same as in unit of length
        """
        string = '# {}\n\n'.format(self.key)
        string += '{} atoms\n'.format(self.number_of_atoms)
        string += '{} atom types\n\n'.format(self.number_of_species)

        lattice_vectors = self.lmp_lattice_vectors
        # pylint: disable=invalid-name
        dx, dy = lattice_vectors[0, 0], lattice_vectors[1, 1]
        xy, xz, yz = lattice_vectors[1, 0], lattice_vectors[2, 0], \
            lattice_vectors[2, 1]

        # from lammps.sandia.gov/doc/Howto_triclinic.html
        # To avoid extremely tilted boxes, LAMMPS normally requires that no
        # tilt factor can skew the box more than half the distance of the
        # parallel box length, which is the 1st dimension in the tilt factor
        # (x for xz) ....... eg.
        # if xlo = 2 and xhi = 12, then the x box length is 10 and the xy
        # tilt factor must be between -5 and 5. Similarly,
        # both xz and yz must be between -(xhi-xlo)/2 and +(yhi-ylo)/2.
        # from lammps.sandia.gov/doc/box.html
        # Note that if a simulation box has a large tilt factor,
        # LAMMPS will run less efficiently, due to the large volume of
        # communication needed to acquire ghost atoms around a processor’s
        # irregular-shaped sub-domain. For extreme values of tilt, LAMMPS may
        # also lose atoms and generate an error.
        def _check_lammps_boundary(var, min_value, max_value):
            if var < -min_value / 2 or var > max_value / 2:
                return True
            return False

        if _check_lammps_boundary(xy, dx, dx) or\
           _check_lammps_boundary(xz, dx, dy) or\
           _check_lammps_boundary(yz, dx, dy):
            logger.warning(
                '%s to run lammps add *box tilt large* '
                'in your input file '
                'lammps performance can degrade', self.key)
        # pylint: enable=invalid-name

        string += '{} {} xlo xhi\n'.format(0.0, dx)
        string += '{} {} ylo yhi\n'.format(0.0, dy)
        string += '{} {} zlo zhi\n'.format(0.0, lattice_vectors[2, 2])
        string += '{} {} {} xy xz yz\n\n'.format(xy, xz, yz)

        string += 'Masses\n\n'
        for idx, specie in enumerate(self.species):
            string += '{} {}\n'.format(idx + 1, recover_property(specie, 'mass'))
        string += '\nAtoms\n\n'

        for idx, (atomic_species, atomic_position) in enumerate(
                zip(self.species_indexes, self.lmp_atomic_positions)):
            string += '{} {} {} {} {}\n'.format(idx + 1, atomic_species + 1,
                                                *atomic_position)
        return string

    def to_qe_cards(self, *, kpoints_density=None, kpoints_grid=None):
        """return a dict with simulation related part of qe input file

        parameters
        ----------
        kpoints_density: k-points density in angstrom (a0 * k0), if passed it will
                         be used to generate the k-points card
        kpoints_grid: string to be putted in to the grid place
        return
        ------
        dict of strings:
            keys:
            CELL_PARAMETERS,
            ATOMIC_SPECIES,
            ATOMIC_POSITIONS,
            K_POINTS
        """
        qe_dict = {}
        logger.debug('kpoints grid %s', kpoints_grid)
        logger.debug('kpoints density %s', kpoints_density)

        tmp = 'CELL_PARAMETERS {}\n'.format(self.unit_of_length) + \
            '\n'.join(['{} {} {} '.format(*x) for x in self.lattice_vectors])
        qe_dict['CELL_PARAMETERS'] = Template(tmp)

        tmp = 'ATOMIC_SPECIES\n' + \
            '\n'.join(
                [f" {specie} {recover_property(specie, 'mass')} ${specie}_pseudo"
                 for specie in self.species]
            )
        qe_dict['ATOMIC_SPECIES'] = Template(tmp)

        tmp = 'ATOMIC_POSITIONS {}\n'.format(self.unit_of_length) + \
            '\n'.join(['{} {} {} {}'.format(self.species[specie], *pos) for
                       specie, pos in zip(self.species_indexes, self.positions)]
                      )
        qe_dict['ATOMIC_POSITIONS'] = Template(tmp)

        if kpoints_density and kpoints_grid:
            raise ValueError(
                'kpoints_density and kpoints_grid can not be provided together')

        if kpoints_density:
            tmp = 'K_POINTS automatic\n {} 1 1 1'
            n_kpoints = np.round(
                kpoints_density
                / np.linalg.norm(self.angstrom_lattice_vectors, axis=1)).astype(np.int)
            tmp = tmp.format(' '.join(n_kpoints.astype(str)))
            qe_dict['K_POINTS'] = Template(tmp)
        elif kpoints_grid:
            tmp = 'K_POINTS automatic\n {} 1 1 1'
            tmp = tmp.format(kpoints_grid)
            qe_dict['K_POINTS'] = Template(tmp)
        else:
            qe_dict['K_POINTS'] = Template('K_POINTS gamma\n')

        return qe_dict
