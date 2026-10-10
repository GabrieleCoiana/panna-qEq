###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
"""Containg different key generators

hash_key_v2: slighly improved key
hash_key_v1: old key, not reimplemented
"""
import hashlib


def hash_key_v2(example_dict):
    """Return a unique key for every example
       IT IS NOT RANDOM

    Args:
        example_dict: dictionary representig a example
    Retun:
        an unique string for the simulator
    """

    # creation of the string
    strings = []
    lforce = True
    for atom in example_dict['atoms']:
        try:
            idx, kind, position, force = atom
        except:
            lforce = False
            idx, kind, position = atom

        s = str(idx)
        s += str(kind)
        position = [float(x).hex() for x in position]
        s += ''.join(map(str, position))
        if lforce:
            force = [float(x).hex() for x in force]
            s += ''.join(map(str, force))
        strings.append(s)

    if 'energy' in example_dict:
        strings.append(str(float(example_dict['energy'][0]).hex()))
        strings.append(example_dict['energy'][1])
    strings.append(str(example_dict['lattice_vectors']))
    strings.append(example_dict['atomic_position_unit'])
    strings.append(example_dict['unit_of_length'])
    strings.append(example_dict['name'])

    strings = ''.join(strings)
    key = hashlib.sha224(strings.encode('utf-8')).hexdigest()
    return key
