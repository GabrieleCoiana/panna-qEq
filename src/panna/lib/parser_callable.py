###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################


def get_list_from_comma_sep_strings(value):
    """ get a list out of comma separated value
    """
    return [x.strip() for x in value.split(',')]


def get_list_floats_from_comma_sep_strings(value):
    """ get a list out of comma separated value
    """
    return [float(x) for x in value.split(',')]


def get_list_integers_from_comma_sep_strings(value):
    """ get a list out of comma separated value
    """
    return [int(float(x)) for x in value.split(',')]


def get_list_floats_from_column_sep_strings(value):
    """ get a list of float form column separated value
    """
    return [float(x) for x in value.split(':')]

def get_1or2_ints(value):
    """ get a single int or 2 ints separated by a dash
        used for parsing profiling range
    """
    ints = [int(i) for i in value.split('-')]
    if len(ints)==1:
        return ints[0]
    elif len(ints)==2:
        return tuple(ints)
    else:
        raise ValueError('Could not parse {}'.format(value))

def get_network_architecture(value):
    """ parse the architecture format

    Args:
        value: string like
        layer_size:layer2_size...

    Return:
        list of size per layer
    """
    return [int(x.strip()) for x in value.split(':')]


def get_network_trainable(value):
    """ parse trainable list

    Args:
        value: string like
        1:0:1

    Return:
        list of trainable flag per layer
    """
    true_value = ['1', 'yes', 'true', 'on']
    false_value = ['0', 'no', 'false', 'off']

    def _convert_to_boolean(value):
        if value in true_value:
            return True
        elif value in false_value:
            return False
        else:
            raise ValueError('Not a boolean: {}'.format(value))

    parsed = [_convert_to_boolean(x.lower()) for x in value.split(':')]
    return parsed


def get_network_act(value):
    """ parse activation list

    Args:
        value: string like
        g:gauss:rbf:l

    Return:
        list of activations per layer
    """
    lin = ['l', 'L', 'linear', 'LINEAR', '0']
    gauss = ['g', 'G', 'gauss', 'gaussian', '1']
    rbf = ['rbf', 'RBF', '2']
    relu = ['relu', 'rl', 'RELU', 'RL', 'ReLU', '3']
    tanh = ['tanh', 'TANH', 'th', '4']
    avail_acts = [lin, gauss, rbf, relu, tanh]

    def _convert_act2int(value):
        for act in avail_acts:
            if value in act:
                return avail_acts.index(act)
            # should put an exception handler

    parsed = [_convert_act2int(x) for x in value.split(':')]
    return parsed


def get_network_w_dist(value):
    """ parse activation list

    Args:
        value: string like
        g:G:u:U:uniform

    Return:
        list of weights distribution per layer
    """
    gauss = ['gaussian', 'g', 'G', 'gauss']
    uniform = ['uniform', 'u', 'U']
    avail_distribution = [gauss, uniform]

    def _convert_act2int(value):
        for act in avail_distribution:
            if value in act:
                return act[0]
        raise ValueError('weight distribution not available')

    parsed = [_convert_act2int(x) for x in value.split(':')]
    return parsed


def get_network_behavior(value):
    """ parse behavior list with respect to default
    Args:
       value: stirng like
       keep:new:load

    Return:
       list of behavior flag per layer
    """

    return [x.strip().lower() for x in value.split(':')]


converters = {
    '_comma_list': get_list_from_comma_sep_strings,
    '_comma_list_floats': get_list_floats_from_comma_sep_strings,
    '_comma_list_integers': get_list_integers_from_comma_sep_strings,
    '_column_list_floats': get_list_floats_from_column_sep_strings,
    '_1or2_ints': get_1or2_ints,
    '_network_architecture': get_network_architecture,
    '_network_trainable': get_network_trainable,
    '_network_act': get_network_act,
    '_network_behavior': get_network_behavior,
    '_network_w_dist': get_network_w_dist
}
