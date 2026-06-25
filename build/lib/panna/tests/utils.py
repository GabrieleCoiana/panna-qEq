###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import configparser

ROOT_FOLDER = '/tmp/panna/'


def section_mocker(instance, section, options):
    """section mocker routine to override configparser values

    Parameters
    ----------
      instance: a configparser instance
      section: a section to mock
      options: the options within the section
    """
    # pylint: disable=protected-access
    instance._sections[section] = options
    instance._proxies[section] = configparser.SectionProxy(instance, section)
