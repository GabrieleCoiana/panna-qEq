###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import configparser
import logging

from lib.parser_callable import converters

logger = logging.getLogger(__name__)


def parameter_file_parser(filename):
    config = configparser.ConfigParser(converters=converters)
    logger.info('reading {}'.format(filename))
    config.read(filename)
    if not config.has_section('PARALLELIZATION'):
        config.add_section('PARALLELIZATION')
        logger.info('Parallelization is not tuned'
                    ' - using defaults may result in low efficiency')
    return config
