###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import logging
import os

import pandas as pd

from collections import namedtuple

logger = logging.getLogger(__name__)


class ModelFile(namedtuple('ModelFile', ['folder', 'epoch', 'step'])):
    """Utility tuple."""
    __slots__ = ()

    @property
    def file_name(self):
        return f'{self.folder}/epoch_{self.epoch}_step_{self.step}'

    def __eq__(self, o):
        return (self.epoch, self.step) == o


def recover_models_files(train_dir):
    def split(x):
        _, e, _, s = x.split('_')
        return int(e), int(s)

    values = [split(x[:-6]) for x in os.listdir(f'{train_dir}/') if x.endswith('index')]
    # Very lazy resorting
    values = pd.DataFrame(values, columns=['epoch', 'step'])
    values.sort_values(['epoch', 'step'], inplace=True)
    return [ModelFile(train_dir, x, y) for x, y in values.values]
