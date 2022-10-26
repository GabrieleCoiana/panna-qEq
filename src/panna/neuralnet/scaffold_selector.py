import logging

from .panna_scaffold import PannaScaffold

logger = logging.getLogger(__name__)


def scaffold_selector(scaffold_type):
    logger.debug('scaffold type: %s', scaffold_type)
    if scaffold_type == 'PANNA':
        scaffold = PannaScaffold
    else:
        raise ValueError('Scaffold not found')
    return scaffold


def scaffold_getter(config):
    """Return a scaffold
    """
    if 'DATA_INFORMATION' in config:
        scaffold_type = config['DATA_INFORMATION'].get('scaffold_type', 'PANNA')
    else:
        scaffold_type = 'PANNA'

    logger.info('scaffold type: %s', scaffold_type)
    Scaffold = scaffold_selector(scaffold_type)
    scaffold = Scaffold(config)
    return scaffold
