""" Helper file to handle centralized logging
"""

import logging
import logging.config

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'formatter_1': {
            '()': 'logging.Formatter',
            'format': '{levelname} - {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'formatter_1'
        },
        'console_err': {
            'level': 'ERROR',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
            'formatter': 'formatter_1'
        },
        'tf_logfile': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'tf.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 3
        },
    },
    'loggers': {
        # set tensorflow logger, only errors in console
        # all is piped in the logfile
        'tensorflow': {
            'handlers': ['console_err', 'tf_logfile'],
            'level': 'INFO',
            'propagate': False,
        },
        # main panna logger :)
        'panna': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

splash_screen_record = logging.makeLogRecord({
    'level': logging.INFO,
    'levelno': logging.INFO,
    'levelname': 'INFO',
    'msg':
    '\n'
    '    ____   _    _   _ _   _    _           \n'
    '   |  _ \\ / \\  | \\ | | \\ | |  / \\     \n'
    '   | |_) / _ \\ |  \\| |  \\| | / _ \\     \n'
    '   |  __/ ___ \\| |\\  | |\\  |/ ___ \\    \n'
    '   |_| /_/   \\_\\_| \\_|_| \\_/_/   \\_\\ \n'
    '\n'
    ' Properties from Artificial Neural Network Architectures'
    '\n'
})


def emit_splash_screen(logger):
    logger.handle(splash_screen_record)


def init_logging(config_file=None):
    """Logger init, to be called in mains
    """
    if config_file:
        raise ValueError('Not implemented')
    else:
        logging.config.dictConfig(DEFAULT_LOGGING)
