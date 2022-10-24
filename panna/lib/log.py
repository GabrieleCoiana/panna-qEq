""" Helper file to handle centralized logging
"""

import logging
import logging.config

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'filters': {
        'require_debug_false': {
            '()': 'lib.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'lib.log.RequireDebugTrue',
        },
    },
    'formatters': {
        'formatter_1': {
            '()': 'logging.Formatter',
            'format': '{levelname} - {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'formatter_1'
        },
        'logfile': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'formatter_1'
        },
    },
    'loggers': {
        'panna': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'panna.lib': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'panna.tools': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'panna.network': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    }
}


splash_screen_record = logging.makeLogRecord({
    'level': logging.INFO,
    'levelno': 20,
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


def _splash_screen(emitter):
    # Splash screen
    record = logging.makeLogRecord({
        'level':
        logging.INFO,
        'levelname':
        'INFO',
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
    emitter(record)

def init_logging():
    """Logger init, to be called in mains
    """
    logging.config.dictConfig(DEFAULT_LOGGING)
    main_logger = logging.getLogger('panna')

    # creating a record is not very smart but comes in handy
    # if you add handler to a logger in a second time and want to output
    # it only on a given handler
    for handler in main_logger.handlers:
        _splash_screen(handler.emit)

class RequireDebugFalse(logging.Filter):
    def filter(self, record):
        return False


class RequireDebugTrue(logging.Filter):
    def filter(self, record):
        return True
