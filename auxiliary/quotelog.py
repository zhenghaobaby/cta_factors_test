# -*- coding: utf-8 -*-
import os
import logging.config

def load_logconfig(prefix=''):
    LOG_DIR = 'logs'
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "log": {
                "level": "INFO",
                "class": "auxiliary.TimedRotatingFileHandlerSafe",
                "filename": LOG_DIR + "/{0}log.log".format(prefix + '_' if prefix != '' else prefix),
                "formatter": "verbose",
                "when": "D",
                "backupCount": 30,
                "encoding": 'utf-8'
            },
            "error": {
                "level": "ERROR",
                "class": "auxiliary.TimedRotatingFileHandlerSafe",
                "filename": LOG_DIR + "/{0}error.log".format(prefix + '_' if prefix != '' else prefix),
                "formatter": "verbose",
                "when": "D",
                "backupCount": 180,
                "encoding": 'utf-8'
            },
            "fatal": {
                "level": "CRITICAL",
                "class": "auxiliary.WeComSenderHandler",
                # "qq": 1035713299,
                # "type": "Private",
                "formatter": "verbose",
                # "encoding": 'utf-8'
            },
            "console":{
                "level":"INFO",
                "class":"logging.StreamHandler",
                "formatter": "verbose"
            }
        },
        "loggers": {
            "": {
                "level": "INFO",
                "handlers": ["log", "error", "console", "fatal"],
                "propagate": False
            },
        }
    }
    logging.config.dictConfig(LOG_CONFIG)


if __name__=='__main__':
    logging.info('test1')
    load_logconfig()
    logging.warning('test2')
