"""
Logger for SYNAPSE web app.
Logs to console only — no file writing needed for a web server.
"""

import logging
import sys


def setup_logger(name: str = 'synapse', level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger


def get_logger(name: str = 'synapse') -> logging.Logger:
    return logging.getLogger(name)