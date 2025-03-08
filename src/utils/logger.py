import logging
from logging import config
from typing import Optional, Dict

from utils.logger_configuration import LOG_CONFIG
import colorlog

from utils.logger_configuration import LOG_CONFIG
import colorlog


def setup_logger(name: str,
                 log_file_path: Optional[str] = None,
                 level: str = 'INFO',
                 log_config: dict = LOG_CONFIG,
                 filter_loggers: Optional[Dict[str, str]] = None) -> logging.Logger:

    assert level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], "Wrong logging level"


    log_config['root'].update({'level': level})
    # Update log file path in 'file' handler
    if log_file_path:
        log_config["handlers"]["file"]["filename"] = log_file_path

    # Create a color formatter
    color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(name)s:%(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
   )

    # Set config and create logger 'name'
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(name)

    # Add a color console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # Set specific level to other logger
    if filter_loggers is not None:
         for module_name, module_level in filter_loggers.items():
            logging.getLogger(module_name).setLevel(module_level)

    return logger
