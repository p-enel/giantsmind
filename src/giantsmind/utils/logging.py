import logging
import sys
from pathlib import Path
from giantsmind.utils.local import get_local_data_path


def setup_logger(name=__name__):
    log_path = Path(get_local_data_path()) / "logs"
    log_path.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.WARNING)

    f_handler = logging.FileHandler(log_path / "all_logs.log")
    f_handler.setLevel(logging.DEBUG)

    e_handler = logging.FileHandler(log_path / "error_logs.log")
    e_handler.setLevel(logging.ERROR)

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    e_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.addHandler(e_handler)

    return logger


logger = setup_logger()
