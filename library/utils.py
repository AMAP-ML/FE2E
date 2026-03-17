import logging
import sys


def setup_logging(args=None, log_level=None, reset=False):
    if logging.root.handlers:
        if reset:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    if log_level is None and args is not None:
        log_level = getattr(args, "console_log_level", None)
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, str(log_level).upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)


setup_logging()
