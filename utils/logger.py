"""Centralized structured logging for the platform."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config.settings import LOG_DIR, settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, creating it once per name.

    Logs are written both to stdout (for local dev / Streamlit) and to
    a rotating file under ``logs/`` for later inspection or the Agent
    Activity dashboard page.
    """
    if name in _configured_loggers:
        return _configured_loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = RotatingFileHandler(
            LOG_DIR / "platform.log", maxBytes=2_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _configured_loggers[name] = logger
    return logger
