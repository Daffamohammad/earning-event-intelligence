"""Logging setup."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "earnings_ml", level: str = "INFO") -> logging.Logger:
    """Return a named logger with a stderr handler.

    Each named logger gets its OWN handler (previously only the first-created logger did,
    which silently dropped every other module's messages because propagate=False).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
