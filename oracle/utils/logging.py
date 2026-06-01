"""
Logging utilities for the ORACLE pipeline.

Provides:
- ``get_logger``    – returns a configured logger
- ``setup_logging`` – configures file + console handlers
- ``OracleLogger``  – context manager that logs step timing
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Default format
# ---------------------------------------------------------------------------

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Keep track of already-configured loggers to avoid duplicate handlers
_CONFIGURED_LOGGERS: set = set()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given name.

    The first call for a given *name* attaches a
    ``StreamHandler(sys.stdout)`` with the ORACLE default format.
    Subsequent calls return the same logger unchanged.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if name not in _CONFIGURED_LOGGERS:
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False

        _CONFIGURED_LOGGERS.add(name)

    return logger


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str,
    level: str = "INFO",
    filename: str = "oracle.log",
) -> None:
    """Configure both file and console (stdout) logging for ORACLE.

    After calling this function, the root logger (and therefore all child
    loggers that propagate) will write to:
    - ``stdout`` – messages at *level* and above
    - ``{log_dir}/{filename}`` – messages at *level* and above

    Parameters
    ----------
    log_dir:
        Directory where the log file will be written.  Created if it does
        not exist.
    level:
        Logging level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, or ``"CRITICAL"``.  Defaults to ``"INFO"``.
    filename:
        Log file name within *log_dir*.  Defaults to ``"oracle.log"``.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on repeated calls
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    log_path = log_dir / filename
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(
        "Logging initialised — level=%s, file=%s", level.upper(), log_path
    )


# ---------------------------------------------------------------------------
# OracleLogger context manager
# ---------------------------------------------------------------------------

class OracleLogger:
    """Context manager that logs the duration of a named pipeline step.

    Usage::

        with OracleLogger("GRN inference", logger=logger):
            grn = infer_grn(adata)

    On entry the start time is recorded and a ``"Started: {step}"`` message
    is logged at INFO level.  On exit a ``"Completed: {step} in X.XXs"``
    message is logged.  If an exception is raised, an ERROR message is logged
    and the exception is re-raised.

    Parameters
    ----------
    step_name:
        Human-readable name of the pipeline step.
    logger:
        Logger to write messages to.  If ``None``, a module-level logger
        named ``"oracle.step"`` is used.
    level:
        Logging level for timing messages (default ``logging.INFO``).
    """

    def __init__(
        self,
        step_name: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
    ) -> None:
        self.step_name = step_name
        self._logger = logger or get_logger("oracle.step")
        self._level = level
        self._start: float = 0.0

    def __enter__(self) -> "OracleLogger":
        self._start = time.perf_counter()
        self._logger.log(self._level, "Started: %s", self.step_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start
        if exc_type is not None:
            self._logger.error(
                "Failed: %s after %.3fs — %s: %s",
                self.step_name,
                elapsed,
                exc_type.__name__,
                exc_val,
            )
            return False  # Re-raise the exception
        self._logger.log(
            self._level,
            "Completed: %s in %.3fs",
            self.step_name,
            elapsed,
        )
        return False
