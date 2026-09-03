"""Structured logging shared by pipeline, API and dashboard."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/app.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 3,
) -> None:
    """Attach console + rotating file handlers to the root logger once."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(level.upper())
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
