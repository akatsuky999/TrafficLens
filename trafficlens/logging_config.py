"""
Logging configuration for TrafficLens.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import PROJECT_ROOT


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with console and rotating file handlers."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "trafficlens.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers in interactive runs
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(console_handler)
        root.addHandler(file_handler)


