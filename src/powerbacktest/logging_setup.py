"""One-time logging configuration for the CLI.

Library modules only call ``logging.getLogger(__name__)``; handlers are attached here,
once, so repeated imports never duplicate log lines.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", file: str | Path | None = None) -> None:
    root = logging.getLogger("powerbacktest")
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(FORMAT, datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if file:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(file_handler)
    root.propagate = False
