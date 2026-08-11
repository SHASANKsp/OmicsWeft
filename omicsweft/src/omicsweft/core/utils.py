"""Cross-cutting utilities: seeding and logging."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if present) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional; only if torch is installed
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


def get_logger(name: str = "omicsweft", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
