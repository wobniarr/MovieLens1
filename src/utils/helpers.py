"""
Utility helpers for configuration, seeding, device selection, and logging.
"""
import os
import random
import logging
from pathlib import Path

import yaml
import numpy as np
import torch


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """
    Returns:
        Dictionary with configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available compute device.

    Args:
        preference: "auto" (pick best), "cuda", or "cpu".

    Returns:
        torch.device instance.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    return device


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger with console and optional file output.

    Args:
        name: Logger name (typically module name).
        log_dir: Directory for log files. If None, logs only to console.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return the Path.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
