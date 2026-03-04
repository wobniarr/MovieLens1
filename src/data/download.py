import os
import zipfile
import urllib.request
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


def download_movielens(config: dict) -> Path:
    """Download and extract MovieLens-1M dataset.

    Args:
        config: Configuration dictionary with paths and data settings.

    Returns:
        Path to the extracted dataset directory.
    """
    raw_dir = Path(config["paths"]["raw_data_dir"])
    dataset_name = config["data"]["dataset_name"]
    dataset_url = config["data"]["dataset_url"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_dir / dataset_name
    zip_path = raw_dir / f"{dataset_name}.zip"

    # Check if already extracted
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        logger.info(f"Dataset already exists at {dataset_dir}. Skipping download.")
        return dataset_dir

    # Download
    logger.info(f"Downloading MovieLens-1M from {dataset_url}...")
    urllib.request.urlretrieve(dataset_url, str(zip_path))
    logger.info(f"Downloaded to {zip_path}")

    # Extract
    logger.info(f"Extracting to {raw_dir}...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(raw_dir))

    # Clean up zip
    os.remove(str(zip_path))
    logger.info(f"Dataset ready at {dataset_dir}")

    return dataset_dir
