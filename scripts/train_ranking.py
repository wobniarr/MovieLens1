"""
Training script for the Deep Ranking model with hybrid features.

Usage:
    python scripts/train_ranking.py [--config configs/default.yaml]
"""
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import RankingDataset
from src.features import FeatureEncoder
from src.models import RankingModel
from src.training import Trainer, RankingLoss
from src.evaluation import RankingMetrics
from src.utils import load_config, set_seed, get_device, get_logger

logger = get_logger(__name__)


def create_ranking_eval_fn():
    """Create an evaluation function for ranking metrics."""

    @torch.no_grad()
    def eval_fn(model, val_loader, device):
        model.eval()
        all_labels = []
        all_scores = []

        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            all_labels.append(batch["label"].cpu().numpy())
            all_scores.append(logits.cpu().numpy())

        labels = np.concatenate(all_labels)
        scores = np.concatenate(all_scores)

        return RankingMetrics.compute_all(labels, scores)

    return eval_fn


def train_ranking(config_path: str = "configs/default.yaml"):
    """Train the Deep Ranking model."""
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    logger.info("=" * 60)
    logger.info("Training Deep Ranking Model (Hybrid Features)")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading preprocessed data...")
    train_df = pd.read_parquet(config["paths"]["processed_data_dir"] + "/train_ranking.parquet")
    val_df = pd.read_parquet(config["paths"]["processed_data_dir"] + "/val.parquet")

    # Load feature encoder (already fitted during candidate gen training)
    encoder = FeatureEncoder(config)
    encoder.load_vocabs()
    vocab_sizes = encoder.get_vocab_sizes()

    logger.info(f"Vocab sizes: {vocab_sizes}")

    # Create datasets & loaders
    train_dataset = RankingDataset(train_df, encoder)
    val_dataset = RankingDataset(val_df, encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["ranking"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["ranking"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # Initialize model, loss, trainer
    model = RankingModel(vocab_sizes, config)
    loss_fn = RankingLoss()
    eval_fn = create_ranking_eval_fn()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        stage="ranking",
        device=device,
        eval_fn=eval_fn,
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    logger.info("Ranking model training complete!")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ranking model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    train_ranking(args.config)
