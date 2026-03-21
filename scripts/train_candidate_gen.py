"""
Training script for the Two-Tower candidate generation model.

Usage:
    python scripts/train_candidate_gen.py [--config configs/default.yaml]
"""
import sys
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import CandidateGenDataset
from src.features import FeatureEncoder
from src.models import TwoTowerModel
from src.training import Trainer, ContrastiveLoss
from src.evaluation import RetrievalMetrics
from src.utils import load_config, set_seed, get_device, get_logger

logger = get_logger(__name__)


def create_retrieval_eval_fn(ks):
    """Create an evaluation function for retrieval metrics."""

    @torch.no_grad()
    def eval_fn(model, val_loader, device):
        model.eval()
        all_user_embs = []
        all_item_embs = []

        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)
            all_user_embs.append(output["user_emb"].cpu())
            all_item_embs.append(output["item_emb"].cpu())

        user_embs = torch.cat(all_user_embs, dim=0).numpy()
        item_embs = torch.cat(all_item_embs, dim=0).numpy()

        # Each user's true positive is the item at the same index
        # Compute similarity matrix
        scores = user_embs @ item_embs.T
        targets = list(range(len(scores)))

        metrics = {}
        for k in ks:
            metrics[f"Recall@{k}"] = RetrievalMetrics.recall_at_k(scores, targets, k)

        return metrics

    return eval_fn


def train_candidate_gen(config_path: str = "configs/default.yaml"):
    """Train the Two-Tower candidate generation model."""
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    logger.info("=" * 60)
    logger.info("Training Two-Tower Candidate Generation Model")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading preprocessed data...")
    train_df = pd.read_parquet(config["paths"]["processed_data_dir"] + "/train_candidate_gen.parquet")
    val_df = pd.read_parquet(config["paths"]["processed_data_dir"] + "/val.parquet")

    # Initialize feature encoder
    encoder = FeatureEncoder(config)
    encoder.fit(train_df)
    encoder.save_vocabs()
    vocab_sizes = encoder.get_vocab_sizes()

    logger.info(f"Vocab sizes: {vocab_sizes}")

    # Create datasets & loaders
    train_dataset = CandidateGenDataset(train_df, encoder)
    val_dataset = CandidateGenDataset(val_df, encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["candidate_gen"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        drop_last=True,  # Important for in-batch negatives
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["candidate_gen"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )

    # Initialize model, loss, trainer
    model = TwoTowerModel(vocab_sizes, config)
    loss_fn = ContrastiveLoss()
    eval_fn = create_retrieval_eval_fn(config["evaluation"]["ks"])

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        stage="candidate_gen",
        device=device,
        eval_fn=eval_fn,
    )

    # Train
    history = trainer.train(train_loader, val_loader)

    logger.info("Candidate generation training complete!")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Tower model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    train_candidate_gen(args.config)
