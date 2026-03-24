"""
Training script for the Two-Tower candidate generation model.

Usage:
    python scripts/train_candidate_gen.py [--config configs/default.yaml]
"""
import sys
import argparse

import numpy as np
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


def create_retrieval_eval_fn(ks, train_df, chunk_size=1024):
    """Create an evaluation function for retrieval metrics.

    Deduplicates user and item embeddings so the similarity matrix is
    (num_unique_users x num_unique_items) instead of (N_interactions x N_interactions).
    Excludes train-seen items from top-K to measure discovery of new relevant items.

    Args:
        ks: List of K values for Recall@K.
        train_df: Training DataFrame with user_id and movie_id columns.
        chunk_size: Number of users to score per chunk (controls peak RAM).
    """
    # Pre-build raw train interactions: user_id -> set of movie_ids
    train_interactions = {}
    for uid, mid in zip(train_df["user_id"], train_df["movie_id"]):
        train_interactions.setdefault(uid, set()).add(mid)

    @torch.no_grad()
    def eval_fn(model, val_loader, device):
        model.eval()
        all_user_ids = []
        all_movie_ids = []
        all_user_embs = []
        all_item_embs = []

        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch)
            all_user_ids.append(batch["user_id"].cpu())
            all_movie_ids.append(batch["movie_id"].cpu())
            all_user_embs.append(output["user_emb"].cpu())
            all_item_embs.append(output["item_emb"].cpu())

        user_ids = torch.cat(all_user_ids, dim=0).numpy()
        movie_ids = torch.cat(all_movie_ids, dim=0).numpy()
        user_embs = torch.cat(all_user_embs, dim=0)
        item_embs = torch.cat(all_item_embs, dim=0)

        # Deduplicate: keep first occurrence of each unique user/item
        unique_user_ids, user_first_idx = np.unique(user_ids, return_index=True)
        unique_movie_ids, item_first_idx = np.unique(movie_ids, return_index=True)
        unique_user_embs = user_embs[user_first_idx]
        unique_item_embs = item_embs[item_first_idx]

        # Build ground truth: user_idx -> set of positive item_indices
        movie_id_to_idx = {mid: idx for idx, mid in enumerate(unique_movie_ids)}
        ground_truth = {}
        for uid, mid in zip(user_ids, movie_ids):
            u_idx = np.searchsorted(unique_user_ids, uid)
            m_idx = movie_id_to_idx[mid]
            ground_truth.setdefault(u_idx, set()).add(m_idx)

        # Build train_seen: user_idx -> set of item_indices seen in training
        train_seen = {}
        for u_idx, raw_uid in enumerate(unique_user_ids):
            seen_movie_ids = train_interactions.get(raw_uid, set())
            seen_item_indices = set()
            for mid in seen_movie_ids:
                if mid in movie_id_to_idx:
                    seen_item_indices.add(movie_id_to_idx[mid])
            if seen_item_indices:
                train_seen[u_idx] = seen_item_indices

        return RetrievalMetrics.chunked_recall_at_k(
            unique_user_embs, unique_item_embs, ground_truth, ks,
            chunk_size, train_seen
        )

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
    eval_fn = create_retrieval_eval_fn(
        config["evaluation"]["ks"],
        train_df,
        chunk_size=config["evaluation"]["eval_chunk_size"],
    )

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
