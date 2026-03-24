"""
Full evaluation script — runs metrics on the test set for both models.

Usage:
    python scripts/evaluate.py [--config configs/default.yaml]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import CandidateGenDataset, RankingDataset
from src.features import FeatureEncoder
from src.models import TwoTowerModel, RankingModel
from src.evaluation import RetrievalMetrics, RankingMetrics
from src.utils import load_config, set_seed, get_device, get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_candidate_gen(model, test_loader, device, ks, chunk_size=1024):
    """Evaluate the Two-Tower model on the test set.

    Deduplicates user and item embeddings so the similarity matrix is
    (num_unique_users x num_unique_items) instead of (N_interactions x N_interactions).
    """
    model.eval()
    all_user_ids = []
    all_movie_ids = []
    all_user_embs = []
    all_item_embs = []

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)
        all_user_ids.append(batch["user_id"].cpu())
        all_movie_ids.append(batch["movie_id"].cpu())
        # Move outputs to CPU memory (RAM) to save GPU VRAM
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

    return RetrievalMetrics.chunked_recall_at_k(
        unique_user_embs, unique_item_embs, ground_truth, ks, chunk_size
    )


@torch.no_grad()
def evaluate_ranking(model, test_loader, device, ks):
    """Evaluate the Ranking model on the test set.

    Computes NDCG@K and MRR (primary) plus AUC (sanity check).
    """
    model.eval()
    all_labels = []
    all_scores = []

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        all_labels.append(batch["label"].cpu().numpy())
        all_scores.append(logits.cpu().numpy())

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)

    # AUC as sanity check (works on flat labels/scores)
    results = {"AUC": RankingMetrics.compute_auc(labels, scores)}

    # NDCG@K and MRR require a per-user score matrix
    # Reshape into (num_users, items_per_user) if test set is structured that way
    # For now, compute on the flat predictions as a score matrix
    score_matrix = scores.reshape(1, -1)  # treat as single-user for flat eval
    target_indices = np.where(labels == 1)[0]

    for k in ks:
        results[f"NDCG@{k}"] = RankingMetrics.ndcg_at_k(
            score_matrix, target_indices, k
        )
    results["MRR"] = RankingMetrics.mrr(score_matrix, target_indices)

    return results


def evaluate(config_path: str = "configs/default.yaml"):
    """Run full evaluation on the test set."""
    config = load_config(config_path)
    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    logger.info("=" * 60)
    logger.info("Full Model Evaluation on Test Set")
    logger.info("=" * 60)

    # Load test data
    test_df = pd.read_parquet(config["paths"]["processed_data_dir"] + "/test.parquet")

    # Load encoder
    encoder = FeatureEncoder(config)
    encoder.load_vocabs()
    vocab_sizes = encoder.get_vocab_sizes()

    # === Evaluate Candidate Generation ===
    logger.info("\n--- Candidate Generation (Two-Tower) ---")
    cg_model = TwoTowerModel(vocab_sizes, config)
    cg_checkpoint = torch.load(
        Path(config["paths"]["checkpoints_dir"]) / "candidate_gen" / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    cg_model.load_state_dict(cg_checkpoint["model_state_dict"])
    cg_model = cg_model.to(device)

    cg_dataset = CandidateGenDataset(test_df, encoder)
    cg_loader = DataLoader(
        cg_dataset,
        batch_size=config["candidate_gen"]["batch_size"],
        shuffle=False,
        drop_last=True,
    )

    cg_metrics = evaluate_candidate_gen(
        cg_model, cg_loader, device,
        config["evaluation"]["ks"],
        chunk_size=config["evaluation"]["eval_chunk_size"],
    )

    logger.info("Candidate Generation Metrics:")
    for name, value in cg_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # === Evaluate Ranking ===
    logger.info("\n--- Ranking Model ---")
    rank_model = RankingModel(vocab_sizes, config)
    rank_checkpoint = torch.load(
        Path(config["paths"]["checkpoints_dir"]) / "ranking" / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    rank_model.load_state_dict(rank_checkpoint["model_state_dict"])
    rank_model = rank_model.to(device)

    rank_dataset = RankingDataset(test_df, encoder)
    rank_loader = DataLoader(
        rank_dataset,
        batch_size=config["ranking"]["batch_size"],
        shuffle=False,
    )

    rank_metrics = evaluate_ranking(
        rank_model, rank_loader, device, config["evaluation"]["ks"]
    )

    logger.info("Ranking Metrics:")
    for name, value in rank_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    logger.info("\nEvaluation complete!")
    return {"candidate_gen": cg_metrics, "ranking": rank_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    evaluate(args.config)
