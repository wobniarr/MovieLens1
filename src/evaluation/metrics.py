"""
Retrieval Metrics (for candidate generation):
- Recall @K: Fraction of relevant items retrieved

Ranking Metrics (for ranking model):
- NDCG @K: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- AUC: Area Under the ROC Curve (sanity check)
"""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class RetrievalMetrics:
    """Metrics for evaluating candidate generation (retrieval) quality.

    Focuses on recall — did the model retrieve the relevant items?
    Ranking quality within the candidate set doesn't matter because
    the ranking model re-orders them downstream.
    """

    @staticmethod
    def chunked_recall_at_k(
        user_embs: torch.Tensor,
        item_embs: torch.Tensor,
        ks: List[int],
        chunk_size: int = 1024,
    ) -> Dict[str, float]:
        """Compute Recall@K using chunked similarity to bound peak memory.

        Instead of materializing the full N*N score matrix, processes users
        in chunks of `chunk_size`, keeping memory at O(chunk_size * N).

        Each user's true positive is the item at the same index (i.e., the
        diagonal of the similarity matrix).

        Args:
            user_embs: Tensor of shape (N, D) with user embeddings.
            item_embs: Tensor of shape (N, D) with item embeddings.
            ks: List of K values for Recall@K.
            chunk_size: Number of users to process per chunk.

        Returns:
            Dictionary mapping metric names (e.g. "Recall@5") to values.
        """
        max_k = max(ks)
        n_users = user_embs.shape[0]
        hits = {k: 0 for k in ks}

        for start in range(0, n_users, chunk_size):
            end = min(start + chunk_size, n_users)
            # (chunk, D) @ (D, N) -> (chunk, N)
            chunk_scores = torch.mm(user_embs[start:end], item_embs.T)
            _, top_indices = torch.topk(chunk_scores, max_k, dim=1)
            top_indices = top_indices.numpy()

            # Targets for this chunk are indices [start, start+1, ..., end-1]
            for i, target in enumerate(range(start, end)):
                for k in ks:
                    if target in top_indices[i, :k]:
                        hits[k] += 1

        return {f"Recall@{k}": hits[k] / n_users for k in ks}


class RankingMetrics:
    """Metrics for evaluating the ranking model quality.

    Primary metrics: NDCG@K and MRR measure top-of-list quality.
    Secondary metric: AUC as a sanity check for discriminative ability.
    """

    @staticmethod
    def ndcg_at_k(predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Compute NDCG @K (single relevant item per user).

        Args:
            predictions: Array of shape (num_users, num_items) with scores.
            targets: Array of shape (num_users,) with true item indices.
            k: Number of top items to consider.

        Returns:
            Average NDCG@K across all users.
        """
        top_k_indices = np.argsort(-predictions, axis=1)[:, :k]
        ndcg_scores = []
        for i, target in enumerate(targets):
            if target in top_k_indices[i]:
                rank = np.where(top_k_indices[i] == target)[0][0]
                ndcg_scores.append(
                    1.0 / np.log2(rank + 2)
                )  # +2 because rank is 0-indexed
            else:
                ndcg_scores.append(0.0)
        return np.mean(ndcg_scores)

    @staticmethod
    def mrr(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Mean Reciprocal Rank.

        Args:
            predictions: Array of shape (num_users, num_items) with scores.
            targets: Array of shape (num_users,) with true item indices.

        Returns:
            Mean reciprocal rank across all users.
        """
        ranks = np.argsort(-predictions, axis=1)
        rr_scores = []
        for i, target in enumerate(targets):
            rank_positions = np.where(ranks[i] == target)[0]
            if len(rank_positions) > 0:
                rr_scores.append(1.0 / (rank_positions[0] + 1))
            else:
                rr_scores.append(0.0)
        return np.mean(rr_scores)

    @staticmethod
    def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute Area Under the ROC Curve.

        Used as a sanity check — confirms the model can distinguish
        positives from negatives. Not a primary ranking metric.

        Args:
            labels: Binary ground truth labels.
            scores: Predicted scores/probabilities.

        Returns:
            AUC score. Returns 0.5 if only one class exists.
        """
        if len(np.unique(labels)) < 2:
            return 0.5
        return roc_auc_score(labels, scores)

