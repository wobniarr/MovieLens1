"""
Retrieval Metrics (for candidate generation):
- Recall @K: Fraction of relevant items retrieved

Ranking Metrics (for ranking model):
- NDCG @K: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- AUC: Area Under the ROC Curve (sanity check)
"""

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


class RetrievalMetrics:
    """Metrics for evaluating candidate generation (retrieval) quality.

    Focuses on recall — did the model retrieve the relevant items?
    Ranking quality within the candidate set doesn't matter because
    the ranking model re-orders them downstream.
    """

    @staticmethod
    def recall_at_k(
        predictions: np.ndarray,
        targets: np.ndarray,
        k: int,
    ) -> float:
        """Compute Recall @K.

        For single relevant item per query, this equals hit rate @K.

        Args:
            predictions: Array of shape (num_users, num_items) with scores.
            targets: Array of shape (num_users,) with true item indices.
            k: Number of top items to consider.

        Returns:
            Average recall@K across all users.
        """
        top_k_indices = np.argsort(-predictions, axis=1)[:, :k]
        hits = 0
        for i, target in enumerate(targets):
            if target in top_k_indices[i]:
                hits += 1
        return hits / len(targets)


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

