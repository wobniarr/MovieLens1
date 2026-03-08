"""
Evaluation metrics for retrieval (candidate generation) and ranking.

Retrieval Metrics (for candidate generation):
- Hit Rate @K: Did the true item appear in top-K?
- NDCG @K: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- Recall @K: Fraction of relevant items retrieved

Ranking Metrics (for ranking model):
- AUC: Area Under the ROC Curve
- Log Loss: Binary cross-entropy
"""

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


class RetrievalMetrics:
    """Metrics for evaluating candidate generation (retrieval) quality.

    Computes metrics based on how well the model retrieves relevant items
    from the full item catalog.
    """

    @staticmethod
    def hit_rate_at_k(predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Compute Hit Rate @K.

        Args:
            predictions: Array of shape (num_users, num_items) with scores.
            targets: Array of shape (num_users,) with true item indices.
            k: Number of top items to consider.

        Returns:
            Hit rate (fraction of users where true item is in top-K).
        """
        top_k_indices = np.argsort(-predictions, axis=1)[:, :k]
        hits = 0
        for i, target in enumerate(targets):
            if target in top_k_indices[i]:
                hits += 1
        return hits / len(targets)

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
    def recall_at_k(
        predictions: np.ndarray,
        targets: np.ndarray,
        k: int,
    ) -> float:
        """Compute Recall @K.

        For single relevant item per query, recall@k equals hit_rate@k.
        Provided separately for clarity in multi-relevant-item scenarios.

        Args:
            predictions: Array of shape (num_users, num_items) with scores.
            targets: Array of shape (num_users,) with true item indices.
            k: Number of top items to consider.

        Returns:
            Average recall@K across all users.
        """
        return RetrievalMetrics.hit_rate_at_k(predictions, targets, k)


class RankingMetrics:
    """Metrics for evaluating the ranking model quality."""

    @staticmethod
    def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute Area Under the ROC Curve.

        Args:
            labels: Binary ground truth labels.
            scores: Predicted scores/probabilities.

        Returns:
            AUC score. Returns 0.5 if only one class exists.
        """
        if len(np.unique(labels)) < 2:
            return 0.5
        return roc_auc_score(labels, scores)

    @staticmethod
    def compute_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute binary cross-entropy (log loss).

        Args:
            labels: Binary ground truth labels.
            scores: Predicted probabilities (after sigmoid).

        Returns:
            Log loss value.
        """
        # Clip to avoid log(0)
        scores = np.clip(scores, 1e-7, 1 - 1e-7)
        return log_loss(labels, scores)

    @staticmethod
    def compute_all(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Compute all ranking metrics.

        Args:
            labels: Binary ground truth labels.
            scores: Predicted scores (raw logits or probabilities).

        Returns:
            Dictionary with AUC and LogLoss values.
        """
        # Convert logits to probabilities for log_loss
        probs = 1 / (1 + np.exp(-np.clip(scores, -10, 10)))  # sigmoid

        return {
            "AUC": RankingMetrics.compute_auc(labels, scores),
            "LogLoss": RankingMetrics.compute_logloss(labels, probs),
        }
