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
        ground_truth: Dict[int, set],
        ks: List[int],
        chunk_size: int = 1024,
        train_seen: Dict[int, set] = None,
    ) -> Dict[str, float]:
        """Compute Recall@K using chunked similarity to bound peak memory.

        Processes users in chunks of `chunk_size`, keeping peak memory at
        O(chunk_size * num_items) instead of O(num_users * num_items).

        Recall@K per user = |retrieved_top_k ∩ relevant| / |relevant|,
        averaged over all users with at least one relevant item.

        Args:
            user_embs: Tensor of shape (num_users, D) with unique user embeddings.
            item_embs: Tensor of shape (num_items, D) with unique item embeddings.
            ground_truth: Maps user index -> set of positive item indices.
            ks: List of K values for Recall@K.
            chunk_size: Number of users to process per chunk.
            train_seen: Optional. Maps user index -> set of item indices the user
                interacted with during training. These items are masked to -inf
                before topk so they cannot occupy top-K slots.

        Returns:
            Dictionary mapping metric names (e.g. "Recall@5") to values.
        """
        max_k = max(ks)
        n_users = user_embs.shape[0]
        recall_sums = {k: 0.0 for k in ks}

        for start in range(0, n_users, chunk_size):
            end = min(start + chunk_size, n_users)
            # (chunk, D) @ (D, num_items) -> (chunk, num_items)
            chunk_scores = torch.mm(user_embs[start:end], item_embs.T)

            # Mask train-seen items so they can't appear in top-K
            if train_seen:
                for i, user_idx in enumerate(range(start, end)):
                    seen = train_seen.get(user_idx, set())
                    if seen:
                        seen_indices = list(seen)
                        chunk_scores[i, seen_indices] = float("-inf")

            _, top_indices = torch.topk(chunk_scores, max_k, dim=1)
            top_indices = top_indices.numpy()

            for i, user_idx in enumerate(range(start, end)):
                true_items = ground_truth.get(user_idx, set())
                if not true_items:
                    continue
                for k in ks:
                    found = len(true_items & set(top_indices[i, :k]))
                    recall_sums[k] += found / len(true_items)

        n_eval_users = sum(1 for v in ground_truth.values() if v)
        if n_eval_users == 0:
            return {f"Recall@{k}": 0.0 for k in ks}

        return {f"Recall@{k}": recall_sums[k] / n_eval_users for k in ks}


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

