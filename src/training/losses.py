"""
Loss functions for candidate generation and ranking models.

- ContrastiveLoss: In-batch negative contrastive loss for Two-Tower model.
  Treats the diagonal of the similarity matrix as positives and all
  off-diagonal entries as negatives (similar to InfoNCE / NT-Xent).

- RankingLoss: Binary cross-entropy with logits for the ranking model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """In-batch contrastive loss for the Two-Tower model.

    Given a batch of N user-item pairs, computes an NxN similarity matrix.
    The diagonal entries (i,i) are positives (user_i interacted with item_i).
    All off-diagonal entries are negatives (in-batch negatives).

    This is equivalent to N-way classification where each user must identify
    its true item among all items in the batch.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            logits: Similarity matrix of shape (batch_size, batch_size),
                    already temperature-scaled.

        Returns:
            Scalar loss value.
        """
        batch_size = logits.size(0)
        # Labels: diagonal indices (user_i should match item_i)
        labels = torch.arange(batch_size, device=logits.device)
        # Cross-entropy over rows (user→item) and columns (item→user)
        loss_u2i = F.cross_entropy(logits, labels)
        loss_i2u = F.cross_entropy(logits.T, labels)
        return (loss_u2i + loss_i2u) / 2


class RankingLoss(nn.Module):
    """Binary cross-entropy loss for the ranking model.

    Wraps BCEWithLogitsLoss for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ranking loss.

        Args:
            logits: Raw prediction scores of shape (batch_size,).
            labels: Binary labels of shape (batch_size,).

        Returns:
            Scalar loss value.
        """
        return self.bce(logits, labels)
