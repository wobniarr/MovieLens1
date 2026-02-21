"""
Two-Tower (Dual Encoder) model for candidate generation.

Architecture:
- User Tower: embeddings for user_id, gender, age, occupation → MLP → user embedding
- Item Tower: embeddings for movie_id + genre multi-hot → MLP → item embedding
- Similarity: dot product between user and item embeddings

Trained with in-batch contrastive loss for recall-focused retrieval.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MLP(nn.Module):
    """Multi-layer perceptron with batch normalization and dropout."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UserTower(nn.Module):
    """User tower: embeds user features and projects to shared embedding space."""

    def __init__(self, vocab_sizes: dict, config: dict):
        super().__init__()
        feat_cfg = config["features"]
        model_cfg = config["candidate_gen"]

        # Embedding layers
        self.user_id_emb = nn.Embedding(
            vocab_sizes["user_id"], feat_cfg["user_id_embedding_dim"], padding_idx=0
        )
        self.gender_emb = nn.Embedding(
            vocab_sizes["gender"], feat_cfg["gender_embedding_dim"], padding_idx=0
        )
        self.age_emb = nn.Embedding(
            vocab_sizes["age"], feat_cfg["age_embedding_dim"], padding_idx=0
        )
        self.occupation_emb = nn.Embedding(
            vocab_sizes["occupation"], feat_cfg["occupation_embedding_dim"], padding_idx=0
        )

        # Calculate total input dimension
        total_dim = (
            feat_cfg["user_id_embedding_dim"]
            + feat_cfg["gender_embedding_dim"]
            + feat_cfg["age_embedding_dim"]
            + feat_cfg["occupation_embedding_dim"]
        )

        # MLP to project to shared embedding space
        self.mlp = MLP(total_dim, model_cfg["user_hidden_dims"], model_cfg["dropout"])

    def forward(self, user_id, gender, age, occupation) -> torch.Tensor:
        """Compute user embedding.

        Returns:
            L2-normalized user embedding of shape (batch_size, embedding_dim).
        """
        x = torch.cat([
            self.user_id_emb(user_id),
            self.gender_emb(gender),
            self.age_emb(age),
            self.occupation_emb(occupation),
        ], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower: embeds item features and projects to shared embedding space."""

    def __init__(self, vocab_sizes: dict, config: dict):
        super().__init__()
        feat_cfg = config["features"]
        model_cfg = config["candidate_gen"]

        # Embedding layers
        self.movie_id_emb = nn.Embedding(
            vocab_sizes["movie_id"], feat_cfg["movie_id_embedding_dim"], padding_idx=0
        )
        self.genre_proj = nn.Linear(
            vocab_sizes["genres"], feat_cfg["genre_embedding_dim"]
        )

        # Calculate total input dimension
        total_dim = feat_cfg["movie_id_embedding_dim"] + feat_cfg["genre_embedding_dim"]

        # MLP to project to shared embedding space
        self.mlp = MLP(total_dim, model_cfg["item_hidden_dims"], model_cfg["dropout"])

    def forward(self, movie_id, genres) -> torch.Tensor:
        """Compute item embedding.

        Returns:
            L2-normalized item embedding of shape (batch_size, embedding_dim).
        """
        x = torch.cat([
            self.movie_id_emb(movie_id),
            self.genre_proj(genres),
        ], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-Tower model for candidate generation.

    Computes user and item embeddings in separate towers, then measures
    similarity via dot product. Trained with in-batch contrastive loss.
    """

    def __init__(self, vocab_sizes: dict, config: dict):
        """Initialize the Two-Tower model.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from FeatureEncoder.
            config: Full configuration dictionary.
        """
        super().__init__()
        self.user_tower = UserTower(vocab_sizes, config)
        self.item_tower = ItemTower(vocab_sizes, config)
        self.temperature = config["candidate_gen"]["temperature"]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute user and item embeddings.

        Args:
            batch: Dictionary with user and item feature tensors.

        Returns:
            Dictionary with 'user_emb', 'item_emb', and 'logits' tensors.
        """
        user_emb = self.user_tower(
            batch["user_id"],
            batch["gender"],
            batch["age"],
            batch["occupation"],
        )
        item_emb = self.item_tower(
            batch["movie_id"],
            batch["genres"],
        )

        # Compute similarity matrix for in-batch negatives
        # logits[i, j] = similarity between user_i and item_j
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature

        return {
            "user_emb": user_emb,
            "item_emb": item_emb,
            "logits": logits,
        }

    def get_user_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute only user embeddings (for inference)."""
        return self.user_tower(
            batch["user_id"],
            batch["gender"],
            batch["age"],
            batch["occupation"],
        )

    def get_item_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute only item embeddings (for inference/index building)."""
        return self.item_tower(
            batch["movie_id"],
            batch["genres"],
        )
