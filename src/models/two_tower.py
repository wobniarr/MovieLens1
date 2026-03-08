"""
Two-Tower (Dual Encoder) model for candidate generation.

Architecture:
- User Tower: embeddings for user_id, gender, age, occupation -> MLP -> projection -> user embedding
- Item Tower: embeddings for movie_id + genre multi-hot -> MLP -> projection -> item embedding
- Similarity: dot product between user and item embeddings

Trained with in-batch contrastive loss for recall-focused retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MLP(nn.Module):
    """MLP with batch normalization and dropout."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UserTower(nn.Module):
    """User tower: embeds user features and projects to shared embedding space."""

    def __init__(self, vocab_sizes: dict, config: dict):
        """Initialize the user tower.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from FeatureEncoder.
            config: Full configuration dictionary.
        """
        super().__init__()

        # Embedding layers
        self.user_id_emb = nn.Embedding(
            vocab_sizes["user_id"],
            config["features"]["user_id_embedding_dim"],
            padding_idx=0,
        )
        self.gender_emb = nn.Embedding(
            vocab_sizes["gender"],
            config["features"]["gender_embedding_dim"],
            padding_idx=0,
        )
        self.age_emb = nn.Embedding(
            vocab_sizes["age"], config["features"]["age_embedding_dim"], padding_idx=0
        )
        self.occupation_emb = nn.Embedding(
            vocab_sizes["occupation"],
            config["features"]["occupation_embedding_dim"],
            padding_idx=0,
        )

        # Calculate total input dimension
        total_dim = (
            config["features"]["user_id_embedding_dim"]
            + config["features"]["gender_embedding_dim"]
            + config["features"]["age_embedding_dim"]
            + config["features"]["occupation_embedding_dim"]
        )

        # MLP to project to shared embedding space
        self.mlp = MLP(
            total_dim,
            config["candidate_gen"]["user_hidden_dims"],
            config["candidate_gen"]["dropout"],
        )
        self.projection = nn.Linear(
            config["candidate_gen"]["user_hidden_dims"][-1],
            config["candidate_gen"]["embedding_dim"],
        )

    def forward(self, user_id, gender, age, occupation) -> torch.Tensor:
        """Compute user embedding.

        Args:
            user_id: User ID tensor of shape (batch_size,).
            gender: Gender tensor of shape (batch_size,).
            age: Age tensor of shape (batch_size,).
            occupation: Occupation tensor of shape (batch_size,).

        Returns:
            L2-normalized user embedding of shape (batch_size, embedding_dim).
        """
        x = torch.cat(
            [
                self.user_id_emb(user_id),
                self.gender_emb(gender),
                self.age_emb(age),
                self.occupation_emb(occupation),
            ],
            dim=-1,
        )
        x = self.mlp(x)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower: embeds item features and projects to shared embedding space."""

    def __init__(self, vocab_sizes: dict, config: dict):
        """Initialize the item tower.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from FeatureEncoder.
            config: Full configuration dictionary.
        """
        super().__init__()

        # Embedding layers
        self.movie_id_emb = nn.Embedding(
            vocab_sizes["movie_id"],
            config["features"]["movie_id_embedding_dim"],
            padding_idx=0,
        )
        self.genre_proj = nn.Linear(
            vocab_sizes["genres"], config["features"]["genre_embedding_dim"]
        )

        # Calculate total input dimension
        total_dim = (
            config["features"]["movie_id_embedding_dim"]
            + config["features"]["genre_embedding_dim"]
        )

        # MLP to project to shared embedding space
        self.mlp = MLP(
            total_dim,
            config["candidate_gen"]["item_hidden_dims"],
            config["candidate_gen"]["dropout"],
        )
        self.projection = nn.Linear(
            config["candidate_gen"]["item_hidden_dims"][-1],
            config["candidate_gen"]["embedding_dim"],
        )

    def forward(self, movie_id, genres) -> torch.Tensor:
        """Compute item embedding.

        Args:
            movie_id: Movie ID tensor of shape (batch_size,).
            genres: Multi-hot genre tensor of shape (batch_size, num_genres).

        Returns:
            L2-normalized item embedding of shape (batch_size, embedding_dim).
        """
        x = torch.cat(
            [
                self.movie_id_emb(movie_id),
                self.genre_proj(genres),
            ],
            dim=-1,
        )
        x = self.mlp(x)
        x = self.projection(x)
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
