"""
Two-Tower (Dual Encoder) model for candidate generation.

Architecture:
- User Tower: embeddings for user_id, gender, age, occupation -> MLP -> projection -> user embedding
- Item Tower: embeddings for movie_id + genre multi-hot -> MLP -> projection -> item embedding
- Similarity: dot product between user and item embeddings

Trained with in-batch softmax contrastive loss for recall-focused retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MLP(nn.Module):
    """MLP with layer normalization and dropout."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UserTower(nn.Module):
    """User tower: embeds user features + watch history and projects to shared embedding space."""

    def __init__(self, vocab_sizes: dict, config: dict, shared_movie_emb: nn.Embedding):
        """Initialize the user tower.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from FeatureEncoder.
            config: Full configuration dictionary.
            shared_movie_emb: Shared movie_id embedding table (same as item tower).
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

        # Shared movie embedding for watch history
        self.history_emb = shared_movie_emb
        history_emb_dim = config["features"]["movie_id_embedding_dim"]

        # Calculate total input dimension (user features + history)
        total_dim = (
            config["features"]["user_id_embedding_dim"]
            + config["features"]["gender_embedding_dim"]
            + config["features"]["age_embedding_dim"]
            + config["features"]["occupation_embedding_dim"]
            + history_emb_dim  # mean-pooled history vector
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

    def forward(self, user_id, gender, age, occupation, watch_history) -> torch.Tensor:
        """Compute user embedding.

        Args:
            user_id: User ID tensor of shape (batch_size,).
            gender: Gender tensor of shape (batch_size,).
            age: Age tensor of shape (batch_size,).
            occupation: Occupation tensor of shape (batch_size,).
            watch_history: Encoded movie IDs of shape (batch_size, max_history_len).
                           Padded with 0s for users with short/no history.

        Returns:
            L2-normalized user embedding of shape (batch_size, embedding_dim).
        """
        # Embed watch history and mean-pool (masking padding)
        hist_emb = self.history_emb(watch_history)  # (B, L, D)
        hist_mask = (watch_history != 0).unsqueeze(-1).float()  # (B, L, 1)
        hist_sum = (hist_emb * hist_mask).sum(dim=1)  # (B, D)
        hist_count = hist_mask.sum(dim=1).clamp(min=1)  # (B, 1) avoid div-by-zero
        hist_pooled = hist_sum / hist_count  # (B, D)

        x = torch.cat(
            [
                self.user_id_emb(user_id),
                self.gender_emb(gender),
                self.age_emb(age),
                self.occupation_emb(occupation),
                hist_pooled,
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
    similarity via dot product. Trained with in-batch softmax contrastive loss.

    The movie_id embedding table is shared between the item tower (for the
    target movie) and the user tower (for watch history mean-pooling).
    """

    def __init__(self, vocab_sizes: dict, config: dict):
        """Initialize the Two-Tower model.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from FeatureEncoder.
            config: Full configuration dictionary.
        """
        super().__init__()

        # Shared movie embedding: used by item tower and user tower history encoder
        self.shared_movie_emb = nn.Embedding(
            vocab_sizes["movie_id"],
            config["features"]["movie_id_embedding_dim"],
            padding_idx=0,
        )

        self.user_tower = UserTower(vocab_sizes, config, self.shared_movie_emb)
        self.item_tower = ItemTower(vocab_sizes, config)
        # Point item tower's movie_id_emb to the shared table
        self.item_tower.movie_id_emb = self.shared_movie_emb

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
            batch["watch_history"],
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
            batch["watch_history"],
        )

    def get_item_embedding(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute only item embeddings (for inference/index building)."""
        return self.item_tower(
            batch["movie_id"],
            batch["genres"],
        )
