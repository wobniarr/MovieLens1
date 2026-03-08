"""
Deep Ranking model with explicit rating features.

Takes user features, item features, explicit rating, and cross features
as input. Uses an MLP with batch normalization, dropout, and residual
connections to predict interaction probability.
"""
import torch
import torch.nn as nn
from typing import Dict


class ResidualBlock(nn.Module):
    """MLP block with residual connection, batch norm, and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual_proj(x)


class RankingModel(nn.Module):
    """Deep ranking model.

    Inputs:
    - User features: user_id, gender, age, occupation embeddings
    - Item features: movie_id, genres embeddings
    - Explicit rating (continuous, 0-5)
    - Cross features: user*genre interaction

    Output: logit score for binary classification
    """

    def __init__(self, vocab_sizes: dict, config: dict):
        """Initialize the ranking model.

        Args:
            vocab_sizes: Dictionary of vocabulary sizes from FeatureEncoder.
            config: Full configuration dictionary.
        """
        super().__init__()

        # Embedding layers
        self.user_id_emb = nn.Embedding(
            vocab_sizes["user_id"], config["features"]["user_id_embedding_dim"], padding_idx=0
        )
        self.gender_emb = nn.Embedding(
            vocab_sizes["gender"], config["features"]["gender_embedding_dim"], padding_idx=0
        )
        self.age_emb = nn.Embedding(
            vocab_sizes["age"], config["features"]["age_embedding_dim"], padding_idx=0
        )
        self.occupation_emb = nn.Embedding(
            vocab_sizes["occupation"], config["features"]["occupation_embedding_dim"], padding_idx=0
        )
        self.movie_id_emb = nn.Embedding(
            vocab_sizes["movie_id"], config["features"]["movie_id_embedding_dim"], padding_idx=0
        )
        self.genre_proj = nn.Linear(
            vocab_sizes["genres"], config["features"]["genre_embedding_dim"]
        )

        # Cross feature: user embedding * genre vector
        cross_dim = config["ranking"]["cross_feature_dim"]
        self.user_genre_proj = nn.Linear(
            config["features"]["user_id_embedding_dim"] * config["features"]["genre_embedding_dim"],
            cross_dim
        )

        # Compute total input dimension
        user_dim = (
            config["features"]["user_id_embedding_dim"]
            + config["features"]["gender_embedding_dim"]
            + config["features"]["age_embedding_dim"]
            + config["features"]["occupation_embedding_dim"]
        )
        item_dim = config["features"]["movie_id_embedding_dim"] + config["features"]["genre_embedding_dim"]
        hybrid_dim = 1  # explicit rating

        total_input_dim = user_dim + item_dim + hybrid_dim + cross_dim

        # MLP with residual connections
        hidden_dims = config["ranking"]["hidden_dims"]
        dropout = config["ranking"]["dropout"]

        layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Map MLP output to final prediction layer
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute ranking score.

        Args:
            batch: Dictionary with all feature tensors including
                   'rating' (explicit).

        Returns:
            Logit scores of shape (batch_size,).
        """
        # User features
        user_id_vec = self.user_id_emb(batch["user_id"])
        gender_vec = self.gender_emb(batch["gender"])
        age_vec = self.age_emb(batch["age"])
        occupation_vec = self.occupation_emb(batch["occupation"])

        # Item features
        movie_id_vec = self.movie_id_emb(batch["movie_id"])
        genre_vec = self.genre_proj(batch["genres"])

        # Cross features: user_id_emb * genre_emb (outer product -> flatten -> project)
        cross = torch.bmm(
            user_id_vec.unsqueeze(2),   # (B, user_dim, 1)
            genre_vec.unsqueeze(1),     # (B, 1, genre_dim)
        ).flatten(1)                    # (B, user_dim * genre_dim)
        cross = self.user_genre_proj(cross)

        # Explicit rating feature
        rating = batch["rating"].unsqueeze(-1)     # (B, 1) explicit rating

        # Concatenate all features
        x = torch.cat([
            user_id_vec, gender_vec, age_vec, occupation_vec,   # User
            movie_id_vec, genre_vec,                            # Item
            rating,                                             # Explicit feedback
            cross,                                              # Cross features
        ], dim=-1)

        # MLP with residual connections
        x = self.mlp(x)
        logits = self.output_layer(x)

        return logits.squeeze(-1)  # (B,)
