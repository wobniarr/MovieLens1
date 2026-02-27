"""
PyTorch Dataset classes for Candidate Generation and Ranking.

- CandidateGenDataset: Feeds the Two-Tower model. Uses in-batch negatives
  at training time, so only returns (user_features, item_features) pairs.
- RankingDataset: Feeds the Ranking model with hybrid features (explicit
  rating + implicit signal) and pre-sampled negatives.
"""
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features import FeatureEncoder


class CandidateGenDataset(Dataset):
    """Dataset for the Two-Tower candidate generation model.

    Returns user and item feature tensors for positive (observed) interactions.
    Negatives are handled via in-batch negative sampling during training.
    """

    def __init__(self, df: pd.DataFrame, encoder: FeatureEncoder):
        """Initialize the dataset.

        Args:
            df: DataFrame of positive interactions (all have implicit=1).
            encoder: Fitted FeatureEncoder for feature transformation.
        """
        self.encoder = encoder
        self.df = df.reset_index(drop=True)

        # Pre-encode all features for performance
        self._user_ids = np.array([encoder.encode_user_id(uid) for uid in self.df["user_id"]])
        self._movie_ids = np.array([encoder.encode_movie_id(mid) for mid in self.df["movie_id"]])
        self._genders = np.array([encoder.encode_gender(g) for g in self.df["gender"]])
        self._ages = np.array([encoder.encode_age(a) for a in self.df["age"]])
        self._occupations = np.array([encoder.encode_occupation(o) for o in self.df["occupation"]])
        self._genres = np.stack([encoder.encode_genres(g) for g in self.df["genres"]])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single user-item interaction.

        Returns:
            Dictionary with:
              - user_id: (1,) user index
              - gender: (1,) gender index
              - age: (1,) age index
              - occupation: (1,) occupation index
              - movie_id: (1,) movie index
              - genres: (num_genres,) multi-hot genre vector
        """
        return {
            "user_id": torch.tensor(self._user_ids[idx], dtype=torch.long),
            "gender": torch.tensor(self._genders[idx], dtype=torch.long),
            "age": torch.tensor(self._ages[idx], dtype=torch.long),
            "occupation": torch.tensor(self._occupations[idx], dtype=torch.long),
            "movie_id": torch.tensor(self._movie_ids[idx], dtype=torch.long),
            "genres": torch.tensor(self._genres[idx], dtype=torch.float),
        }


class RankingDataset(Dataset):
    """Dataset for the Ranking model with hybrid features.

    Returns user features, item features, explicit rating, implicit signal,
    and the binary label (positive/negative). Includes pre-sampled negatives.
    """

    def __init__(self, df: pd.DataFrame, encoder: FeatureEncoder):
        """Initialize the dataset.

        Args:
            df: DataFrame with both positive interactions and pre-sampled
                negatives. Must contain 'rating', 'implicit', and 'label' cols.
            encoder: Fitted FeatureEncoder for feature transformation.
        """
        self.encoder = encoder
        self.df = df.reset_index(drop=True)

        # Pre-encode features
        self._user_ids = np.array([encoder.encode_user_id(uid) for uid in self.df["user_id"]])
        self._movie_ids = np.array([encoder.encode_movie_id(mid) for mid in self.df["movie_id"]])
        self._genders = np.array([encoder.encode_gender(g) for g in self.df["gender"]])
        self._ages = np.array([encoder.encode_age(a) for a in self.df["age"]])
        self._occupations = np.array([encoder.encode_occupation(o) for o in self.df["occupation"]])
        self._genres = np.stack([encoder.encode_genres(g) for g in self.df["genres"]])

        # Hybrid features
        self._ratings = self.df["rating"].values.astype(np.float32)
        self._implicit = self.df["implicit"].values.astype(np.float32)
        self._labels = self.df["label"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with hybrid features.

        Returns:
            Dictionary with user/item features plus:
              - rating: (1,) explicit rating value (0 for negatives)
              - implicit: (1,) binary implicit signal (1=observed, 0=unobserved)
              - label: (1,) binary label (1=positive, 0=negative)
        """
        return {
            # User features
            "user_id": torch.tensor(self._user_ids[idx], dtype=torch.long),
            "gender": torch.tensor(self._genders[idx], dtype=torch.long),
            "age": torch.tensor(self._ages[idx], dtype=torch.long),
            "occupation": torch.tensor(self._occupations[idx], dtype=torch.long),
            # Item features
            "movie_id": torch.tensor(self._movie_ids[idx], dtype=torch.long),
            "genres": torch.tensor(self._genres[idx], dtype=torch.float),
            # Hybrid feedback features
            "rating": torch.tensor(self._ratings[idx], dtype=torch.float),
            "implicit": torch.tensor(self._implicit[idx], dtype=torch.float),
            # Target
            "label": torch.tensor(self._labels[idx], dtype=torch.float),
        }
