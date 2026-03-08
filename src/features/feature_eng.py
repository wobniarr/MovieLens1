"""
Feature engineering: encode categorical features and build vocabularies.

Handles user features (user_id, gender, age, occupation) and item features
(movie_id, genres multi-hot). Vocabularies are saved to disk for inference.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# MovieLens-1M genre list (fixed order) pre-hardcoded because list is static and small. Also, allows us to conveniently parse pipe-separated genres.
ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


class FeatureEncoder:
    """Encode raw features into model-ready indices and multi-hot vectors.

    Builds and manages vocabulary mappings for all categorical features.
    Vocabularies use 0 as a reserved padding/unknown index.
    """

    def __init__(self, config: dict):
        self.config = config
        self.vocab_dir = Path(config["paths"]["vocab_dir"])
        self.vocabs: Dict[str, dict] = {}
        self.genre_to_idx = {g: i for i, g in enumerate(ALL_GENRES)}

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        """Build vocabularies from the training data.

        Creates mappings: raw_value -> integer_index for each categorical
        feature. Index 0 is reserved for unknown/padding.

        Args:
            df: Training DataFrame with all feature columns.

        Returns:
            self (for method chaining).
        """
        categorical_cols = ["user_id", "movie_id", "gender", "age", "occupation"]

        for col in categorical_cols:
            unique_vals = sorted(
                df[col].dropna().unique()
            )  # Sort to ensure consistent ordering
            self.vocabs[col] = {str(v): i + 1 for i, v in enumerate(unique_vals)}

        logger.info("Built vocabularies:")
        for col, vocab in self.vocabs.items():
            logger.info(f"  {col}: {len(vocab)} unique values")

        return self

    def save_vocabs(self) -> None:
        """Save vocabulary mappings to disk as JSON."""
        ensure_dir(self.vocab_dir)
        for col, vocab in self.vocabs.items():
            path = self.vocab_dir / f"{col}_vocab.json"
            with open(path, "w") as f:
                json.dump(vocab, f, indent=2)

        # Save genre mapping too
        path = self.vocab_dir / "genre_vocab.json"
        with open(path, "w") as f:
            json.dump(self.genre_to_idx, f, indent=2)

        logger.info(f"Saved vocabularies to {self.vocab_dir}")

    def load_vocabs(self) -> "FeatureEncoder":
        """Load vocabulary mappings from disk.

        Returns:
            self (for method chaining).
        """
        categorical_cols = ["user_id", "movie_id", "gender", "age", "occupation"]
        for col in categorical_cols:
            path = self.vocab_dir / f"{col}_vocab.json"
            with open(path, "r") as f:
                self.vocabs[col] = json.load(f)

        path = self.vocab_dir / "genre_vocab.json"
        with open(path, "r") as f:
            self.genre_to_idx = json.load(f)

        logger.info(f"Loaded vocabularies from {self.vocab_dir}")
        return self

    def encode_user_id(self, user_id) -> int:
        """Encode a user ID to its vocabulary index."""
        return self.vocabs["user_id"].get(
            str(user_id), 0
        )  # Returns padded index if user_id is not found

    def encode_movie_id(self, movie_id) -> int:
        """Encode a movie ID to its vocabulary index."""
        return self.vocabs["movie_id"].get(str(movie_id), 0)

    def encode_gender(self, gender) -> int:
        """Encode gender to its vocabulary index."""
        return self.vocabs["gender"].get(str(gender), 0)

    def encode_age(self, age) -> int:
        """Encode age group to its vocabulary index."""
        return self.vocabs["age"].get(str(age), 0)

    def encode_occupation(self, occupation) -> int:
        """Encode occupation to its vocabulary index."""
        return self.vocabs["occupation"].get(str(occupation), 0)

    def encode_genres(self, genres_str: str) -> np.ndarray:
        """Encode pipe-separated genres string to multi-hot vector.

        Args:
            genres_str: Pipe-separated genres, e.g. "Action|Comedy|Drama".

        Returns:
            NumPy array of shape (num_genres,) with 1s at genre positions.
        """
        multi_hot = np.zeros(len(ALL_GENRES), dtype=np.float32)
        if not genres_str or pd.isna(genres_str):
            return multi_hot
        for genre in genres_str.split("|"):
            genre = genre.strip()
            if genre in self.genre_to_idx:
                multi_hot[self.genre_to_idx[genre]] = 1.0
        return multi_hot

    def get_vocab_sizes(self) -> dict:
        """Get vocabulary sizes for embedding layer initialization.

        Returns:
            Dictionary mapping feature name -> vocab size (including padding idx).
        """
        sizes = {}
        for col, vocab in self.vocabs.items():
            sizes[col] = len(vocab) + 1  # +1 for padding index 0
        sizes["genres"] = len(ALL_GENRES)
        return sizes

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode all features in a DataFrame.

        Args:
            df: Raw DataFrame with user/movie features.

        Returns:
            DataFrame with encoded feature columns added.
        """
        df = df.copy()
        df["user_id_enc"] = df["user_id"].apply(self.encode_user_id)
        df["movie_id_enc"] = df["movie_id"].apply(self.encode_movie_id)
        df["gender_enc"] = df["gender"].apply(self.encode_gender)
        df["age_enc"] = df["age"].apply(self.encode_age)
        df["occupation_enc"] = df["occupation"].apply(self.encode_occupation)
        df["genres_enc"] = df["genres"].apply(self.encode_genres)
        return df
