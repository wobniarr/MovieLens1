"""
Data preprocessing pipeline for MovieLens-1M.

Loads raw data, merges tables, engineers features, creates train/val/test
splits with temporal ordering, and generates popularity-weighted negative
samples for the ranking model.
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def _load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load ratings.dat with proper column names."""
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(ratings):,} ratings")
    return ratings


def _load_users(data_dir: Path) -> pd.DataFrame:
    """Load users.dat with proper column names."""
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(users):,} users")
    return users


def _load_movies(data_dir: Path) -> pd.DataFrame:
    """Load movies.dat with proper column names."""
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(movies):,} movies")
    return movies


def _merge_data(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ratings, users, and movies into a single DataFrame."""
    df = ratings.merge(users, on="user_id", how="left")
    df = df.merge(movies, on="movie_id", how="left")
    logger.info(f"Merged dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def _add_implicit_signal(df: pd.DataFrame, min_rating: int) -> pd.DataFrame:
    """Add implicit binary signal and positive label columns.

    - implicit = 1 for all observed interactions (user actually rated the movie)
    - label = 1 if rating >= min_rating (positive for ranking), else 0

    Args:
        df: Merged DataFrame with rating column.
        min_rating: Threshold for positive label.

    Returns:
        DataFrame with added 'implicit' and 'label' columns.
    """
    df = df.copy()
    df["implicit"] = 1  # All observed interactions are implicit positives
    df["label"] = (df["rating"] >= min_rating).astype(int)
    logger.info(
        f"Labels: {df['label'].sum():,} positives, "
        f"{(df['label'] == 0).sum():,} negatives "
        f"(threshold: rating >= {min_rating})"
    )
    return df


def _temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally to avoid data leakage.

    Sort by timestamp, then split into train/val/test by ratio.

    Args:
        df: Full DataFrame sorted by timestamp.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Split sizes — Train: {len(train_df):,}, "
        f"Val: {len(val_df):,}, Test: {len(test_df):,}"
    )
    return train_df, val_df, test_df


def _generate_negative_samples(
    df: pd.DataFrame,
    all_movie_ids: np.ndarray,
    movie_popularity: pd.Series,
    ratio: int,
    seed: int,
) -> pd.DataFrame:
    """Generate popularity-weighted negative samples for ranking.

    For each user, sample `ratio` movies they haven't interacted with,
    weighted by movie popularity (popular unseen movies = harder negatives).

    Args:
        df: DataFrame of observed interactions.
        all_movie_ids: Array of all movie IDs in the dataset.
        movie_popularity: Series mapping movie_id -> interaction count.
        ratio: Number of negatives per positive.
        seed: Random seed.

    Returns:
        DataFrame of negative samples with implicit=0, label=0, rating=0.
    """
    rng = np.random.RandomState(seed)
    user_groups = df.groupby("user_id")

    negative_rows = []
    # Pre-compute user features for lookup
    user_features = df.drop_duplicates("user_id").set_index("user_id")[
        ["gender", "age", "occupation", "zip_code"]
    ]

    for user_id, group in user_groups:
        watched = set(group["movie_id"].values)
        candidates = np.array([m for m in all_movie_ids if m not in watched])

        if len(candidates) == 0:
            continue

        n_neg = min(len(group) * ratio, len(candidates))

        # Popularity-weighted sampling
        pop_weights = np.array([movie_popularity.get(m, 1) for m in candidates], dtype=float)
        pop_weights /= pop_weights.sum()

        sampled = rng.choice(candidates, size=n_neg, replace=False, p=pop_weights)

        user_info = user_features.loc[user_id]
        for movie_id in sampled:
            negative_rows.append({
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": 0,
                "timestamp": 0,
                "gender": user_info["gender"],
                "age": user_info["age"],
                "occupation": user_info["occupation"],
                "zip_code": user_info["zip_code"],
                "title": "",
                "genres": "",
                "implicit": 0,
                "label": 0,
            })

    neg_df = pd.DataFrame(negative_rows)
    logger.info(f"Generated {len(neg_df):,} negative samples (ratio 1:{ratio})")
    return neg_df


def preprocess_data(config: dict) -> dict:
    """Run the full preprocessing pipeline.

    Steps:
    1. Load raw data files
    2. Merge into single DataFrame
    3. Add implicit signal and labels
    4. Temporal train/val/test split
    5. Generate negative samples for ranking (train set only)
    6. Save processed data as parquet files

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with paths to saved files and data statistics.
    """
    raw_dir = Path(config["paths"]["raw_data_dir"]) / config["data"]["dataset_name"]
    processed_dir = ensure_dir(config["paths"]["processed_data_dir"])

    # Step 1: Load raw data
    logger.info("=" * 60)
    logger.info("Step 1: Loading raw data...")
    ratings = _load_ratings(raw_dir)
    users = _load_users(raw_dir)
    movies = _load_movies(raw_dir)

    # Step 2: Merge
    logger.info("Step 2: Merging datasets...")
    df = _merge_data(ratings, users, movies)

    # Step 3: Add implicit signal and labels
    logger.info("Step 3: Adding implicit signal and labels...")
    min_rating = config["data"]["min_rating_for_positive"]
    df = _add_implicit_signal(df, min_rating)

    # Step 4: Temporal split
    logger.info("Step 4: Performing temporal train/val/test split...")
    train_df, val_df, test_df = _temporal_split(
        df,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
    )

    # Step 5: Generate negative samples for ranking (train only)
    logger.info("Step 5: Generating negative samples for ranking...")
    all_movie_ids = df["movie_id"].unique()
    movie_popularity = df.groupby("movie_id").size()

    train_negatives = _generate_negative_samples(
        train_df,
        all_movie_ids,
        movie_popularity,
        ratio=config["data"]["negative_sample_ratio"],
        seed=config["data"]["random_seed"],
    )

    # Combine train positives with negatives for ranking
    train_ranking_df = pd.concat([train_df, train_negatives], ignore_index=True)
    train_ranking_df = train_ranking_df.sample(
        frac=1, random_state=config["data"]["random_seed"]
    ).reset_index(drop=True)

    # Step 6: Save to parquet
    logger.info("Step 6: Saving processed data...")
    save_paths = {
        "train_candidate_gen": processed_dir / "train_candidate_gen.parquet",
        "train_ranking": processed_dir / "train_ranking.parquet",
        "val": processed_dir / "val.parquet",
        "test": processed_dir / "test.parquet",
        "movies": processed_dir / "movies.parquet",
    }

    # Candidate gen uses only positive interactions (implicit=1)
    train_df.to_parquet(save_paths["train_candidate_gen"], index=False)
    train_ranking_df.to_parquet(save_paths["train_ranking"], index=False)
    val_df.to_parquet(save_paths["val"], index=False)
    test_df.to_parquet(save_paths["test"], index=False)
    movies.to_parquet(save_paths["movies"], index=False)

    stats = {
        "total_interactions": len(df),
        "num_users": df["user_id"].nunique(),
        "num_movies": df["movie_id"].nunique(),
        "train_size": len(train_df),
        "train_ranking_size": len(train_ranking_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_negatives": len(train_negatives),
    }

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    for k, v in stats.items():
        logger.info(f"  {k}: {v:,}")
    logger.info(f"  Files saved to: {processed_dir}")

    return {"paths": save_paths, "stats": stats}
