"""
Data preprocessing pipeline for MovieLens-1M.

Loads raw data, merges tables, engineers features, creates train/val/test
splits with temporal ordering, and generates popularity-weighted negative
samples for the ranking model.
"""

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
    ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame
) -> pd.DataFrame:
    """Merge ratings, users, and movies into a single DataFrame."""
    df = ratings.merge(users, on="user_id", how="left")
    df = df.merge(movies, on="movie_id", how="left")
    logger.info(f"Merged dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def _add_labels(df: pd.DataFrame, min_rating: int) -> pd.DataFrame:
    """Add label column based on rating threshold. 1 for positive, 0 for negative.

    Args:
        df: Merged DataFrame with rating column.
        min_rating: Threshold for positive label.

    Returns:
        DataFrame with added 'label' column.
    """
    df = df.copy()
    df["label"] = (df["rating"] >= min_rating).astype(int)
    logger.info(
        f"Labels: {df['label'].sum():,} positives, "
        f"{(df['label'] == 0).sum():,} negatives "
        f"(threshold: rating >= {min_rating})"
    )
    return df


def _temporal_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by timestamp, then split into train/val/test by ratio."""
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


def _build_watch_histories(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_history_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-interaction watch history for each split.

    For training rows: history = user's prior movie_ids within train (by timestamp).
    For val/test rows: history = all of user's train interactions (no leakage).

    Args:
        train_df: Training DataFrame, sorted by timestamp.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        max_history_len: Maximum number of history items to keep.

    Returns:
        Tuple of (train_df, val_df, test_df) with 'watch_history' column added.
    """
    # --- Training histories: sliding window per user within train ---
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    train_histories = []
    user_history = {}  # user_id -> list of movie_ids seen so far

    for _, row in train_df.iterrows():
        uid = row["user_id"]
        mid = row["movie_id"]

        # Get current history for this user (before this interaction)
        hist = user_history.get(uid, [])
        # Cap to last N items, pad to fixed length
        hist_trimmed = hist[-max_history_len:]
        padded = [0] * (max_history_len - len(hist_trimmed)) + hist_trimmed
        train_histories.append(padded)

        # Update running history
        user_history.setdefault(uid, []).append(mid)

    train_df = train_df.copy()
    train_df["watch_history"] = train_histories

    # --- Val/Test histories: all train interactions per user ---
    # user_history now contains each user's full training history
    def _build_eval_history(df):
        histories = []
        for _, row in df.iterrows():
            uid = row["user_id"]
            hist = user_history.get(uid, [])
            hist_trimmed = hist[-max_history_len:]
            padded = [0] * (max_history_len - len(hist_trimmed)) + hist_trimmed
            histories.append(padded)
        result = df.copy()
        result["watch_history"] = histories
        return result

    val_df = _build_eval_history(val_df)
    test_df = _build_eval_history(test_df)

    # Log stats
    train_lens = [len([x for x in h if x != 0]) for h in train_histories]
    logger.info(
        f"Watch histories built (max_len={max_history_len}): "
        f"avg={np.mean(train_lens):.1f}, "
        f"median={np.median(train_lens):.0f}, "
        f"users_with_history={sum(1 for l in train_lens if l > 0):,}"
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
        movie_popularity: Series mapping movie_id -> interaction count. Generated in preprocess_data function.
        ratio: Number of negatives per positive.
        seed: Local random seed. Used to ensure data consistency across experimental runs.

    Returns:
        DataFrame of negative samples with label=0, rating=0.
    """
    rng = np.random.RandomState(seed)
    user_groups = df.groupby("user_id")

    negative_rows = []
    # Pre-compute user features for lookup. Less overhead than calling .loc in the loop.
    user_features = df.drop_duplicates("user_id").set_index("user_id")[
        ["gender", "age", "occupation", "zip_code"]
    ]

    for user_id, group in user_groups:
        watched = set(group["movie_id"].values)  # Set used for O(1) lookups
        candidates = np.array([m for m in all_movie_ids if m not in watched])

        n_neg = min(
            len(group) * ratio, len(candidates)
        )  # Make sure our negative samples don't exceed the number of movies the user hasn't watched

        pop_weights = np.array(
            [movie_popularity.get(m, 1) for m in candidates], dtype=float
        )  # Get the popularity of each candidate movie
        pop_weights /= pop_weights.sum()  # Normalize the popularity weights to sum to 1

        sampled = rng.choice(
            candidates, size=n_neg, replace=False, p=pop_weights
        )  # Generate random negative samples weighted by movie popularity

        user_info = user_features.loc[user_id]
        # Create negative samples as a dict and append to the list. Entire list is later converted to a DataFrame to avoid overhead of repeated DataFrame appends.
        for movie_id in sampled:
            negative_rows.append(
                {
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
                    "label": 0,
                }
            )

    neg_df = pd.DataFrame(negative_rows)
    logger.info(f"Generated {len(neg_df):,} negative samples (ratio 1:{ratio})")
    return neg_df


def preprocess_data(config: dict) -> dict:
    """Run the full preprocessing pipeline.

    Steps:
    1. Load raw data files
    2. Merge into single DataFrame
    3. Add labels
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

    # Step 3: Add labels
    logger.info("Step 3: Adding labels...")
    min_rating = config["data"]["min_rating_for_positive"]
    df = _add_labels(df, min_rating)

    # Step 4: Temporal split
    logger.info("Step 4: Performing temporal train/val/test split...")
    train_df, val_df, test_df = _temporal_split(
        df,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
    )

    # Step 4.5: Build user watch histories
    logger.info("Step 4.5: Building user watch histories...")
    max_history_len = config["candidate_gen"].get("max_history_len", 50)
    train_df, val_df, test_df = _build_watch_histories(
        train_df, val_df, test_df, max_history_len
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

    # Combine train positives with negatives for ranking and shuffle so that positives and negatives are not grouped together
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

    train_df.to_parquet(
        save_paths["train_candidate_gen"], index=False
    )  # Candidate gen uses only positive interactions
    train_ranking_df.to_parquet(
        save_paths["train_ranking"], index=False
    )  # Ranking uses positives + negatives
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
