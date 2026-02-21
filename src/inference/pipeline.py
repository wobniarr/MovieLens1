"""
End-to-end recommendation inference pipeline.

Two-stage process:
1. Candidate Generation: Use the Two-Tower model to retrieve top-K candidate
   movies via dot-product similarity between user and item embeddings.
2. Ranking: Score the candidates with the Ranking model using hybrid features
   (explicit rating + implicit signal) and return the final ranked list.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.features import FeatureEncoder
from src.models import TwoTowerModel, RankingModel
from src.utils import get_logger

logger = get_logger(__name__)


class ItemIndexDataset(Dataset):
    """Helper dataset for computing all item embeddings."""

    def __init__(self, movies_df: pd.DataFrame, encoder: FeatureEncoder):
        self.movie_ids = np.array([
            encoder.encode_movie_id(mid) for mid in movies_df["movie_id"]
        ])
        self.genres = np.stack([
            encoder.encode_genres(g) for g in movies_df["genres"]
        ])
        self.raw_movie_ids = movies_df["movie_id"].values

    def __len__(self):
        return len(self.movie_ids)

    def __getitem__(self, idx):
        return {
            "movie_id": torch.tensor(self.movie_ids[idx], dtype=torch.long),
            "genres": torch.tensor(self.genres[idx], dtype=torch.float),
        }


class RecommendationPipeline:
    """End-to-end recommendation pipeline.

    Combines the Two-Tower candidate generation model and the Ranking model
    to produce final movie recommendations for a user.
    """

    def __init__(
        self,
        two_tower_model: TwoTowerModel,
        ranking_model: RankingModel,
        encoder: FeatureEncoder,
        movies_df: pd.DataFrame,
        config: dict,
        device: torch.device,
    ):
        """Initialize the pipeline.

        Args:
            two_tower_model: Trained Two-Tower model.
            ranking_model: Trained Ranking model.
            encoder: Fitted FeatureEncoder with loaded vocabularies.
            movies_df: DataFrame of all movies with movie_id, title, genres.
            config: Configuration dictionary.
            device: Compute device.
        """
        self.two_tower = two_tower_model.to(device).eval()
        self.ranker = ranking_model.to(device).eval()
        self.encoder = encoder
        self.movies_df = movies_df.reset_index(drop=True)
        self.config = config
        self.device = device

        # Build item index (pre-compute all item embeddings)
        self._item_embeddings = None
        self._item_movie_ids = None
        self._build_item_index()

    @torch.no_grad()
    def _build_item_index(self) -> None:
        """Pre-compute embeddings for all items in the catalog."""
        logger.info("Building item embedding index...")
        item_dataset = ItemIndexDataset(self.movies_df, self.encoder)
        item_loader = DataLoader(item_dataset, batch_size=512, shuffle=False)

        all_embeddings = []
        for batch in item_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            emb = self.two_tower.get_item_embedding(batch)
            all_embeddings.append(emb.cpu())

        self._item_embeddings = torch.cat(all_embeddings, dim=0)
        self._item_movie_ids = self.movies_df["movie_id"].values
        logger.info(f"Item index built: {len(self._item_movie_ids)} movies")

    @torch.no_grad()
    def _retrieve_candidates(
        self,
        user_features: Dict[str, torch.Tensor],
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1: Retrieve top-K candidates using Two-Tower dot product.

        Args:
            user_features: Encoded user feature tensors.
            top_k: Number of candidates to retrieve.

        Returns:
            Tuple of (candidate_movie_ids, candidate_scores).
        """
        user_emb = self.two_tower.get_user_embedding(user_features)  # (1, emb_dim)

        # Brute-force dot product against all items
        item_embs = self._item_embeddings.to(self.device)
        scores = torch.matmul(user_emb, item_embs.T).squeeze(0)  # (num_items,)

        # Get top-K
        top_k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, top_k)

        candidate_ids = self._item_movie_ids[top_indices.cpu().numpy()]
        candidate_scores = top_scores.cpu().numpy()

        return candidate_ids, candidate_scores

    @torch.no_grad()
    def _rank_candidates(
        self,
        user_info: dict,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        """Stage 2: Score candidates with the Ranking model.

        Args:
            user_info: Dictionary with raw user features.
            candidate_ids: Array of candidate movie IDs.

        Returns:
            Array of ranking scores for each candidate.
        """
        # Build ranking input for each candidate
        batch = {
            "user_id": [],
            "gender": [],
            "age": [],
            "occupation": [],
            "movie_id": [],
            "genres": [],
            "rating": [],
            "implicit": [],
        }

        for movie_id in candidate_ids:
            movie_row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            genres_str = movie_row["genres"].values[0] if len(movie_row) > 0 else ""

            batch["user_id"].append(self.encoder.encode_user_id(user_info["user_id"]))
            batch["gender"].append(self.encoder.encode_gender(user_info["gender"]))
            batch["age"].append(self.encoder.encode_age(user_info["age"]))
            batch["occupation"].append(self.encoder.encode_occupation(user_info["occupation"]))
            batch["movie_id"].append(self.encoder.encode_movie_id(movie_id))
            batch["genres"].append(self.encoder.encode_genres(genres_str))
            # For inference: no prior rating, mark as new interaction
            batch["rating"].append(0.0)
            batch["implicit"].append(0.0)

        # Convert to tensors
        tensor_batch = {
            "user_id": torch.tensor(batch["user_id"], dtype=torch.long).to(self.device),
            "gender": torch.tensor(batch["gender"], dtype=torch.long).to(self.device),
            "age": torch.tensor(batch["age"], dtype=torch.long).to(self.device),
            "occupation": torch.tensor(batch["occupation"], dtype=torch.long).to(self.device),
            "movie_id": torch.tensor(batch["movie_id"], dtype=torch.long).to(self.device),
            "genres": torch.stack([torch.tensor(g, dtype=torch.float) for g in batch["genres"]]).to(self.device),
            "rating": torch.tensor(batch["rating"], dtype=torch.float).to(self.device),
            "implicit": torch.tensor(batch["implicit"], dtype=torch.float).to(self.device),
        }

        logits = self.ranker(tensor_batch)
        scores = torch.sigmoid(logits).cpu().numpy()

        return scores

    def recommend(
        self,
        user_id: int,
        users_df: pd.DataFrame,
        top_k_candidates: int = 100,
        top_n_final: int = 10,
    ) -> pd.DataFrame:
        """Generate recommendations for a user.

        Two-stage pipeline:
        1. Retrieve top_k_candidates using Two-Tower model
        2. Re-rank with Ranking model, return top_n_final

        Args:
            user_id: User ID to generate recommendations for.
            users_df: DataFrame with user information.
            top_k_candidates: Number of candidates from stage 1.
            top_n_final: Number of final recommendations to return.

        Returns:
            DataFrame with columns: rank, movie_id, title, genres,
            retrieval_score, ranking_score.
        """
        # Get user info
        user_row = users_df[users_df["user_id"] == user_id]
        if len(user_row) == 0:
            raise ValueError(f"User {user_id} not found in dataset")

        user_info = {
            "user_id": user_id,
            "gender": user_row["gender"].values[0],
            "age": user_row["age"].values[0],
            "occupation": user_row["occupation"].values[0],
        }

        # Stage 1: Candidate Generation
        user_features = {
            "user_id": torch.tensor([self.encoder.encode_user_id(user_id)], dtype=torch.long).to(self.device),
            "gender": torch.tensor([self.encoder.encode_gender(user_info["gender"])], dtype=torch.long).to(self.device),
            "age": torch.tensor([self.encoder.encode_age(user_info["age"])], dtype=torch.long).to(self.device),
            "occupation": torch.tensor([self.encoder.encode_occupation(user_info["occupation"])], dtype=torch.long).to(self.device),
        }

        candidate_ids, retrieval_scores = self._retrieve_candidates(
            user_features, top_k_candidates
        )
        logger.info(f"Stage 1: Retrieved {len(candidate_ids)} candidates")

        # Stage 2: Ranking
        ranking_scores = self._rank_candidates(user_info, candidate_ids)
        logger.info(f"Stage 2: Ranked {len(candidate_ids)} candidates")

        # Build results
        results = []
        for i, (movie_id, ret_score, rank_score) in enumerate(
            zip(candidate_ids, retrieval_scores, ranking_scores)
        ):
            movie_row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            title = movie_row["title"].values[0] if len(movie_row) > 0 else "Unknown"
            genres = movie_row["genres"].values[0] if len(movie_row) > 0 else ""
            results.append({
                "movie_id": movie_id,
                "title": title,
                "genres": genres,
                "retrieval_score": float(ret_score),
                "ranking_score": float(rank_score),
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("ranking_score", ascending=False).head(top_n_final)
        results_df = results_df.reset_index(drop=True)
        results_df.index += 1
        results_df.index.name = "rank"

        return results_df
