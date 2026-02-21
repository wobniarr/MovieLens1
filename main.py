"""
Main entry point for the MovieLens Recommender System.

CLI with subcommands:
    python main.py download         - Download MovieLens-1M dataset
    python main.py preprocess       - Run data preprocessing pipeline
    python main.py train --stage candidate_gen  - Train Two-Tower model
    python main.py train --stage ranking        - Train Ranking model
    python main.py evaluate         - Run full evaluation on test set
    python main.py recommend --user_id 42       - Get recommendations for a user
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from src.utils import load_config, set_seed, get_device, get_logger

logger = get_logger("main")


def cmd_download(args):
    """Download the MovieLens-1M dataset."""
    config = load_config(args.config)
    from src.data import download_movielens
    download_movielens(config)


def cmd_preprocess(args):
    """Run the data preprocessing pipeline."""
    config = load_config(args.config)
    from src.data import preprocess_data
    preprocess_data(config)


def cmd_train(args):
    """Train a model (candidate_gen or ranking)."""
    if args.stage == "candidate_gen":
        from scripts.train_candidate_gen import train_candidate_gen
        train_candidate_gen(args.config)
    elif args.stage == "ranking":
        from scripts.train_ranking import train_ranking
        train_ranking(args.config)
    else:
        logger.error(f"Unknown stage: {args.stage}. Choose 'candidate_gen' or 'ranking'.")
        sys.exit(1)


def cmd_evaluate(args):
    """Run evaluation on the test set."""
    from scripts.evaluate import evaluate
    evaluate(args.config)


def cmd_recommend(args):
    """Generate recommendations for a specific user."""
    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])

    from src.features import FeatureEncoder
    from src.models import TwoTowerModel, RankingModel
    from src.inference import RecommendationPipeline

    # Load encoder and vocab
    encoder = FeatureEncoder(config)
    encoder.load_vocabs()
    vocab_sizes = encoder.get_vocab_sizes()

    # Load movies data
    movies_df = pd.read_parquet(config["paths"]["processed_data_dir"] + "/movies.parquet")

    # Load users data from raw dataset
    raw_dir = Path(config["paths"]["raw_data_dir"]) / config["data"]["dataset_name"]
    users_df = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    # Load Two-Tower model
    two_tower = TwoTowerModel(vocab_sizes, config)
    cg_checkpoint = torch.load(
        Path(config["paths"]["checkpoints_dir"]) / "candidate_gen" / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    two_tower.load_state_dict(cg_checkpoint["model_state_dict"])

    # Load Ranking model
    ranker = RankingModel(vocab_sizes, config)
    rank_checkpoint = torch.load(
        Path(config["paths"]["checkpoints_dir"]) / "ranking" / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    ranker.load_state_dict(rank_checkpoint["model_state_dict"])

    # Build pipeline
    pipeline = RecommendationPipeline(
        two_tower_model=two_tower,
        ranking_model=ranker,
        encoder=encoder,
        movies_df=movies_df,
        config=config,
        device=device,
    )

    # Generate recommendations
    logger.info(f"\nGenerating recommendations for user {args.user_id}...")
    results = pipeline.recommend(
        user_id=args.user_id,
        users_df=users_df,
        top_k_candidates=config["candidate_gen"]["top_k"],
        top_n_final=args.top_n,
    )

    # Display results
    print(f"\n{'=' * 70}")
    print(f" Top-{args.top_n} Recommendations for User {args.user_id}")
    print(f"{'=' * 70}")
    for rank, row in results.iterrows():
        print(
            f"  {rank:2d}. {row['title']:<45s} "
            f"({row['genres']:<30s}) "
            f"score: {row['ranking_score']:.4f}"
        )
    print(f"{'=' * 70}\n")

    return results


def main():
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="MovieLens Recommender System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download
  python main.py preprocess
  python main.py train --stage candidate_gen
  python main.py train --stage ranking
  python main.py evaluate
  python main.py recommend --user_id 42
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download
    dl_parser = subparsers.add_parser("download", help="Download MovieLens-1M dataset")
    dl_parser.add_argument("--config", default="configs/default.yaml")

    # Preprocess
    pp_parser = subparsers.add_parser("preprocess", help="Preprocess raw data")
    pp_parser.add_argument("--config", default="configs/default.yaml")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--stage",
        required=True,
        choices=["candidate_gen", "ranking"],
        help="Which model to train",
    )
    train_parser.add_argument("--config", default="configs/default.yaml")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models on test set")
    eval_parser.add_argument("--config", default="configs/default.yaml")

    # Recommend
    rec_parser = subparsers.add_parser("recommend", help="Get recommendations for a user")
    rec_parser.add_argument("--user_id", type=int, required=True, help="User ID")
    rec_parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations")
    rec_parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "recommend": cmd_recommend,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
