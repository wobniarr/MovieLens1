"""
Generic, reusable training loop for both candidate generation and ranking.

Features:
- Config-driven (epochs, LR, scheduler, early stopping)
- Gradient clipping
- Checkpoint saving (best model by validation metric)
- Console logging with progress bars
"""

import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class Trainer:
    """Generic trainer for PyTorch models.

    Handles the training loop, validation, checkpointing, and early stopping.
    Works for both the Two-Tower and Ranking models through configurable
    loss functions and evaluation callbacks.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: dict,
        stage: str,
        device: torch.device,
        eval_fn: Optional[Callable] = None,
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            loss_fn: Loss function module.
            config: Full configuration dictionary.
            stage: Either "candidate_gen" or "ranking".
            device: Training device.
            eval_fn: Optional evaluation function called after each epoch.
                     Should accept (model, val_loader, device) and return
                     a dict with metric names and values. The first metric
                     is used for early stopping.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.stage = stage
        self.device = device
        self.eval_fn = eval_fn

        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config[stage]["learning_rate"],
            weight_decay=config[stage]["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )

        # Training settings
        self.num_epochs = config[stage]["num_epochs"]
        self.gradient_clip_norm = config["training"]["gradient_clip_norm"]
        self.early_stopping_patience = config[stage]["early_stopping_patience"]
        self.log_every_n_steps = config["training"]["log_every_n_steps"]

        # Checkpointing
        self.checkpoint_dir = ensure_dir(
            Path(config["paths"]["checkpoints_dir"]) / stage
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch.

        Args:
            train_loader: Training DataLoader.
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]",
            leave=False,
        )

        for step, batch in enumerate(progress):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            if self.stage == "candidate_gen":
                output = self.model(batch)
                loss = self.loss_fn(output["logits"])
            else:  # ranking
                logits = self.model(batch)
                loss = self.loss_fn(logits, batch["label"])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and compute average loss.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.stage == "candidate_gen":
                output = self.model(batch)
                loss = self.loss_fn(output["logits"])
            else:
                logits = self.model(batch)
                loss = self.loss_fn(logits, batch["label"])

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch.
            val_loss: Validation loss.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info(f"  💾 Saved best model (val_loss={val_loss:.4f})")

    def load_best_model(self) -> None:
        """Load the best model checkpoint."""
        path = self.checkpoint_dir / "best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded best model from epoch {checkpoint['epoch'] + 1} "
            f"(val_loss={checkpoint['val_loss']:.4f})"
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Dictionary with training history (losses and metrics per epoch).
        """
        logger.info(f"Starting training: {self.stage}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(
            f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        logger.info("=" * 60)

        history = {"train_loss": [], "val_loss": [], "metrics": []}

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train
            train_loss = self._train_one_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self._validate(val_loader)
            history["val_loss"].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            # Evaluate with custom metrics
            metrics = {}
            if self.eval_fn is not None:
                metrics = self.eval_fn(self.model, val_loader, self.device)
                history["metrics"].append(metrics)
                metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            else:
                metrics_str = ""

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
            if metrics_str:
                logger.info(f"  Metrics: {metrics_str}")

            # Checkpointing & early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, val_loss, is_best)

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={self.early_stopping_patience})"
                )
                break

        # Load best model at end
        self.load_best_model()
        logger.info("Training complete! Best model loaded.")

        return history
