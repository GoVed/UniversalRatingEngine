"""Model B: Biased Matrix Factorization (Latent Factor Decomposition).

Formulates ratings as:
    r_hat_ui = mu + b_u + b_i + p_u^T q_i

Optimizes via vectorized SGD / AdamW with L2 regularization:
    min_{b, p, q} sum_{(u, i)} (r_ui - mu - b_u - b_i - p_u^T q_i)^2 + lambda * (b_u^2 + b_i^2 + ||p_u||_2^2 + ||q_i||_2^2)

Extracts b_i as the primary latent indicator of cross-genre quality, then re-scales
b_i to the 1.0 - 10.0 range.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from rich.console import Console

console = Console()


class BiasedMFModule(nn.Module):
    """PyTorch module implementing Biased Matrix Factorization."""

    def __init__(self, num_users: int, num_items: int, k: int = 32) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.k = k

        # Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Latent factor vectors
        self.user_factors = nn.Embedding(num_users, k)
        self.item_factors = nn.Embedding(num_items, k)

        # Parameter initialization
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)

    def forward(
        self,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
        global_mean: float,
    ) -> torch.Tensor:
        """Computes predicted rating: mu + b_u + b_i + p_u^T q_i."""
        b_u = self.user_bias(u_idx).squeeze(-1)
        b_i = self.item_bias(i_idx).squeeze(-1)
        interaction = (self.user_factors(u_idx) * self.item_factors(i_idx)).sum(dim=-1)
        return global_mean + b_u + b_i + interaction

    def regularization_loss(
        self,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Computes L2 regularization penalty for the active batch."""
        b_u = self.user_bias(u_idx)
        b_i = self.item_bias(i_idx)
        p_u = self.user_factors(u_idx)
        q_i = self.item_factors(i_idx)

        return (
            b_u.pow(2).mean()
            + b_i.pow(2).mean()
            + p_u.pow(2).mean()
            + q_i.pow(2).mean()
        )


class BiasedMatrixFactorization:
    """Biased Matrix Factorization model for decomposing user leniency and item quality."""

    def __init__(
        self,
        k: int = 32,
        lambda_reg: float = 0.02,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 32768,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.k = k
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.global_mean: Optional[float] = None
        self.model: Optional[BiasedMFModule] = None
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.item_scores_df: Optional[pl.DataFrame] = None
        self.final_rmse: Optional[float] = None

    def fit(self, ratings_df: pl.DataFrame, verbose: bool = True) -> "BiasedMatrixFactorization":
        """Fits biased matrix factorization on the ratings DataFrame.

        Args:
            ratings_df: Polars DataFrame with ['userId', 'movieId', 'rating']
            verbose: Whether to log epoch progress
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.global_mean = float(ratings_df["rating"].mean())

        # Build contiguous index mappings
        unique_users = ratings_df["userId"].unique().to_list()
        unique_items = ratings_df["movieId"].unique().to_list()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        self.item_to_idx = {mid: idx for idx, mid in enumerate(unique_items)}
        self.idx_to_item = {idx: mid for mid, idx in self.item_to_idx.items()}

        num_users = len(unique_users)
        num_items = len(unique_items)
        num_ratings = len(ratings_df)

        if verbose:
            console.print(
                f"[bold cyan]Biased MF Training[/bold cyan] (k={self.k}, λ={self.lambda_reg}, "
                f"lr={self.learning_rate}, device={self.device}) | "
                f"Users: {num_users:,}, Items: {num_items:,}, Ratings: {num_ratings:,}"
            )

        # Vectorized mapping using Polars hash joins
        user_map_df = pl.DataFrame({
            "userId": unique_users,
            "u_idx": np.arange(num_users, dtype=np.int64),
        })
        item_map_df = pl.DataFrame({
            "movieId": unique_items,
            "i_idx": np.arange(num_items, dtype=np.int64),
        })

        indexed_ratings = (
            ratings_df
            .join(user_map_df, on="userId")
            .join(item_map_df, on="movieId")
        )

        u_tensor = torch.tensor(indexed_ratings["u_idx"].to_numpy(), dtype=torch.long)
        i_tensor = torch.tensor(indexed_ratings["i_idx"].to_numpy(), dtype=torch.long)
        r_tensor = torch.tensor(indexed_ratings["rating"].to_numpy(), dtype=torch.float32)

        # Move to target device if feasible
        try:
            u_tensor = u_tensor.to(self.device)
            i_tensor = i_tensor.to(self.device)
            r_tensor = r_tensor.to(self.device)
            on_device = True
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            on_device = False
            console.print("[yellow]Warning: Insufficient GPU VRAM for full tensor cache; using CPU streaming.[/yellow]")

        self.model = BiasedMFModule(num_users, num_items, k=self.k).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,  # Explicitly handled in loss function
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.learning_rate * 0.1
        )

        self.model.train()
        t_start = time.time()

        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(num_ratings, device=self.device if on_device else "cpu")
            total_loss = 0.0
            total_samples = 0

            for start in range(0, num_ratings, self.batch_size):
                end = min(start + self.batch_size, num_ratings)
                batch_indices = perm[start:end]

                if on_device:
                    b_u = u_tensor[batch_indices]
                    b_i = i_tensor[batch_indices]
                    b_r = r_tensor[batch_indices]
                else:
                    b_u = u_tensor[batch_indices].to(self.device)
                    b_i = i_tensor[batch_indices].to(self.device)
                    b_r = r_tensor[batch_indices].to(self.device)

                pred = self.model(b_u, b_i, self.global_mean)
                mse_loss = nn.functional.mse_loss(pred, b_r)
                reg_loss = self.lambda_reg * self.model.regularization_loss(b_u, b_i)
                loss = mse_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += mse_loss.item() * len(b_r)
                total_samples += len(b_r)

            scheduler.step()
            epoch_rmse = np.sqrt(total_loss / total_samples)
            self.final_rmse = float(epoch_rmse)

            if verbose:
                console.print(
                    f"  [dim]Epoch {epoch:02d}/{self.epochs:02d}[/dim] - "
                    f"RMSE: [bold green]{epoch_rmse:.4f}[/bold green] "
                    f"(lr: {scheduler.get_last_lr()[0]:.5f})"
                )

        train_time = time.time() - t_start
        if verbose:
            console.print(
                f"[green]✓[/green] Biased MF trained in [bold]{train_time:.2f}s[/bold] | "
                f"Final Training RMSE: [bold green]{self.final_rmse:.4f}[/bold green]"
            )

        # Free training tensors and empty CUDA cache
        del u_tensor, i_tensor, r_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._extract_item_scores(ratings_df)
        return self

    def _extract_item_scores(self, ratings_df: pl.DataFrame) -> None:
        """Extracts latent item bias b_i and calculates calibrated quality scores."""
        self.model.eval()
        with torch.no_grad():
            b_i_all = self.model.item_bias.weight.squeeze(-1).cpu().numpy()

        items = [self.idx_to_item[idx] for idx in range(len(b_i_all))]
        b_i_vals = b_i_all

        # Raw arithmetic averages and vote counts
        item_stats = (
            ratings_df
            .group_by("movieId")
            .agg([
                pl.col("rating").mean().alias("raw_avg"),
                pl.col("rating").count().alias("vote_count"),
            ])
        )

        # Construct items DataFrame
        scores_df = pl.DataFrame({
            "movieId": items,
            "bias_b_i": b_i_vals,
        }).join(item_stats, on="movieId", how="left")

        # Rescaling b_i to 1.0 - 10.0 scale:
        # 1. Primary latent indicator: intrinsic neutral quality = mu + b_i
        # 2. Rescaled version: min-max mapping b_i to 1.0 - 10.0 range
        b_min = float(scores_df["bias_b_i"].min())
        b_max = float(scores_df["bias_b_i"].max())
        b_range = b_max - b_min if b_max > b_min else 1.0

        scores_df = scores_df.with_columns([
            (self.global_mean + pl.col("bias_b_i")).clip(1.0, 10.0).alias("score_model_b"),
            (1.0 + 9.0 * (pl.col("bias_b_i") - b_min) / b_range).alias("score_model_b_rescaled"),
        ]).sort("score_model_b", descending=True)

        self.item_scores_df = scores_df

    def get_item_scores(self) -> pl.DataFrame:
        """Returns the item-level scores DataFrame."""
        if self.item_scores_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.item_scores_df

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicts rating for a specific user and movie: mu + b_u + b_i + p_u^T q_i."""
        if self.model is None or self.global_mean is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(movie_id)

        self.model.eval()
        with torch.no_grad():
            if u_idx is None and i_idx is None:
                return self.global_mean
            elif u_idx is None:
                b_i = self.model.item_bias.weight[i_idx].item()
                return float(np.clip(self.global_mean + b_i, 1.0, 10.0))
            elif i_idx is None:
                b_u = self.model.user_bias.weight[u_idx].item()
                return float(np.clip(self.global_mean + b_u, 1.0, 10.0))
            else:
                u_t = torch.tensor([u_idx], device=self.device)
                i_t = torch.tensor([i_idx], device=self.device)
                pred = self.model(u_t, i_t, self.global_mean).item()
                return float(np.clip(pred, 1.0, 10.0))
