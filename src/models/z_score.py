"""Model A: User-Normalized Z-Score Aggregator.

Calibrates ratings by standardizing user leniency and variance, then re-projecting
item mean z-scores back onto the global 1.0 - 10.0 scale.
"""

from __future__ import annotations

from typing import Optional
import polars as pl


class UserNormalizedZScore:
    """User-Normalized Z-Score Baseline Model.

    De-biases ratings at the user level:
        z_ui = (r_ui - mu_u) / (sigma_u + eps)
        z_bar_i = (1 / |U_i|) * sum_{u in U_i} z_ui
        S_hat_i = mu_global + (z_bar_i * sigma_global)
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.global_mean: Optional[float] = None
        self.global_std: Optional[float] = None
        self.item_scores_df: Optional[pl.DataFrame] = None
        self.user_stats_df: Optional[pl.DataFrame] = None

    def fit(self, ratings_df: pl.DataFrame) -> "UserNormalizedZScore":
        """Fits user normalization statistics and computes item calibrated scores.

        Args:
            ratings_df: Polars DataFrame with columns ['userId', 'movieId', 'rating']
                        (ratings scaled to 1.0 - 10.0).
        """
        # Global population parameters
        self.global_mean = float(ratings_df["rating"].mean())
        self.global_std = float(ratings_df["rating"].std())

        # Compute user statistics
        # Note: if a user has all identical ratings, std is 0.0
        ratings_with_user_stats = ratings_df.with_columns([
            pl.col("rating").mean().over("userId").alias("user_mean"),
            pl.col("rating").std().fill_null(0.0).over("userId").alias("user_std"),
        ])

        # Standardize every interaction
        z_ratings = ratings_with_user_stats.with_columns(
            (
                (pl.col("rating") - pl.col("user_mean"))
                / (pl.col("user_std") + self.eps)
            ).alias("z_score")
        )

        # Store user stats for reference
        self.user_stats_df = (
            ratings_with_user_stats
            .select(["userId", "user_mean", "user_std"])
            .unique(subset=["userId"])
        )

        # Aggregate item mean z-scores
        self.item_scores_df = (
            z_ratings
            .group_by("movieId")
            .agg([
                pl.col("z_score").mean().alias("mean_z"),
                pl.col("z_score").count().alias("vote_count"),
                pl.col("rating").mean().alias("raw_avg"),
            ])
            .with_columns([
                (
                    self.global_mean + pl.col("mean_z") * self.global_std
                ).clip(1.0, 10.0).alias("score_model_a")
            ])
            .sort("score_model_a", descending=True)
        )

        return self

    def get_item_scores(self) -> pl.DataFrame:
        """Returns the item-level scores DataFrame."""
        if self.item_scores_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.item_scores_df

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicts calibrated rating for a specific user and movie."""
        if self.item_scores_df is None or self.user_stats_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        item_row = self.item_scores_df.filter(pl.col("movieId") == movie_id)
        if len(item_row) == 0:
            return self.global_mean

        z_i = item_row["mean_z"][0]

        user_row = self.user_stats_df.filter(pl.col("userId") == user_id)
        if len(user_row) == 0:
            return float(item_row["score_model_a"][0])

        u_mean = user_row["user_mean"][0]
        u_std = user_row["user_std"][0]

        pred = u_mean + (z_i * (u_std if u_std > 0 else self.global_std))
        return float(min(10.0, max(1.0, pred)))
