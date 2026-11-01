"""Analytical metrics and ranking evaluation for UniversalRatingEngine.

Implements:
- IMDb Bayesian weighted average formula
- Score deltas and rank shifts (enthusiast inflation vs. harsh audience drag)
- RMSE, MAE, and Spearman rank correlation
- Cross-genre summary aggregation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def compute_bayesian_average(
    raw_avg: float | pl.Expr,
    vote_count: int | pl.Expr,
    global_mean: float,
    min_votes_threshold: int = 50,
) -> pl.Expr | float:
    """Computes the standard IMDb Bayesian weighted rating:

        WR = (v / (v + m)) * R + (m / (v + m)) * C

    Args:
        raw_avg: Arithmetic mean rating (R).
        vote_count: Number of votes/ratings (v).
        global_mean: Global dataset mean rating (C).
        min_votes_threshold: Minimum votes threshold (m).
    """
    if isinstance(raw_avg, pl.Expr):
        v = pl.col("vote_count") if not isinstance(vote_count, pl.Expr) else vote_count
        r = raw_avg
        m = float(min_votes_threshold)
        c = float(global_mean)
        return (v / (v + m)) * r + (m / (v + m)) * c
    else:
        v = float(vote_count)
        m = float(min_votes_threshold)
        return (v / (v + m)) * float(raw_avg) + (m / (v + m)) * float(global_mean)


def format_rank_change(raw_rank: int, cal_rank: int) -> str:
    """Formats rank change with diagnostic explanation.

    Positive delta (cal_rank > raw_rank): item fell down the board.
    Negative delta (cal_rank < raw_rank): item moved up the board.
    """
    delta = cal_rank - raw_rank
    if delta > 0:
        return f"Dropped {delta:,} ranks due to enthusiast bias"
    elif delta < 0:
        return f"Gained {abs(delta):,} ranks due to harsh audience penalty"
    else:
        return "Rank unchanged (neutral audience calibration)"


def calculate_metrics_summary(
    true_ratings: np.ndarray,
    predicted_ratings: np.ndarray,
) -> Dict[str, float]:
    """Calculates RMSE and MAE between ground truth and model predictions."""
    diff = true_ratings - predicted_ratings
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {
        "rmse": rmse,
        "mae": mae,
    }


def compute_rank_correlation(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
) -> float:
    """Calculates Spearman rank correlation coefficient between two score sets."""
    corr, _ = spearmanr(scores_1, scores_2)
    return float(corr)


def compute_cross_genre_summary(
    calibrated_df: pl.DataFrame,
    primary_cal_col: str = "score_model_b",
) -> pl.DataFrame:
    """Aggregates average inflation/deflation per genre.

    Calibration delta = Calibrated_Score - Raw_Avg.
    A negative delta indicates the genre was inflated by enthusiast siloing.
    A positive delta indicates the genre was dragged down by mismatched viewers.
    """
    exploded = (
        calibrated_df
        .with_columns(pl.col("genres").str.split("|"))
        .explode("genres")
        .filter((pl.col("genres") != "(no genres listed)") & (pl.col("genres") != ""))
    )

    genre_summary = (
        exploded
        .group_by("genres")
        .agg([
            pl.col("movieId").count().alias("title_count"),
            pl.col("vote_count").sum().alias("total_votes"),
            pl.col("raw_avg").mean().alias("avg_raw_rating"),
            pl.col("bayesian_avg").mean().alias("avg_bayesian_rating"),
            pl.col("score_model_a").mean().alias("avg_model_a"),
            pl.col(primary_cal_col).mean().alias("avg_model_b"),
            pl.col("score_model_c").mean().alias("avg_model_c"),
            (pl.col(primary_cal_col) - pl.col("raw_avg")).mean().alias("avg_calibration_delta"),
            (pl.col("score_model_c") - pl.col("raw_avg")).mean().alias("avg_bridge_delta"),
        ])
        .sort("avg_calibration_delta")
    )

    return genre_summary
