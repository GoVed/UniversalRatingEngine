"""Model C: Bridge-User & Genre Deflation Index.

Counteracts niche enthusiast score inflation and broad-audience score drag:
1. Calculates a "Genre Entropism / Diversity Index" for each user based on the Shannon
   entropy of the genres they rate.
2. Identifies "Bridge Users" (high genre diversity) and "Genre Loyalists" (siloed consumption).
3. Computes a genre-level inflation multiplier M_g comparing loyalist vs. bridge rating averages.
4. Calibrates items: S_bridge(i) = RawAvg(i) * M_{g(i)}.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from rich.console import Console

console = Console()


class BridgeCalibrationModel:
    """Bridge-User & Genre Deflation Calibration Model."""

    def __init__(
        self,
        bridge_percentile: float = 0.70,
        loyalist_genre_threshold: float = 0.25,
        shrinkage_weight: float = 30.0,
        min_multiplier: float = 0.70,
        max_multiplier: float = 1.30,
    ) -> None:
        self.bridge_percentile = bridge_percentile
        self.loyalist_genre_threshold = loyalist_genre_threshold
        self.shrinkage_weight = shrinkage_weight
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

        self.user_entropy_df: Optional[pl.DataFrame] = None
        self.genre_multipliers_df: Optional[pl.DataFrame] = None
        self.genre_multiplier_map: Dict[str, float] = {}
        self.item_scores_df: Optional[pl.DataFrame] = None
        self.bridge_threshold: Optional[float] = None

    def fit(
        self,
        ratings_df: pl.DataFrame,
        movies_df: pl.DataFrame,
        verbose: bool = True,
    ) -> "BridgeCalibrationModel":
        """Fits user genre entropy and computes genre inflation multipliers."""
        global_mean = float(ratings_df["rating"].mean())

        # Explode genres per movie
        raw_movie_genres = (
            movies_df
            .with_columns(pl.col("genres").str.split("|"))
            .explode("genres")
            .filter((pl.col("genres") != "(no genres listed)") & (pl.col("genres") != ""))
            .select(["movieId", "genres"])
        )

        all_genres = sorted(raw_movie_genres["genres"].unique().to_list())
        num_genres = len(all_genres)
        if num_genres == 0:
            raise ValueError("No valid genres found in movies dataset.")

        genre_to_id = {g: i for i, g in enumerate(all_genres)}
        id_to_genre = {i: g for i, g in enumerate(all_genres)}

        # Encode genres as Int16 to prevent memory explosion during joins on 50M+ rows
        movie_genres = raw_movie_genres.with_columns(
            pl.col("genres").replace(genre_to_id).cast(pl.Int16).alias("gId")
        ).select(["movieId", "gId"])

        max_possible_entropy = np.log2(num_genres)

        # 1. Calculate Genre Entropism / Diversity Index for each user
        # Fast lazy group-by without materializing giant string tables
        user_genre_counts = (
            ratings_df.lazy().select(["userId", "movieId"])
            .join(movie_genres.lazy(), on="movieId", how="inner")
            .group_by(["userId", "gId"])
            .len()
            .rename({"len": "genre_count"})
            .collect()
        )

        user_totals = (
            user_genre_counts
            .group_by("userId")
            .agg(pl.col("genre_count").sum().alias("user_total_genres"))
        )

        user_genre_dist = (
            user_genre_counts
            .join(user_totals, on="userId")
            .with_columns((pl.col("genre_count") / pl.col("user_total_genres")).alias("p_ug"))
        )

        # Shannon entropy: - sum(p * log2(p))
        user_entropy = (
            user_genre_dist
            .group_by("userId")
            .agg([
                (- (pl.col("p_ug") * pl.col("p_ug").log(2)).sum()).alias("entropy"),
                pl.col("gId").n_unique().alias("num_genres_rated"),
            ])
            .with_columns(
                (pl.col("entropy") / max_possible_entropy).clip(0.0, 1.0).alias("diversity_index")
            )
        )
        self.user_entropy_df = user_entropy

        # 2. Identify Bridge Users vs. Genre Loyalists
        self.bridge_threshold = float(user_entropy["diversity_index"].quantile(self.bridge_percentile))

        bridge_user_ids = (
            user_entropy
            .filter(pl.col("diversity_index") >= self.bridge_threshold)
            .select("userId")
            .unique()
        )

        loyalist_user_genre = (
            user_genre_dist
            .join(user_entropy, on="userId")
            .filter(
                (pl.col("diversity_index") < self.bridge_threshold)
                & (pl.col("p_ug") >= self.loyalist_genre_threshold)
            )
            .select(["userId", "gId"])
        )

        if verbose:
            console.print(
                f"[bold cyan]Bridge Calibration[/bold cyan] (Diversity Threshold: {self.bridge_threshold:.3f} "
                f"[{self.bridge_percentile*100:.0f}th percentile]) | "
                f"Bridge Users: {bridge_user_ids.height:,} / {user_entropy.height:,} "
                f"({bridge_user_ids.height / user_entropy.height * 100:.1f}%)"
            )

        # 3. Vectorized Genre Rating Statistics using Integer IDs
        bridge_genre_stats = (
            ratings_df.lazy().select(["userId", "movieId", "rating"])
            .join(bridge_user_ids.lazy(), on="userId", how="semi")
            .join(movie_genres.lazy(), on="movieId", how="inner")
            .group_by("gId")
            .agg([
                pl.col("rating").mean().alias("bridge_mean"),
                pl.len().alias("bridge_count"),
            ])
            .collect()
        )

        loyalist_genre_stats = (
            ratings_df.lazy().select(["userId", "movieId", "rating"])
            .join(movie_genres.lazy(), on="movieId", how="inner")
            .join(loyalist_user_genre.lazy(), on=["userId", "gId"], how="semi")
            .group_by("gId")
            .agg([
                pl.col("rating").mean().alias("loyal_mean"),
                pl.len().alias("loyal_count"),
            ])
            .collect()
        )

        # Map back integer gId to genre string
        all_genres_df = pl.DataFrame({
            "gId": list(range(num_genres)),
            "genres": all_genres,
        }).with_columns(pl.col("gId").cast(pl.Int16))

        merged_genre_stats = (
            all_genres_df
            .join(bridge_genre_stats, on="gId", how="left")
            .join(loyalist_genre_stats, on="gId", how="left")
            .with_columns([
                pl.col("bridge_mean").fill_null(global_mean),
                pl.col("bridge_count").fill_null(0),
                pl.col("loyal_mean").fill_null(pl.col("bridge_mean").fill_null(global_mean)),
                pl.col("loyal_count").fill_null(0),
            ])
        )

        genre_records = []
        for row in merged_genre_stats.iter_rows(named=True):
            g = row["genres"]
            b_mean = float(row["bridge_mean"])
            b_count = int(row["bridge_count"])
            l_mean = float(row["loyal_mean"])
            l_count = int(row["loyal_count"])

            raw_mult = (b_mean / l_mean) if l_mean > 0 else 1.0
            shrunk_mult = (
                (l_count * raw_mult + self.shrinkage_weight * 1.0)
                / (l_count + self.shrinkage_weight)
            )
            final_mult = float(np.clip(shrunk_mult, self.min_multiplier, self.max_multiplier))

            genre_records.append({
                "genre": g,
                "bridge_mean": b_mean,
                "bridge_count": b_count,
                "loyalist_mean": l_mean,
                "loyalist_count": l_count,
                "raw_multiplier": raw_mult,
                "inflation_multiplier": final_mult,
                "inflation_pct": (1.0 - final_mult) * 100.0,
            })

        self.genre_multipliers_df = pl.DataFrame(genre_records).sort("inflation_multiplier")
        self.genre_multiplier_map = {
            row["genre"]: row["inflation_multiplier"] for row in genre_records
        }

        # 4. Item-Level Scoring: S_bridge(i) = RawAvg(i) * M_{g(i)}
        self._score_items(ratings_df, movies_df)
        return self

    def _score_items(self, ratings_df: pl.DataFrame, movies_df: pl.DataFrame) -> None:
        """Computes composite multiplier and calibrated score for each item."""
        item_stats = (
            ratings_df
            .group_by("movieId")
            .agg([
                pl.col("rating").mean().alias("raw_avg"),
                pl.col("rating").count().alias("vote_count"),
            ])
        )

        movies_with_stats = movies_df.join(item_stats, on="movieId", how="inner")

        def get_item_multiplier(genres_str: str) -> float:
            if not genres_str or genres_str == "(no genres listed)":
                return 1.0
            genres = genres_str.split("|")
            multipliers = [self.genre_multiplier_map.get(g, 1.0) for g in genres if g in self.genre_multiplier_map]
            return float(np.mean(multipliers)) if multipliers else 1.0

        item_multipliers = [
            get_item_multiplier(g_str) for g_str in movies_with_stats["genres"].to_list()
        ]

        scores_df = (
            movies_with_stats
            .with_columns(pl.Series("genre_multiplier", item_multipliers))
            .with_columns(
                (pl.col("raw_avg") * pl.col("genre_multiplier")).clip(1.0, 10.0).alias("score_model_c")
            )
            .sort("score_model_c", descending=True)
        )

        self.item_scores_df = scores_df

    def get_genre_multipliers(self) -> pl.DataFrame:
        """Returns the genre inflation multipliers DataFrame."""
        if self.genre_multipliers_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.genre_multipliers_df

    def get_item_scores(self) -> pl.DataFrame:
        """Returns the item calibrated scores DataFrame."""
        if self.item_scores_df is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.item_scores_df
