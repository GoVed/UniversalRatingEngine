"""UniversalRatingEngine - Central Analytical Engine Orchestrator.

Orchestrates:
- Data ingestion & cold-start filtering (Polars)
- Model A: User-Normalized Z-Score Aggregator
- Model B: Biased Matrix Factorization (PyTorch GPU/CPU)
- Model C: Bridge-User & Genre Deflation Multiplier
- IMDb Bayesian Average & Rank Shift Diagnostics
- Result persistence and fuzzy title inspection
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data_loader import get_data
from src.metrics import (
    compute_bayesian_average,
    compute_cross_genre_summary,
    format_rank_change,
)
from src.models.bridge_calibration import BridgeCalibrationModel
from src.models.matrix_factorization import BiasedMatrixFactorization
from src.models.z_score import UserNormalizedZScore

console = Console()


def format_display_title(title: str) -> str:
    """Formats inverted database titles ('Shawshank Redemption, The') into natural display ('The Shawshank Redemption')."""
    m_year = re.search(r"\s*\((\d{4})\)$", title)
    year_suffix = f" ({m_year.group(1)})" if m_year else ""
    t = re.sub(r"\s*\(\d{4}\)$", "", title).strip()

    m = re.match(
        r"^(.*?),\s*(The|A|An|Le|La|Les|L'|Der|Die|Das|Il|El|Los|Las)$",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        base = m.group(1).strip()
        prefix = m.group(2).strip()
        if prefix.endswith("'"):
            return f"{prefix}{base}{year_suffix}"
        return f"{prefix} {base}{year_suffix}"
    return f"{t}{year_suffix}"


class UniversalRatingEngine:
    """Production-grade engine computing cross-genre, uninflated quality scores."""

    def __init__(
        self,
        dataset: str = "ml-1m",
        artifacts_dir: str | Path = "artifacts",
        data_dir: str | Path = "data",
    ) -> None:
        self.dataset = dataset.lower().strip()
        self.artifacts_dir = Path(artifacts_dir) / self.dataset
        self.data_dir = Path(data_dir)

        self.calibrated_df: Optional[pl.DataFrame] = None
        self.genre_summary_df: Optional[pl.DataFrame] = None
        self.genre_multipliers_df: Optional[pl.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        min_user_ratings: int = 15,
        min_item_ratings: int = 30,
        k: int = 32,
        lambda_reg: float = 0.02,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 32768,
        bridge_percentile: float = 0.70,
        loyalist_genre_threshold: float = 0.25,
        device: str = "auto",
        verbose: bool = True,
    ) -> "UniversalRatingEngine":
        """Executes full end-to-end calibration pipeline across all three models."""
        console.rule(f"[bold green]UniversalRatingEngine: Fitting on {self.dataset}[/bold green]")

        if self.dataset == "combined":
            console.print(
                Panel(
                    "[bold yellow]COMBINED MULTI-COHORT DATASET NOTICE & DISCLAIMER[/bold yellow]\n\n"
                    "• [bold]Cohort Namespacing[/bold]: User IDs are offset by database (+1,000,000,000) to isolate individual leniency (b_u) without identity collisions.\n"
                    "• [bold]Interaction Deduplication[/bold]: Multi-vote ratings by the same rater on the same item are deduplicated by timestamp.\n"
                    "• [bold]Cold-Start Convergence[/bold]: Iterative dual-threshold filtering (min_user, min_item) is strictly enforced.\n"
                    "• [bold]Longitudinal Drift[/bold]: Latent factor decomposition isolates cross-cohort intrinsic quality (b_i), but merging distinct historical eras combines distinct cultural receptions.",
                    border_style="yellow",
                    title="Data Integrity Disclaimer",
                )
            )

        # 1. Ingest & Filter Data
        ratings_df, movies_df = get_data(
            dataset=self.dataset,
            data_dir=self.data_dir,
            min_user_ratings=min_user_ratings,
            min_item_ratings=min_item_ratings,
            scale_to_10=True,
        )

        global_mean = float(ratings_df["rating"].mean())
        global_std = float(ratings_df["rating"].std())

        if verbose:
            console.print(
                f"[dim]Dataset Statistics: Global Mean = {global_mean:.3f}, "
                f"Global Std = {global_std:.3f}[/dim]"
            )

        # 2. Fit Model A: User-Normalized Z-Score Aggregator
        console.print("[bold yellow]→[/bold yellow] Fitting [cyan]Model A: User-Normalized Z-Score[/cyan]...")
        model_a = UserNormalizedZScore(eps=1e-6)
        model_a.fit(ratings_df)
        scores_a = model_a.get_item_scores().select([
            "movieId",
            "mean_z",
            "score_model_a",
        ])

        # 3. Fit Model B: Biased Matrix Factorization
        console.print("[bold yellow]→[/bold yellow] Fitting [cyan]Model B: Biased Matrix Factorization (k={k})[/cyan]...")
        model_b = BiasedMatrixFactorization(
            k=k,
            lambda_reg=lambda_reg,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        model_b.fit(ratings_df, verbose=verbose)
        scores_b = model_b.get_item_scores().select([
            "movieId",
            "bias_b_i",
            "score_model_b",
            "score_model_b_rescaled",
        ])

        # 4. Fit Model C: Bridge-User & Genre Deflation Index
        console.print("[bold yellow]→[/bold yellow] Fitting [cyan]Model C: Bridge-User Calibration[/cyan]...")
        model_c = BridgeCalibrationModel(
            bridge_percentile=bridge_percentile,
            loyalist_genre_threshold=loyalist_genre_threshold,
        )
        model_c.fit(ratings_df, movies_df, verbose=verbose)
        scores_c = model_c.get_item_scores().select([
            "movieId",
            "genre_multiplier",
            "score_model_c",
        ])
        self.genre_multipliers_df = model_c.get_genre_multipliers()

        # 5. Raw Item Statistics & Bayesian Mean
        item_stats = (
            ratings_df
            .group_by("movieId")
            .agg([
                pl.col("rating").mean().alias("raw_avg"),
                pl.col("rating").count().alias("vote_count"),
            ])
            .with_columns(
                compute_bayesian_average(
                    pl.col("raw_avg"),
                    pl.col("vote_count"),
                    global_mean=global_mean,
                    min_votes_threshold=50,
                ).alias("bayesian_avg")
            )
        )

        # 6. Unify All Scores into Master Table
        master_df = (
            movies_df
            .join(item_stats, on="movieId", how="inner")
            .join(scores_a, on="movieId", how="inner")
            .join(scores_b, on="movieId", how="inner")
            .join(scores_c, on="movieId", how="inner")
        )

        # Calculate Rankings and Deltas
        # Raw Rank: sorted by raw_avg descending
        raw_rank_df = (
            master_df
            .select(["movieId", "raw_avg"])
            .sort(["raw_avg", "movieId"], descending=[True, False])
            .with_columns(
                pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("raw_rank")
            )
            .select(["movieId", "raw_rank"])
        )

        # Calibrated Rank: sorted by primary calibrated score (Model B) descending
        cal_rank_df = (
            master_df
            .select(["movieId", "score_model_b"])
            .sort(["score_model_b", "movieId"], descending=[True, False])
            .with_columns(
                pl.int_range(1, pl.len() + 1, dtype=pl.Int64).alias("calibrated_rank")
            )
            .select(["movieId", "calibrated_rank"])
        )

        calibrated_df = (
            master_df
            .join(raw_rank_df, on="movieId", how="inner")
            .join(cal_rank_df, on="movieId", how="inner")
            .with_columns([
                (pl.col("calibrated_rank") - pl.col("raw_rank")).alias("rank_delta"),
                (pl.col("score_model_b") - pl.col("raw_avg")).alias("calibration_delta"),
            ])
        )

        # Generate Human-Readable Rank Diagnostic strings
        diagnostics = [
            format_rank_change(r_rank, c_rank)
            for r_rank, c_rank in zip(
                calibrated_df["raw_rank"].to_list(),
                calibrated_df["calibrated_rank"].to_list(),
            )
        ]

        self.calibrated_df = (
            calibrated_df
            .with_columns(pl.Series("rank_diagnostic", diagnostics))
            .sort("score_model_b", descending=True)
        )

        # 7. Cross-Genre Aggregation
        self.genre_summary_df = compute_cross_genre_summary(self.calibrated_df)

        # 8. Save Artifacts
        self.metadata = {
            "dataset": self.dataset,
            "is_combined": self.dataset == "combined",
            "sources": ["ml-1m", "ml-latest-small", "mal-57m"] if self.dataset == "combined" else [self.dataset],
            "global_mean": global_mean,
            "global_std": global_std,
            "total_ratings": len(ratings_df),
            "total_titles": len(self.calibrated_df),
            "min_user_ratings": min_user_ratings,
            "min_item_ratings": min_item_ratings,
            "k": k,
            "lambda_reg": lambda_reg,
            "final_rmse": model_b.final_rmse,
            "bridge_threshold": model_c.bridge_threshold,
        }

        self._save_artifacts()
        console.print(f"[bold green]✓ Pipeline fit completed successfully for {self.dataset}![/bold green]")
        return self

    def _save_artifacts(self) -> None:
        """Persists trained engine artifacts to disk."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.calibrated_df is not None:
            self.calibrated_df.write_parquet(self.artifacts_dir / "calibrated_scores.parquet")
            self.calibrated_df.write_csv(self.artifacts_dir / "calibrated_scores.csv")

        if self.genre_summary_df is not None:
            self.genre_summary_df.write_parquet(self.artifacts_dir / "genre_summary.parquet")

        if self.genre_multipliers_df is not None:
            self.genre_multipliers_df.write_parquet(self.artifacts_dir / "genre_multipliers.parquet")

        with open(self.artifacts_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self) -> "UniversalRatingEngine":
        """Loads cached artifacts if available, otherwise runs fit()."""
        cal_parquet = self.artifacts_dir / "calibrated_scores.parquet"
        genre_parquet = self.artifacts_dir / "genre_summary.parquet"
        meta_json = self.artifacts_dir / "metadata.json"

        if cal_parquet.exists() and genre_parquet.exists() and meta_json.exists():
            self.calibrated_df = pl.read_parquet(cal_parquet)
            self.genre_summary_df = pl.read_parquet(genre_parquet)
            with open(meta_json, "r") as f:
                self.metadata = json.load(f)
            return self
        else:
            return self.fit()

    def search_title(self, query: str) -> List[Dict[str, Any]]:
        """Searches for movies by title using substring and fuzzy matching."""
        if self.calibrated_df is None:
            self.load()

        assert self.calibrated_df is not None
        titles_df = self.calibrated_df.select(["movieId", "title"]).to_dicts()
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Exact match (case-insensitive)
        exact_matches = [
            row for row in titles_df if row["title"].lower() == q_lower
        ]
        if exact_matches:
            return exact_matches

        # 2. Substring match
        sub_matches = [
            row for row in titles_df if q_lower in row["title"].lower()
        ]
        if sub_matches:
            return sub_matches

        # 3. Normalized article matching (e.g., "The Matrix" -> "Matrix, The")
        norm_q = re.sub(r"^(the|a|an)\s+", "", q_lower)
        article_matches = [
            row for row in titles_df if norm_q in row["title"].lower()
        ]
        if article_matches:
            return article_matches

        # 4. Fuzzy matching via RapidFuzz
        choices = {row["movieId"]: row["title"] for row in titles_df}
        fuzzy_results = process.extract(
            query,
            choices,
            scorer=fuzz.WRatio,
            limit=5,
        )

        matched_rows = []
        for match_title, score, mid in fuzzy_results:
            if score >= 50.0:
                matched_rows.append({"movieId": mid, "title": match_title, "score": score})

        return matched_rows

    def inspect(self, title_query: str) -> Optional[Dict[str, Any]]:
        """Inspects a specific title and outputs comprehensive calibration diagnostics."""
        if self.calibrated_df is None:
            self.load()

        assert self.calibrated_df is not None
        matches = self.search_title(title_query)

        if not matches:
            # Check if present in other dataset
            other_dataset = "ml-32m" if self.dataset == "ml-1m" else "ml-1m"
            console.print(
                f"[bold red]✗ Title '{title_query}' was not found in dataset '{self.dataset}'.[/bold red]"
            )
            other_path = self.data_dir / other_dataset / "movies.csv"
            if other_path.exists():
                other_movies = pl.read_csv(other_path)
                other_found = other_movies.filter(
                    pl.col("title").str.to_lowercase().str.contains(title_query.lower())
                )
                if other_found.height > 0:
                    sample_title = other_found["title"][0]
                    console.print(
                        f"[yellow]💡 Note:[/yellow] Found match in [bold cyan]{other_dataset}[/bold cyan]: "
                        f"[italic]'{sample_title}'[/italic].\n"
                        f"Run: [bold green]python run.py inspect \"{title_query}\" --dataset {other_dataset}[/bold green]"
                    )
            return None

        # Take top match
        target_mid = matches[0]["movieId"]
        item_row = self.calibrated_df.filter(pl.col("movieId") == target_mid).to_dicts()[0]

        # Extract values
        title = format_display_title(item_row["title"])
        genres = item_row["genres"]
        votes = item_row["vote_count"]
        raw_avg = item_row["raw_avg"]
        bayes_avg = item_row["bayesian_avg"]
        score_a = item_row["score_model_a"]
        bias_bi = item_row["bias_b_i"]
        score_b = item_row["score_model_b"]
        score_c = item_row["score_model_c"]
        g_mult = item_row["genre_multiplier"]
        raw_rank = item_row["raw_rank"]
        cal_rank = item_row["calibrated_rank"]
        rank_delta = item_row["rank_delta"]
        diag = item_row["rank_diagnostic"]

        # Render Rich Panel
        table = Table.grid(padding=(0, 2))
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="bold white")

        table.add_row("Movie ID:", f"{target_mid}")
        table.add_row("Genres:", genres)
        table.add_row("Ratings Count:", f"{votes:,} reviews")
        table.add_row("Dataset:", f"{self.dataset.upper()} (scale: 1.0 - 10.0)")
        table.add_row("", "")

        # Scoring metrics
        table.add_row("Raw Arithmetic Average:", f"[yellow]{raw_avg:.2f} / 10.0[/yellow]")
        table.add_row("IMDb / Bayesian Average:", f"{bayes_avg:.2f} / 10.0 (m=50 votes)")
        table.add_row("Model A (User Z-Score):", f"{score_a:.2f} / 10.0")
        table.add_row("Model B (Latent Quality b_i):", f"[bold green]{score_b:.2f} / 10.0[/bold green] (bias b_i: {bias_bi:+.3f})")
        table.add_row("Model C (Bridge Calibrated):", f"{score_c:.2f} / 10.0 (genre mult: {g_mult:.3f})")
        table.add_row("", "")

        # Rank metrics
        table.add_row("Raw Rank:", f"#{raw_rank:,} of {len(self.calibrated_df):,}")
        table.add_row("Calibrated Rank:", f"#{cal_rank:,} of {len(self.calibrated_df):,}")

        # Diagnosis styling
        if rank_delta > 0:
            diag_styled = f"[bold red]{diag}[/bold red]"
        elif rank_delta < 0:
            diag_styled = f"[bold green]{diag}[/bold green]"
        else:
            diag_styled = f"[dim]{diag}[/dim]"

        table.add_row("Overall Rank Shift:", diag_styled)

        # Calibration Delta
        cal_delta = score_b - raw_avg
        delta_str = f"{cal_delta:+.2f} pts"
        table.add_row("Calibration Delta (Δ):", f"[bold red]{delta_str}[/bold red]" if cal_delta < 0 else f"[bold green]{delta_str}[/bold green]")

        panel = Panel(
            table,
            title=f"[bold white]{title}[/bold white]",
            subtitle=f"[dim]UniversalRatingEngine v1.0 • {self.dataset}[/dim]",
            border_style="cyan",
            expand=False,
        )
        console.print(panel)

        if len(matches) > 1:
            alt_titles = [f"'{m['title']}'" for m in matches[1:4]]
            console.print(f"[dim]Other close matches: {', '.join(alt_titles)}[/dim]")

        return item_row
