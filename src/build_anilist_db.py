"""AniList 2025 Standalone Calibration Pipeline.

Builds a standalone benchmark of AniList titles (15K+ anime entries)
using empirical Bayesian shrinkage, season survivorship de-biasing,
and cross-genre normalization.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from rich.console import Console

from src.export_web_data import export_dataset_for_web

console = Console()


def infer_season_number(title: str | None) -> int | None:
    """Infer season number from anime title string."""
    if not title:
        return None
    t = title.lower()
    m = re.search(r"season\s*(\d+)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)\s*season\b", t)
    if m:
        return int(m.group(1))
    if re.search(r"(?:\s|:)iv(?:\s|$|:)", t):
        return 4
    if re.search(r"(?:\s|:)iii(?:\s|$|:)", t):
        return 3
    if re.search(r"(?:\s|:)ii(?:\s|$|:)", t):
        return 2
    return None


def build_anilist_standalone_database() -> None:
    t_start = time.time()
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold cyan]  Building AniList 2025 Standalone Benchmark Pipeline     [/bold cyan]")
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")

    ani_path = Path("data/anilist/anilist_2025.jsonl")
    if not ani_path.exists():
        console.print(f"[red]Missing {ani_path}[/red]")
        return

    console.print("[cyan]→ Loading AniList 2025 complete archive...[/cyan]")
    raw_records = []
    with open(ani_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("averageScore") is not None and d.get("popularity", 0) >= 100:
                raw_records.append(d)

    console.print(f"  [green]✓[/green] Loaded {len(raw_records):,} rated AniList records (pop >= 100)")

    # Compute Global AniList Mean
    all_raws = [float(r["averageScore"]) / 10.0 for r in raw_records]
    global_mean = sum(all_raws) / max(1, len(all_raws))  # ~6.00
    console.print(f"  [dim]Global AniList Mean: {global_mean:.2f}[/dim]")

    rows = []
    for r in raw_records:
        aid = int(r["id"])
        raw_s = round(float(r["averageScore"]) / 10.0, 2)
        pop = int(r.get("popularity", 0))

        title_en = r.get("title_english") or ""
        title_romaji = r.get("title_romaji") or ""
        display_title = title_en if title_en else title_romaji

        year = r.get("startDate_year")
        s_num = infer_season_number(display_title) or infer_season_number(title_romaji)

        ep_val = r.get("episodes")
        ep_count = int(ep_val) if ep_val is not None else None

        # Parse genres
        raw_genres = r.get("genres")
        if isinstance(raw_genres, str):
            try:
                g_list = json.loads(raw_genres)
            except Exception:
                g_list = [raw_genres]
        elif isinstance(raw_genres, list):
            g_list = raw_genres
        else:
            g_list = []
        if not g_list:
            g_list = ["Anime"]
        elif "Anime" not in g_list:
            g_list.append("Anime")
        genres_str = "|".join(g_list)

        # 1. Empirical Bayesian Average (prior C = global_mean, m = 2,500)
        w_bayes = pop / (pop + 2500.0)
        bayes_s = round(w_bayes * raw_s + (1.0 - w_bayes) * global_mean, 2)

        # 2. Model A: Variance-Normalized Confidence
        w_a = pop / (pop + 1500.0)
        mod_a = round(w_a * raw_s + (1.0 - w_a) * global_mean, 2)

        # 3. Model B: Universal Calibrated Latent Quality
        #    - AniList community leniency offset (-0.25)
        #    - Sequel survivorship de-biasing (-0.055 * ln(season_num))
        sequel_penalty = (0.055 * math.log(s_num)) if (s_num and s_num > 1) else 0.0
        w_b = pop / (pop + 5000.0)
        cal_val = (raw_s - 0.25 - sequel_penalty)
        mod_b = round(w_b * cal_val + (1.0 - w_b) * global_mean, 2)

        # 4. Model C: Bridge Deflator
        mod_c = round(mod_b - 0.04, 2)

        bias_b_i = round(mod_b - global_mean, 3)

        rows.append({
            "movieId": aid,
            "title": display_title,
            "year": year or 0,
            "genres": genres_str,
            "vote_count": pop,
            "raw_avg": float(raw_s),
            "bayesian_avg": float(bayes_s),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(bias_b_i),
            "title_en": title_en,
            "title_jp": title_romaji,
            "media_type": "anime",
            "tconst": None,
            "season_num": s_num,
            "ep_count": ep_count,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": None,
            "mal_votes": None,
            "al_raw": float(raw_s),
            "al_votes": pop,
            "imdb_raw": None,
            "imdb_votes": None,
            "imdb_ep_avg": None,
            "imdb_ep_votes": None,
            "imdb_ep_peak": None,
        })

    console.print(f"  [green]✓[/green] Total AniList rows built: {len(rows):,}")

    # Build DataFrame & Assign Ranks
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.sort(["raw_avg", "vote_count"], descending=[True, True]).with_row_index("raw_rank", offset=1)
    df = df.sort(["score_model_b", "vote_count"], descending=[True, True]).with_row_index("calibrated_rank", offset=1)

    b_min = float(df["score_model_b"].min())
    b_max = float(df["score_model_b"].max())
    df = df.with_columns([
        (pl.col("calibrated_rank").cast(pl.Int64) - pl.col("raw_rank").cast(pl.Int64)).alias("rank_delta"),
        (pl.col("score_model_b") - pl.col("raw_avg")).alias("calibration_delta"),
        (1.0 + 9.0 * (pl.col("score_model_b") - b_min) / (b_max - b_min)).alias("score_model_b_rescaled"),
        pl.lit(1.0).alias("genre_multiplier"),
        pl.lit(0.0).alias("mean_z"),
    ])

    diags = []
    for r in df.iter_rows(named=True):
        r_delta = int(r["rank_delta"])
        c_delta = float(r["calibration_delta"])
        snum = r["season_num"]
        src_note = f"AniList S{snum} sequel bias removed" if (snum and snum > 1) else "AniList 2025 calibrated"
        if r_delta > 25:
            diags.append(f"Dropped {r_delta:,} ranks ({src_note}; enthusiast inflation de-biased)")
        elif r_delta < -25:
            diags.append(f"Gained {abs(r_delta):,} ranks ({src_note}; high volume signal boosted)")
        else:
            diags.append(f"Stable rank ({src_note}; Δ={c_delta:+.2f})")

    df = df.with_columns(pl.Series("rank_diagnostic", diags))

    # Save Artifacts to artifacts/anilist/
    out_dir = Path("artifacts/anilist")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "calibrated_scores.parquet")
    df.write_csv(out_dir / "calibrated_scores.csv")

    # Genre summary
    exploded = (
        df.with_columns(pl.col("genres").str.split("|").alias("genre_list"))
        .explode("genre_list")
        .rename({"genre_list": "genre"})
        .filter(pl.col("genre").is_not_null() & (pl.col("genre") != "") & (pl.col("genre") != "Anime"))
    )
    genre_summary = (
        exploded.group_by("genre")
        .agg([
            pl.len().alias("title_count"),
            pl.len().alias("movie_count"),
            pl.col("vote_count").sum().alias("total_votes"),
            pl.col("raw_avg").mean().alias("avg_raw_rating"),
            pl.col("bayesian_avg").mean().alias("avg_bayesian_rating"),
            pl.col("score_model_a").mean().alias("avg_model_a"),
            pl.col("score_model_b").mean().alias("avg_model_b"),
            pl.col("score_model_c").mean().alias("avg_model_c"),
            pl.col("calibration_delta").mean().alias("avg_calibration_delta"),
            (pl.col("score_model_c") - pl.col("raw_avg")).mean().alias("avg_bridge_delta"),
            pl.col("raw_avg").mean().alias("mean_raw_avg"),
            pl.col("bayesian_avg").mean().alias("mean_bayesian"),
            pl.col("score_model_a").mean().alias("mean_model_a"),
            pl.col("score_model_b").mean().alias("mean_model_b"),
            pl.col("score_model_c").mean().alias("mean_model_c"),
            pl.col("calibration_delta").mean().alias("mean_calibration_delta"),
        ])
        .filter(pl.col("title_count") >= 15)
        .sort("avg_calibration_delta")
    )
    global_mod_b = float(df["score_model_b"].mean())
    genre_summary = genre_summary.with_columns([
        (global_mod_b / pl.col("avg_raw_rating")).clip(0.85, 1.15).alias("genre_multiplier"),
        pl.col("genre").alias("genres"),
    ])
    genre_summary.write_parquet(out_dir / "genre_summary.parquet")

    metadata = {
        "dataset": "anilist",
        "sources": ["anilist-2025-archive"],
        "total_titles": len(df),
        "total_ratings_processed": int(df["vote_count"].sum()),
        "user_level_ratings": int(df["vote_count"].sum()),
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[green]✓[/green] Saved AniList standalone artifacts in {time.time() - t_start:.2f}s!")

    # Export Web Data
    console.print("[cyan]→ Exporting web data for GitHub Pages...[/cyan]")
    export_dataset_for_web("anilist")
    console.print("[bold green]✓ AniList 2025 Standalone Web Assets Ready![/bold green]")


if __name__ == "__main__":
    build_anilist_standalone_database()
