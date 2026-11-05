"""Douban Official (豆瓣电影) Standalone Calibration Pipeline.

Builds a standalone benchmark of Douban titles (30K+ Chinese & International Cinema / TV entries)
using empirical Bayesian shrinkage, genre-level de-biasing, star-distribution calibration,
and cross-platform linking with IMDb.
"""

from __future__ import annotations

import datetime
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

DOUBAN_GENRE_MAP = {
    "剧情": "Drama",
    "喜剧": "Comedy",
    "爱情": "Romance",
    "动作": "Action",
    "惊悚": "Thriller",
    "犯罪": "Crime",
    "短片": "Short",
    "动画": "Animation",
    "恐怖": "Horror",
    "冒险": "Adventure",
    "悬疑": "Mystery",
    "奇幻": "Fantasy",
    "纪录片": "Documentary",
    "科幻": "Sci-Fi",
    "家庭": "Family",
    "传记": "Biography",
    "历史": "History",
    "战争": "War",
    "音乐": "Music",
    "同性": "LGBTQ+",
    "情色": "Erotica",
    "古装": "Costume",
    "歌舞": "Musical",
    "运动": "Sport",
    "武侠": "Wuxia",
    "西部": "Western",
    "戏曲": "Opera",
    "儿童": "Children",
    "真人秀": "Reality-TV",
    "脱口秀": "Talk-Show",
    "黑色电影": "Film-Noir",
    "灾难": "Disaster",
}


def parse_douban_genres(raw_genre_str: str | None) -> str:
    """Map slash-separated Chinese genres to standardized pipe-separated English genres."""
    if not raw_genre_str:
        return "Drama"
    mapped = []
    for piece in raw_genre_str.split("/"):
        p = piece.strip()
        if not p:
            continue
        eng = DOUBAN_GENRE_MAP.get(p, p)
        if eng not in mapped:
            mapped.append(eng)
    return "|".join(mapped) if mapped else "Drama"


def extract_year(date_str: str | None) -> int:
    """Extract 4-digit release year from Douban release date string."""
    if not date_str:
        return 0
    m = re.search(r"(\d{4})", str(date_str))
    return int(m.group(1)) if m else 0


def build_douban_standalone_database() -> None:
    t_start = time.time()
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold cyan]  Building Douban 豆瓣 Standalone Benchmark Pipeline     [/bold cyan]")
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")

    douban_path = Path("data/douban/douban_movie_info.csv")
    if not douban_path.exists():
        console.print(f"[red]Missing {douban_path}. Please download first.[/red]")
        return

    console.print("[cyan]→ Loading Douban complete archive...[/cyan]")
    raw_df = pl.read_csv(douban_path, ignore_errors=True)
    console.print(f"  [green]✓[/green] Loaded {len(raw_df):,} raw Douban rows")

    # Filter valid scores & votes >= 30
    valid_df = raw_df.filter(
        pl.col("score").is_not_null()
        & pl.col("score_amt").is_not_null()
        & (pl.col("score_amt") >= 30)
    )
    console.print(f"  [green]✓[/green] Filtered {len(valid_df):,} valid rated titles (votes >= 30)")

    # Load IMDb basics if present to cross-match English titles & media types
    imdb_basics_path = Path("data/imdb/title.basics.tsv.gz")
    imdb_map = {}
    if imdb_basics_path.exists():
        console.print("[cyan]→ Cross-referencing IMDb official basics table for English titles...[/cyan]")
        basics_df = pl.read_csv(
            imdb_basics_path,
            separator="\t",
            null_values=r"\N",
            quote_char=None,
            columns=["tconst", "titleType", "primaryTitle", "originalTitle", "startYear"],
        )
        for row in basics_df.iter_rows(named=True):
            imdb_map[row["tconst"]] = (
                row["primaryTitle"],
                row["originalTitle"],
                row["titleType"],
                row["startYear"],
            )
        console.print(f"  [green]✓[/green] Indexed {len(imdb_map):,} IMDb basics entries")

    # Compute Global Douban Statistics
    all_scores = valid_df["score"].to_list()
    global_mean = sum(all_scores) / max(1, len(all_scores))  # ~6.95
    console.print(f"  [dim]Global Douban Mean Score: {global_mean:.3f}[/dim]")

    # Pass 1: Compute Genre Means for Bias Correction
    genre_totals = {}
    genre_counts = {}
    temp_records = []
    for row in valid_df.iter_rows(named=True):
        raw_s = float(row["score"])
        votes = int(row["score_amt"])
        genres_str = parse_douban_genres(row.get("类型"))
        g_list = [g.strip() for g in genres_str.split("|") if g.strip()]
        for g in g_list:
            genre_totals[g] = genre_totals.get(g, 0.0) + raw_s
            genre_counts[g] = genre_counts.get(g, 0) + 1
        temp_records.append((row, raw_s, votes, genres_str, g_list))

    genre_means = {g: genre_totals[g] / max(1, genre_counts[g]) for g in genre_totals}

    # Pass 2: Calibrate Titles (Empirical Bayes + Latent Model B + Platform Leniency Offset)
    c_prior = 10000.0  # Bayesian prior credibility weight
    douban_leniency_offset = 0.58  # Empirical leniency offset of Douban vs Universal/MovieLens scale
    rows = []
    for row, raw_s, votes, genres_str, g_list in temp_records:
        mid = int(row["movie_id"])
        movie_name_cn = str(row.get("movie_name") or "")
        imdb_id = str(row.get("IMDb") or "").strip()
        if not imdb_id.startswith("tt"):
            imdb_id = None

        # Cross-matched IMDb metadata
        imdb_meta = imdb_map.get(imdb_id) if imdb_id else None
        title_en = imdb_meta[0] if imdb_meta else (str(row.get("又名") or "").split("/")[0].strip() or movie_name_cn)
        title_orig = imdb_meta[1] if imdb_meta else None
        raw_type = imdb_meta[2] if imdb_meta else None
        imdb_year = imdb_meta[3] if imdb_meta else None

        # Determine authentic original language title
        if title_orig and title_en and title_orig.strip().lower() == title_en.strip().lower():
            # Movie was originally in English/Western language (e.g. Inception, Shawshank, Godfather)
            title_jp = ""
        elif title_orig and title_en and title_orig.strip().lower() != title_en.strip().lower():
            # Foreign film with authentic original title distinct from English title
            if movie_name_cn and re.search(r"[\u4e00-\u9fff]", movie_name_cn) and (
                " " not in title_orig
                or any(p in title_orig.lower() for p in ["ba wang", "cha guan", "wo hu", "gei", "de", "zi", "ji", "hua"])
            ):
                title_jp = movie_name_cn
            else:
                title_jp = title_orig
        else:
            title_jp = movie_name_cn if not title_en or title_en == movie_name_cn else ""

        # Determine media type
        if raw_type:
            media_type = "anime" if "Animation" in genres_str and raw_type in ["movie", "tvSeries", "tvMiniSeries"] else (
                "tv_series" if "tv" in raw_type.lower() else "movie"
            )
        else:
            media_type = "anime" if "Animation" in genres_str else "movie"

        # Determine release year
        year = imdb_year or extract_year(row.get("上映日期"))

        # Display Title: Chinese title with English subtitle when distinct
        if title_en and title_en.lower() != movie_name_cn.lower() and not re.search(r"[\u4e00-\u9fff]", title_en):
            display_title = f"{movie_name_cn} ({title_en})"
        else:
            display_title = movie_name_cn

        # 1. Bayesian Credibility Shrinkage (Shrink raw rating towards global mean)
        w_bayes = votes / (votes + c_prior)
        bayes_s = round(w_bayes * raw_s + (1.0 - w_bayes) * global_mean, 2)

        # 2. Model A: Linear shrinkage of de-biased score
        w_b = votes / (votes + 30000.0)
        raw_debiased = raw_s - douban_leniency_offset
        mod_a = round(w_b * raw_debiased + (1.0 - w_b) * 6.80, 2)

        # 3. Model B: Genre Latent Quality Calibration
        genre_bias_offsets = [genre_means.get(g, global_mean) - global_mean for g in g_list]
        avg_genre_bias = sum(genre_bias_offsets) / max(1, len(genre_bias_offsets)) if genre_bias_offsets else 0.0

        # Adjust score for genre inflation / deflation
        calibrated_val = raw_debiased - 0.35 * avg_genre_bias
        mod_b = round(w_b * calibrated_val + (1.0 - w_b) * 6.80, 2)

        # 4. Model C: Bridge Deflator
        mod_c = round(mod_b - 0.04, 2)

        bias_b_i = round(mod_b - 6.80, 3)

        rows.append({
            "movieId": mid,
            "title": display_title,
            "year": year or 0,
            "genres": genres_str,
            "vote_count": votes,
            "raw_avg": float(raw_s),
            "bayesian_avg": float(bayes_s),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(bias_b_i),
            "title_en": title_en,
            "title_jp": title_jp,
            "media_type": media_type,
            "tconst": imdb_id,
            "season_num": None,
            "ep_count": None,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": None,
            "mal_votes": None,
            "al_raw": None,
            "al_votes": None,
            "imdb_raw": None,
            "imdb_votes": None,
            "imdb_ep_avg": None,
            "imdb_ep_votes": None,
            "imdb_ep_peak": None,
        })

    console.print(f"  [green]✓[/green] Total Douban calibrated rows built: {len(rows):,}")

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
        src_note = "Douban 豆瓣 calibrated"
        if r_delta > 35:
            diags.append(f"Dropped {r_delta:,} ranks ({src_note}; niche inflation de-biased)")
        elif r_delta < -35:
            diags.append(f"Gained {abs(r_delta):,} ranks ({src_note}; high volume consensus boosted)")
        else:
            diags.append(f"Stable rank ({src_note}; Δ={c_delta:+.2f})")

    df = df.with_columns(pl.Series("rank_diagnostic", diags))

    # Save Artifacts to artifacts/douban/
    out_dir = Path("artifacts/douban")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "calibrated_scores.parquet")
    df.write_csv(out_dir / "calibrated_scores.csv")
    console.print(f"  [green]✓[/green] Wrote artifacts to {out_dir}")

    # Build Genre Summary
    exploded = (
        df.with_columns(pl.col("genres").str.split("|").alias("genre_list"))
        .explode("genre_list")
        .rename({"genre_list": "genre"})
        .filter(pl.col("genre").is_not_null() & (pl.col("genre") != ""))
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
            (pl.col("score_model_b") - pl.col("raw_avg")).mean().alias("bias_correction"),
            pl.lit(1.0).alias("genre_multiplier"),
            pl.lit(0.0).alias("mean_z"),
        ])
        .sort("total_votes", descending=True)
    )
    genre_summary.write_parquet(out_dir / "genre_summary.parquet")

    # Metadata
    total_votes = int(df["vote_count"].sum())
    meta = {
        "dataset": "douban",
        "sources": ["douban-movie-info-hf"],
        "total_titles": len(df),
        "total_ratings_processed": total_votes,
        "user_level_ratings": total_votes,
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Export for Web
    console.print("[cyan]→ Exporting Web Assets for Douban...[/cyan]")
    export_dataset_for_web("douban")

    console.print(f"[bold green]✓ Douban Benchmark Built & Exported in {time.time() - t_start:.2f}s![/bold green]")


if __name__ == "__main__":
    build_douban_standalone_database()
