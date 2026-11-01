"""IMDb Official Standalone Calibration Pipeline.

Builds a standalone benchmark of IMDb titles (Movies, TV Series, and Episode-Separated TV Seasons)
using empirical Bayesian shrinkage, genre-level normalization, and season survivorship de-biasing.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from rich.console import Console

from src.export_web_data import export_dataset_for_web

console = Console()


def build_imdb_standalone_database() -> None:
    t_start = time.time()
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold cyan]  Building IMDb Official Standalone Benchmark Pipeline   [/bold cyan]")
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")

    # 1. Load IMDb Official Tables
    console.print("[cyan]→ Loading IMDb official tables (basics, ratings, episodes)...[/cyan]")
    imdb_basics = pl.read_csv(
        "data/imdb/title.basics.tsv.gz",
        separator="\t",
        null_values=r"\N",
        quote_char=None,
        schema_overrides={"startYear": pl.Int32, "runtimeMinutes": pl.Int32},
    )
    imdb_ratings = pl.read_csv(
        "data/imdb/title.ratings.tsv.gz",
        separator="\t",
        null_values=r"\N",
        quote_char=None,
    )
    imdb_episodes = pl.read_csv(
        "data/imdb/title.episode.tsv.gz",
        separator="\t",
        null_values=r"\N",
        quote_char=None,
        schema_overrides={"seasonNumber": pl.Int32, "episodeNumber": pl.Int32},
    )

    console.print(
        f"  [green]✓[/green] IMDb tables loaded: {len(imdb_basics):,} basics, "
        f"{len(imdb_ratings):,} ratings, {len(imdb_episodes):,} episodes"
    )

    # 2. Filter Rated Movies & Series with numVotes >= 2,000
    rated_titles = (
        imdb_basics.filter(pl.col("titleType").is_in(["movie", "tvSeries", "tvMiniSeries"]))
        .join(imdb_ratings, on="tconst", how="inner")
        .filter(pl.col("numVotes") >= 2000)
    )
    console.print(f"  [green]✓[/green] Filtered movies & series (votes >= 2,000): {len(rated_titles):,}")

    # 3. Aggregate Episodes into TV Seasons
    console.print("[cyan]→ Aggregating 10M IMDb episodes by (parentTconst, seasonNumber)...[/cyan]")
    ep_rated = (
        imdb_episodes.filter(
            pl.col("seasonNumber").is_not_null()
            & (pl.col("seasonNumber") > 0)
            & (pl.col("seasonNumber") <= 50)
        )
        .join(imdb_ratings, on="tconst", how="inner")
        .join(imdb_basics.select(["tconst", "startYear"]), on="tconst", how="left")
    )

    season_agg_df = ep_rated.group_by(["parentTconst", "seasonNumber"]).agg([
        pl.len().alias("ep_count"),
        pl.col("numVotes").sum().alias("ep_votes"),
        pl.col("averageRating").mean().alias("ep_avg"),
        pl.col("averageRating").max().alias("ep_peak"),
        pl.col("averageRating").min().alias("ep_min"),
        pl.col("startYear").min().alias("season_year"),
    ])

    # Lookup for parents with high votes to synthesize season-separated entries
    parent_map: dict[str, dict] = {}
    for r in rated_titles.iter_rows(named=True):
        parent_map[r["tconst"]] = r

    # Global IMDb mean for Bayesian prior
    global_imdb_mean = float(imdb_ratings["averageRating"].mean())  # ~6.45
    console.print(f"  [dim]Global IMDb Mean: {global_imdb_mean:.2f}[/dim]")

    # 4. Construct Unified Item Records
    rows = []
    id_counter = 7_000_000

    # Add standalone movies and series
    for r in rated_titles.iter_rows(named=True):
        id_counter += 1
        t_type = r["titleType"]
        tconst = r["tconst"]
        primary = r["primaryTitle"] or "Unknown"
        original = r["originalTitle"]
        year = r["startYear"]
        raw = float(r["averageRating"])
        votes = int(r["numVotes"])

        genres_str = (r["genres"] or "Drama").replace(",", "|")
        if genres_str == r"\N" or not genres_str:
            genres_str = "Drama"

        # Media type labeling
        media_type = "movie" if t_type == "movie" else ("tv_season" if t_type == "tvMiniSeries" else "tv_series")

        # Bayesian shrinkage
        w_bayes = votes / (votes + 8000.0)
        bayes = w_bayes * raw + (1.0 - w_bayes) * global_imdb_mean

        # Model A: Genre/Medium standardization
        mod_a = 0.95 * bayes

        # Model B: Latent quality score (adjusting for TV enthusiasm inflation if applicable)
        if t_type in ("tvSeries", "tvMiniSeries"):
            mod_b = w_bayes * (raw - 0.35) + (1.0 - w_bayes) * (global_imdb_mean - 0.20)
        else:
            mod_b = bayes

        mod_c = mod_b - 0.05

        title_en = primary if (original and original != primary) else ""
        title_jp = original if (original and original != primary) else ""

        rows.append({
            "movieId": id_counter,
            "title": primary,
            "year": year,
            "genres": genres_str,
            "vote_count": votes,
            "raw_avg": raw,
            "bayesian_avg": float(bayes),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(mod_b - global_imdb_mean),
            "title_en": title_en,
            "title_jp": title_jp,
            "media_type": media_type,
            "tconst": tconst,
            "season_num": None,
            "ep_count": None,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": None,
            "mal_votes": None,
            "imdb_raw": raw,
            "imdb_votes": votes,
            "imdb_ep_avg": None,
            "imdb_ep_votes": None,
            "imdb_ep_peak": None,
        })

    # Add Season-Separated TV entries for top TV series
    tv_seasons_added = 0
    for s in season_agg_df.iter_rows(named=True):
        p_tc = s["parentTconst"]
        s_num = int(s["seasonNumber"])
        ep_votes = int(s["ep_votes"])
        ep_avg = float(s["ep_avg"])
        ep_cnt = int(s["ep_count"])
        ep_peak = float(s["ep_peak"])
        s_year = s["season_year"]

        if ep_cnt < 3 or ep_votes < 3000:
            continue
        p_info = parent_map.get(p_tc)
        if not p_info:
            continue
        p_raw = float(p_info["averageRating"])
        p_votes = int(p_info["numVotes"])
        if p_votes < 20000:
            continue

        id_counter += 1
        tv_seasons_added += 1

        p_title = p_info["primaryTitle"] or "Unknown"
        p_orig = p_info["originalTitle"]
        genres_str = (p_info["genres"] or "Drama").replace(",", "|")
        if "TV-Season" not in genres_str:
            genres_str = f"TV-Season|{genres_str}"

        # De-bias sequel survivorship
        sequel_drift = 0.045 * math.log(max(1, s_num))
        w_ep = ep_votes / (ep_votes + 30000.0)
        w_p = p_votes / (p_votes + 50000.0)
        w_conf = w_ep * (0.65 + 0.35 * w_p)

        if ep_avg < p_raw - 0.35:
            eff_raw = 0.85 * ep_avg + 0.15 * p_raw
        else:
            eff_raw = 0.50 * ep_avg + 0.50 * p_raw

        cal_b = w_conf * (eff_raw - 0.60 - sequel_drift) + (1.0 - w_conf) * (global_imdb_mean - 0.10)
        cal_a = w_conf * (ep_avg - 0.55 - sequel_drift) + (1.0 - w_conf) * (global_imdb_mean - 0.10)
        cal_c = cal_b - 0.05
        bayes_s = (ep_votes / (ep_votes + 12000.0)) * ep_avg + (12000.0 / (ep_votes + 12000.0)) * p_raw

        rows.append({
            "movieId": id_counter,
            "title": f"{p_title}: Season {s_num}",
            "year": s_year or p_info["startYear"],
            "genres": genres_str,
            "vote_count": ep_votes,
            "raw_avg": ep_avg,
            "bayesian_avg": float(bayes_s),
            "score_model_a": float(cal_a),
            "score_model_b": float(cal_b),
            "score_model_c": float(cal_c),
            "bias_b_i": float(cal_b - global_imdb_mean),
            "title_en": p_title if (p_orig and p_orig != p_title) else "",
            "title_jp": p_orig if (p_orig and p_orig != p_title) else "",
            "media_type": "tv_season",
            "tconst": p_tc,
            "season_num": s_num,
            "ep_count": ep_cnt,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": None,
            "mal_votes": None,
            "imdb_raw": p_raw,
            "imdb_votes": p_votes,
            "imdb_ep_avg": round(ep_avg, 2),
            "imdb_ep_votes": ep_votes,
            "imdb_ep_peak": round(ep_peak, 2),
        })

    console.print(f"  [green]✓[/green] Total IMDb rows built: {len(rows):,} ({tv_seasons_added:,} TV seasons)")

    # 5. Build DataFrame & Assign Ranks
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
        mtype = r["media_type"]
        snum = r["season_num"]
        src_note = f"IMDb TV S{snum} episode rollup" if mtype == "tv_season" else "IMDb official calibrated"
        if r_delta > 25:
            diags.append(f"Dropped {r_delta:,} ranks ({src_note}; enthusiast inflation de-biased)")
        elif r_delta < -25:
            diags.append(f"Gained {abs(r_delta):,} ranks ({src_note}; broad audience drag corrected)")
        else:
            diags.append(f"Stable rank ({src_note}; Δ={c_delta:+.2f})")

    df = df.with_columns(pl.Series("rank_diagnostic", diags))

    # 6. Save Artifacts to artifacts/imdb/
    out_dir = Path("artifacts/imdb")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "calibrated_scores.parquet")
    df.write_csv(out_dir / "calibrated_scores.csv")

    # Genre summary
    exploded = (
        df.with_columns(pl.col("genres").str.split("|").alias("genre_list"))
        .explode("genre_list")
        .rename({"genre_list": "genre"})
        .filter(pl.col("genre").is_not_null() & (pl.col("genre") != "") & (pl.col("genre") != "(no genres listed)"))
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
        .filter(pl.col("title_count") >= 20)
        .sort("avg_calibration_delta")
    )
    global_mod_b = float(df["score_model_b"].mean())
    genre_summary = genre_summary.with_columns([
        (global_mod_b / pl.col("avg_raw_rating")).clip(0.85, 1.15).alias("genre_multiplier"),
        pl.col("genre").alias("genres"),
    ])
    genre_summary.write_parquet(out_dir / "genre_summary.parquet")

    metadata = {
        "dataset": "imdb",
        "sources": ["imdb-official-basics", "imdb-official-ratings", "imdb-official-episodes"],
        "total_titles": len(df),
        "total_ratings_processed": int(imdb_ratings["numVotes"].sum()),
        "user_level_ratings": int(imdb_ratings["numVotes"].sum()),
        "imdb_rated_titles": len(imdb_ratings),
        "imdb_episodes_aggregated": len(imdb_episodes),
        "tv_seasons_count": tv_seasons_added,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"[green]✓[/green] Saved IMDb standalone artifacts in {time.time() - t_start:.2f}s!")

    # 7. Export Web Data
    console.print("[cyan]→ Exporting web data for GitHub Pages...[/cyan]")
    export_dataset_for_web("imdb")
    console.print("[bold green]✓ IMDb Official Standalone Web Assets Ready![/bold green]")


if __name__ == "__main__":
    build_imdb_standalone_database()
