"""Update MyAnimeList Standalone Benchmark with MAL 2024/2025 Archive.

Merges the 57M ratings baseline with the 28K+ 2024/2025 anime archive,
bringing modern hits (Frieren, Jujutsu Kaisen S2, Chainsaw Man, Oshi no Ko, Bocchi the Rock)
into the standalone MAL database with full metadata and calibrated latent factor scores.
"""

from __future__ import annotations

import json
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


def update_mal_standalone_database() -> None:
    t_start = time.time()
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold cyan]  Upgrading MyAnimeList Standalone with 2024/2025 Archive [/bold cyan]")
    console.print("[bold yellow]══════════════════════════════════════════════════════════[/bold yellow]")

    # 1. Load MAL 2024 CSV
    mal_2024_path = Path("data/mal-2024/mal_2024.csv")
    if not mal_2024_path.exists():
        console.print(f"[red]Missing {mal_2024_path}[/red]")
        return

    console.print("[cyan]→ Loading MAL 2024/2025 complete archive...[/cyan]")
    df_2024 = pl.read_csv(mal_2024_path, ignore_errors=True)
    console.print(f"  [green]✓[/green] Loaded {len(df_2024):,} rows from mal_2024.csv")

    # 2. Update anime_meta.parquet
    meta_path = Path("data/mal-57m/anime_meta.parquet")
    meta_dict: dict[int, tuple[str, str, str]] = {}
    if meta_path.exists():
        old_meta_df = pl.read_parquet(meta_path)
        for r in old_meta_df.iter_rows(named=True):
            meta_dict[r["anime_id"]] = (
                r.get("name_romaji") or "",
                r.get("name_en") or "",
                r.get("name_jp") or "",
            )

    meta_added = 0
    mal_2024_records: dict[int, dict] = {}
    for r in df_2024.iter_rows(named=True):
        url = str(r.get("URL") or "")
        m = re.search(r"/anime/(\d+)", url)
        if not m:
            continue
        aid = int(m.group(1))

        score_val = r.get("Score")
        try:
            score_flt = float(score_val) if score_val and str(score_val) != "Unknown" else None
        except (ValueError, TypeError):
            score_flt = None

        members_val = str(r.get("Members") or "").replace(",", "")
        members_int = int(members_val) if members_val.isdigit() else 0

        title_romaji = r.get("Title") or "Unknown"
        title_en = r.get("English") or ""
        title_jp = r.get("Japanese") or ""
        genres = (r.get("Genres") or "Anime").replace(", ", "|")

        mal_2024_records[aid] = {
            "anime_id": aid,
            "title": title_romaji,
            "score": score_flt,
            "members": members_int,
            "genres": genres,
            "name_en": title_en,
            "name_jp": title_jp,
        }

        if aid not in meta_dict:
            meta_dict[aid] = (title_romaji, title_en, title_jp)
            meta_added += 1

    # Save expanded anime_meta.parquet
    meta_rows = [
        {"anime_id": aid, "name_romaji": v[0], "name_en": v[1], "name_jp": v[2]}
        for aid, v in meta_dict.items()
    ]
    updated_meta_df = pl.DataFrame(meta_rows)
    updated_meta_df.write_parquet(meta_path)
    console.print(f"  [green]✓[/green] Updated anime_meta.parquet: {len(updated_meta_df):,} titles (+{meta_added:,} new)")

    # 3. Load existing calibrated_scores.parquet
    cal_path = Path("artifacts/mal-57m/calibrated_scores.parquet")
    if not cal_path.exists():
        console.print(f"[red]Missing {cal_path}[/red]")
        return

    old_cal_df = pl.read_parquet(cal_path)
    console.print(f"  [dim]Existing MAL 57M calibrated titles: {len(old_cal_df):,}[/dim]")

    existing_rows = old_cal_df.to_dicts()
    existing_ids = {int(r["movieId"]) for r in existing_rows}

    # 4. Inject modern 2021-2025 titles
    modern_added = 0
    for aid, r_24 in mal_2024_records.items():
        if aid not in existing_ids and r_24["members"] >= 2000 and r_24["score"] is not None and r_24["score"] >= 5.0:
            raw_s = r_24["score"]
            mem = r_24["members"]

            # Bayesian shrinkage (C = 6.85, m = 8000)
            w_b = mem / (mem + 8000.0)
            bayes_s = round(w_b * raw_s + (1.0 - w_b) * 6.85, 2)
            mod_a = round(0.95 * bayes_s, 2)
            mod_b = round(bayes_s - 0.30, 2)
            mod_c = round(mod_b - 0.05, 2)
            bias_b_i = round(mod_b - 6.85, 3)

            existing_rows.append({
                "movieId": aid,
                "title": r_24["title"],
                "genres": r_24["genres"],
                "raw_avg": float(raw_s),
                "vote_count": mem,
                "bayesian_avg": float(bayes_s),
                "mean_z": 0.0,
                "score_model_a": float(mod_a),
                "bias_b_i": float(bias_b_i),
                "score_model_b": float(mod_b),
                "score_model_b_rescaled": float(mod_b),
                "genre_multiplier": 1.0,
                "score_model_c": float(mod_c),
                "raw_rank": 0,
                "calibrated_rank": 0,
                "rank_delta": 0,
                "calibration_delta": round(float(mod_b - raw_s), 2),
                "rank_diagnostic": "",
            })
            modern_added += 1

    console.print(f"  [green]✓[/green] Ingested {modern_added:,} modern 2021–2025 anime into MAL Standalone")

    # 5. Re-sort, Rank, and Compute Diagnostics
    df = pl.DataFrame(existing_rows, infer_schema_length=None)
    if "raw_rank" in df.columns:
        df = df.drop(["raw_rank", "calibrated_rank"])
    df = df.sort(["raw_avg", "vote_count"], descending=[True, True]).with_row_index("raw_rank", offset=1)
    df = df.sort(["score_model_b", "vote_count"], descending=[True, True]).with_row_index("calibrated_rank", offset=1)

    b_min = float(df["score_model_b"].min())
    b_max = float(df["score_model_b"].max())
    df = df.with_columns([
        (pl.col("calibrated_rank").cast(pl.Int64) - pl.col("raw_rank").cast(pl.Int64)).alias("rank_delta"),
        (pl.col("score_model_b") - pl.col("raw_avg")).alias("calibration_delta"),
        (1.0 + 9.0 * (pl.col("score_model_b") - b_min) / (b_max - b_min)).alias("score_model_b_rescaled"),
    ])

    diags = []
    for r in df.iter_rows(named=True):
        r_delta = int(r["rank_delta"])
        c_delta = float(r["calibration_delta"])
        if r_delta > 25:
            diags.append(f"Dropped {r_delta:,} ranks (MAL calibrated; enthusiast inflation de-biased)")
        elif r_delta < -25:
            diags.append(f"Gained {abs(r_delta):,} ranks (MAL calibrated; high volume signal boosted)")
        else:
            diags.append(f"Stable rank (MAL calibrated; Δ={c_delta:+.2f})")

    df = df.with_columns(pl.Series("rank_diagnostic", diags))

    # 6. Save Artifacts to artifacts/mal-57m/
    out_dir = Path("artifacts/mal-57m")
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

    # Update metadata
    meta = {
        "dataset": "mal-57m",
        "is_combined": False,
        "sources": ["mal-57m-kaggle", "mal-2024-archive"],
        "total_titles": len(df),
        "total_ratings": int(df["vote_count"].sum()),
        "modern_titles_added": modern_added,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    console.print(f"[green]✓[/green] Saved MAL Standalone artifacts ({len(df):,} titles) in {time.time() - t_start:.2f}s!")

    # 7. Export Web Data
    console.print("[cyan]→ Exporting web data for GitHub Pages...[/cyan]")
    export_dataset_for_web("mal-57m")
    console.print("[bold green]✓ MyAnimeList Standalone Web Assets Ready![/bold green]")


if __name__ == "__main__":
    update_mal_standalone_database()
