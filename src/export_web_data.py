"""Export complete calibrated datasets and metadata for the GitHub Pages web UI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
from rich.console import Console

console = Console()


def extract_year(title: str) -> int:
    """Extract 4-digit release year from title string."""
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else 0


def clean_movie_title(title: str) -> str:
    """Removes trailing year and uninverts articles (e.g. 'Shawshank Redemption, The' -> 'The Shawshank Redemption')."""
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
            return f"{prefix}{base}"
        return f"{prefix} {base}"
    return t


def export_dataset_for_web(
    dataset_name: str,
    artifacts_dir: str | Path = "artifacts",
    output_dir: str | Path = "docs/data",
) -> None:
    artifacts_path = Path(artifacts_dir) / dataset_name
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cal_file = artifacts_path / "calibrated_scores.parquet"
    genre_file = artifacts_path / "genre_summary.parquet"
    mult_file = artifacts_path / "genre_multipliers.parquet"
    meta_file = artifacts_path / "metadata.json"

    if not cal_file.exists():
        console.print(f"[red]Missing artifacts for {dataset_name}. Run fit first.[/red]")
        return

    df = pl.read_parquet(cal_file)
    genre_df = pl.read_parquet(genre_file) if genre_file.exists() else None
    mult_df = pl.read_parquet(mult_file) if mult_file.exists() else None

    # Write full CSV for download
    csv_out = out_path / f"universal_ratings_{dataset_name}.csv"
    df.write_csv(csv_out)
    console.print(f"[green]✓[/green] Saved complete CSV: [bold]{csv_out}[/bold] ({len(df):,} titles)")

    # Load anime metadata (English and Japanese names) if available
    anime_meta_file = Path("data/mal-57m/anime_meta.parquet")
    anime_meta_map = {}
    if anime_meta_file.exists():
        meta_df = pl.read_parquet(anime_meta_file)
        for row in meta_df.iter_rows(named=True):
            anime_meta_map[row["anime_id"]] = (
                row["name_en"] if row["name_en"] and row["name_en"] != "Unknown" else "",
                row["name_jp"] if row["name_jp"] and row["name_jp"] != "Unknown" else "",
            )

    # Construct compact column-oriented JSON
    cols = [
        "id",
        "title",
        "year",
        "genres",
        "votes",
        "raw",
        "bayes",
        "mod_a",
        "mod_b",
        "mod_c",
        "bias_b_i",
        "raw_rank",
        "cal_rank",
        "rank_delta",
        "cal_delta",
        "diag",
        "title_en",
        "title_jp",
        "media_type",
        "tconst",
        "season_num",
        "ep_count",
        "ml_raw",
        "ml_votes",
        "mal_raw",
        "mal_votes",
        "al_raw",
        "al_votes",
        "imdb_raw",
        "imdb_votes",
        "imdb_ep_avg",
        "imdb_ep_votes",
        "imdb_ep_peak",
    ]

    has_unified_cols = "media_type" in df.columns

    rows = []
    for r in df.iter_rows(named=True):
        year = r.get("year") if has_unified_cols and r.get("year") else extract_year(r["title"])
        clean_title = r["title"] if has_unified_cols else clean_movie_title(r["title"])
        mid = r["movieId"]

        if has_unified_cols:
            title_en = r.get("title_en") or ""
            title_jp = r.get("title_jp") or ""
            media_type = r.get("media_type") or "movie"
            tconst = r.get("tconst") or ""
            season_num = r.get("season_num")
            ep_count = r.get("ep_count")
            ml_raw = r.get("ml_raw")
            ml_votes = r.get("ml_votes")
            mal_raw = r.get("mal_raw")
            mal_votes = r.get("mal_votes")
            al_raw = r.get("al_raw")
            al_votes = r.get("al_votes")
            imdb_raw = r.get("imdb_raw")
            imdb_votes = r.get("imdb_votes")
            imdb_ep_avg = r.get("imdb_ep_avg")
            imdb_ep_votes = r.get("imdb_ep_votes")
            imdb_ep_peak = r.get("imdb_ep_peak")
        else:
            aid = mid if dataset_name.startswith("mal") else None
            title_en, title_jp = anime_meta_map.get(aid, ("", "")) if aid is not None else ("", "")
            media_type = "anime" if dataset_name.startswith("mal") else "movie"
            tconst = ""
            season_num = None
            ep_count = None
            ml_raw = round(r["raw_avg"], 2) if dataset_name.startswith("ml") else None
            ml_votes = r["vote_count"] if dataset_name.startswith("ml") else None
            mal_raw = round(r["raw_avg"], 2) if dataset_name.startswith("mal") else None
            mal_votes = r["vote_count"] if dataset_name.startswith("mal") else None
            al_raw = None
            al_votes = None
            imdb_raw = None
            imdb_votes = None
            imdb_ep_avg = None
            imdb_ep_votes = None
            imdb_ep_peak = None

        rows.append([
            r["movieId"],
            clean_title,
            year or 0,
            r["genres"],
            r["vote_count"],
            round(r["raw_avg"], 2),
            round(r["bayesian_avg"], 2),
            round(r["score_model_a"], 2),
            round(r["score_model_b"], 2),
            round(r["score_model_c"], 2),
            round(r["bias_b_i"], 3),
            r["raw_rank"],
            r["calibrated_rank"],
            r["rank_delta"],
            round(r["calibration_delta"], 2),
            r["rank_diagnostic"],
            title_en,
            title_jp,
            media_type,
            tconst,
            season_num,
            ep_count,
            ml_raw,
            ml_votes,
            mal_raw,
            mal_votes,
            round(al_raw, 2) if al_raw is not None else None,
            al_votes,
            imdb_raw,
            imdb_votes,
            imdb_ep_avg,
            imdb_ep_votes,
            imdb_ep_peak,
        ])

    payload = {
        "dataset": dataset_name,
        "total_titles": len(rows),
        "cols": cols,
        "data": rows,
    }

    json_out = out_path / f"movies_{dataset_name}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    console.print(f"[green]✓[/green] Saved compact JSON: [bold]{json_out}[/bold] ({json_out.stat().st_size / (1024*1024):.2f} MB)")

    # Export genre summary JSON
    if genre_df is not None:
        genre_records = genre_df.to_dicts()
        genre_out = out_path / f"genre_summary_{dataset_name}.json"
        with open(genre_out, "w", encoding="utf-8") as f:
            json.dump(genre_records, f, indent=2)
        console.print(f"[green]✓[/green] Saved genre summary: [bold]{genre_out}[/bold]")

    # Export metadata
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
        with open(out_path / f"metadata_{dataset_name}.json", "w") as f:
            json.dump(meta, f, indent=2)


def export_all():
    console.print("[bold yellow]Exporting web data for GitHub Pages...[/bold yellow]")
    for ds in ["ml-32m", "ml-1m", "ml-latest-small", "combined", "mal", "mal-57m", "imdb", "anilist"]:
        if (Path("artifacts") / ds / "calibrated_scores.parquet").exists():
            export_dataset_for_web(ds)
    console.print("[bold green]✓ All web assets successfully exported![/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export web data for GitHub Pages")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "ml-1m", "ml-32m", "ml-latest-small", "combined", "mal", "mal-57m", "imdb", "anilist", "letterboxd"],
        help="Dataset to export (default: all)",
    )
    args = parser.parse_args()
    if args.dataset == "all":
        export_all()
    else:
        export_dataset_for_web(args.dataset)
