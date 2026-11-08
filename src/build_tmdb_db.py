"""TMDb (The Movie Database) TV & Cinema Benchmark Integration Pipeline.

Ingests TMDb user evaluations and season-level ratings:
1. TV Series & Season-level scores (1,120+ series mapped via Wikidata to TMDb TV IDs).
2. Cinema Movie scores (38,000+ films cross-referenced by IMDb tconst).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import polars as pl
from rich.console import Console

console = Console()

DATA_DIR = Path("data/tmdb")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = Path("artifacts/tmdb")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TV_RATINGS_FILE = DATA_DIR / "tmdb_tv_ratings.json"
MOVIE_RATINGS_FILE = DATA_DIR / "tmdb_movie_ratings.parquet"
WIKIDATA_MAP_FILE = DATA_DIR / "wikidata_imdb_tmdb_tv_map.json"


def fetch_wikidata_tv_mappings(parent_tconsts: list[str]) -> dict[str, int]:
    """Fetch IMDb tconst -> TMDb TV series ID mappings via Wikidata SPARQL."""
    if WIKIDATA_MAP_FILE.exists():
        console.print(f"[cyan]→ Loading cached Wikidata TV mappings from {WIKIDATA_MAP_FILE}...[/cyan]")
        with open(WIKIDATA_MAP_FILE) as f:
            return json.load(f)

    console.print(f"[cyan]→ Querying Wikidata SPARQL for {len(parent_tconsts):,} parent TV series mappings...[/cyan]")
    mappings: dict[str, int] = {}
    t0 = time.time()

    for i in range(0, len(parent_tconsts), 250):
        batch = parent_tconsts[i : i + 250]
        values_str = " ".join(f'"{tc}"' for tc in batch)
        query = f"""
        SELECT ?imdb ?tmdb WHERE {{
          VALUES ?imdb {{ {values_str} }}
          ?item wdt:P345 ?imdb ;
                wdt:P4983 ?tmdb .
        }}
        """
        url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode(
            {"query": query, "format": "json"}
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "UniversalRatingEngine/1.0 (contact@ure.org)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode())
                for b in res["results"]["bindings"]:
                    mappings[b["imdb"]["value"]] = int(b["tmdb"]["value"])
        except Exception as e:
            console.print(f"[yellow]Warning: Wikidata batch {i} error: {e}[/yellow]")

    console.print(
        f"  [green]✓[/green] Mapped {len(mappings):,} / {len(parent_tconsts):,} parent TV shows in {time.time()-t0:.2f}s"
    )

    with open(WIKIDATA_MAP_FILE, "w") as f:
        json.dump(mappings, f, indent=2)

    return mappings


def fetch_tmdb_tv_ratings(mappings: dict[str, int]) -> dict:
    """Stream TMDb TV series dataset to extract show and season-level ratings."""
    if TV_RATINGS_FILE.exists():
        console.print(f"[cyan]→ Loading cached TMDb TV ratings from {TV_RATINGS_FILE}...[/cyan]")
        with open(TV_RATINGS_FILE) as f:
            return json.load(f)

    console.print("[cyan]→ Streaming TMDb TV series & season-level evaluations from HuggingFace...[/cyan]")
    stream_url = "https://huggingface.co/datasets/tmkhoa/tmdb-craw-dataset/resolve/main/data/tv_series/tv_series_12_31_2025_1.jsonl"
    target_tmdb_ids = set(mappings.values())
    tmdb_to_imdb = {tmdb_id: tc for tc, tmdb_id in mappings.items()}

    tv_results = {}
    t0 = time.time()
    req = urllib.request.Request(stream_url, headers={"User-Agent": "Mozilla/5.0"})

    count_matched = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        buffer = ""
        while chunk := resp.read(131072):
            buffer += chunk.decode("utf-8", errors="ignore")
            lines = buffer.split("\n")
            buffer = lines.pop()
            for line in lines:
                if not line.strip() or '"tv_series_id":' not in line:
                    continue
                try:
                    sid_str = line.split('"tv_series_id":')[1].split(",")[0].strip()
                    sid = int(sid_str)
                    if sid in target_tmdb_ids:
                        obj = json.loads(line)
                        detail = obj.get("tv_series_detail", {})
                        tc = tmdb_to_imdb[sid]

                        seasons_dict = {}
                        for s in detail.get("seasons", []):
                            s_num = s.get("season_number")
                            if s_num is not None:
                                seasons_dict[str(s_num)] = {
                                    "vote_average": round(float(s.get("vote_average") or 0.0), 2),
                                    "episode_count": int(s.get("episode_count") or 0),
                                    "name": s.get("name") or "",
                                }

                        tv_results[tc] = {
                            "tmdb_id": sid,
                            "name": detail.get("name") or "",
                            "vote_average": round(float(detail.get("vote_average") or 0.0), 2),
                            "vote_count": int(detail.get("vote_count") or 0),
                            "first_air_date": detail.get("first_air_date") or "",
                            "seasons": seasons_dict,
                        }
                        count_matched += 1
                        if count_matched >= len(target_tmdb_ids):
                            break
                except Exception:
                    pass
            if count_matched >= len(target_tmdb_ids):
                break

    console.print(
        f"  [green]✓[/green] Extracted {len(tv_results):,} TMDb TV show records in {time.time()-t0:.2f}s"
    )

    with open(TV_RATINGS_FILE, "w") as f:
        json.dump(tv_results, f, indent=2)

    return tv_results


def fetch_tmdb_movie_ratings() -> pl.DataFrame:
    """Ingest TMDb cinema ratings from the Hugging Face TMDB-IMDb dataset."""
    if MOVIE_RATINGS_FILE.exists():
        console.print(f"[cyan]→ Loading cached TMDb movie ratings from {MOVIE_RATINGS_FILE}...[/cyan]")
        return pl.read_parquet(MOVIE_RATINGS_FILE)

    console.print("[cyan]→ Streaming TMDb Cinema Movie ratings from HuggingFace dataset...[/cyan]")
    url = "https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset/resolve/main/TMDB%20%20IMDB%20Movies%20Dataset.csv"
    t0 = time.time()

    chunks = []
    chunk_iter = pd.read_csv(
        url,
        chunksize=100000,
        usecols=["tconst", "vote_average", "vote_count"],
    )
    for chunk in chunk_iter:
        valid = chunk[chunk["tconst"].notna() & (chunk["vote_count"] > 0)]
        chunks.append(valid)

    full_df = pd.concat(chunks, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["tconst"])

    p_df = pl.from_pandas(full_df).with_columns([
        pl.col("vote_average").cast(pl.Float64).round(2),
        pl.col("vote_count").cast(pl.Int64),
    ])

    p_df.write_parquet(MOVIE_RATINGS_FILE)
    console.print(
        f"  [green]✓[/green] Ingested {len(p_df):,} TMDb movie ratings in {time.time()-t0:.2f}s"
    )
    return p_df


def build_tmdb_database() -> tuple[dict, pl.DataFrame]:
    """Run full TMDb ingestion pipeline."""
    console.print("[bold cyan]═══ Building TMDb Multi-Platform Benchmark Database ═══[/bold cyan]")

    # 1. Identify TV Parent tconsts
    with open("docs/data/movies_combined.json") as f:
        comb = json.load(f)
    cols = comb["cols"]
    data = comb["data"]
    mt_idx = cols.index("media_type")
    tc_idx = cols.index("tconst")
    tv_items = [r for r in data if r[mt_idx] in ("tv_season", "tv_series")]
    parent_tconsts = sorted(list(set(r[tc_idx] for r in tv_items if r[tc_idx])))

    # 2. Fetch TV mappings and ratings
    wikidata_map = fetch_wikidata_tv_mappings(parent_tconsts)
    tv_ratings = fetch_tmdb_tv_ratings(wikidata_map)

    # 3. Fetch Movie ratings
    movie_df = fetch_tmdb_movie_ratings()

    console.print(
        f"[bold green]✓ TMDb Benchmark Ready: {len(tv_ratings):,} TV series and {len(movie_df):,} movies.[/bold green]"
    )
    return tv_ratings, movie_df


if __name__ == "__main__":
    build_tmdb_database()
