"""UniversalRatingEngine - Data loading and ingestion module."""

from __future__ import annotations

import argparse
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

DATASET_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "mal": "https://huggingface.co/datasets/krishnaveni76/Anime_UserRatings/resolve/main/Anime_UserRatings.csv",
    "mal-57m": "https://huggingface.co/datasets/profoz/anime_rating_complete/resolve/main/rating_complete.csv",
    "mal-large": "https://huggingface.co/datasets/profoz/anime_rating_complete/resolve/main/rating_complete.csv",
    "mal-7m": "https://huggingface.co/datasets/jason1966/CooperUnion_anime-recommendations-database/resolve/main/rating.csv",
    "letterboxd": "https://huggingface.co/datasets/titouann/letterboxd-10m-movies-ratings-2025/resolve/main/train.parquet",
}


def _download_file(url: str, dest_path: Path) -> None:
    """Helper to download a remote file with a rich progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        req = urllib.request.Request(url, headers={"User-Agent": "UniversalRatingEngine/1.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            task_id = progress.add_task(f"Downloading {dest_path.name}", total=total_size)
            with open(dest_path, "wb") as f_out:
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    progress.update(task_id, advance=len(chunk))


def download_and_extract(dataset: str, data_dir: str | Path = "data") -> Path:
    """
    Downloads and extracts the specified dataset (MovieLens, MyAnimeList, Letterboxd).
    Converts diverse source structures to canonical ratings.csv and movies.csv.
    
    Returns the Path to the directory containing ratings.csv and movies.csv.
    """
    dataset = dataset.lower().strip()
    if dataset not in DATASET_URLS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. Supported options: {list(DATASET_URLS.keys())}"
        )

    data_dir = Path(data_dir)
    target_dir = data_dir / dataset
    ratings_csv = target_dir / "ratings.csv"
    movies_csv = target_dir / "movies.csv"

    if ratings_csv.exists() and movies_csv.exists():
        console.print(
            f"[green]✓[/green] Dataset [bold cyan]{dataset}[/bold cyan] already exists at [dim]{target_dir}[/dim]"
        )
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    url = DATASET_URLS[dataset]

    if dataset == "mal":
        raw_path = target_dir / "raw_mal.csv"
        if not raw_path.exists():
            console.print(f"[bold yellow]→[/bold yellow] Downloading MyAnimeList dataset from {url}...")
            _download_file(url, raw_path)

        console.print("[dim]Converting MyAnimeList dataset to canonical format...[/dim]")
        df = pl.read_csv(raw_path)
        movies_df = df.select([
            pl.col("anime_id").cast(pl.Int64).alias("movieId"),
            pl.col("name").alias("title"),
            pl.col("genres").fill_null("Anime").str.replace_all(", ", "|").alias("genres")
        ]).unique(subset=["movieId"], keep="first")
        movies_df.write_csv(movies_csv)

        ratings_df = df.select([
            pl.col("user_id").cast(pl.Int64).alias("userId"),
            pl.col("anime_id").cast(pl.Int64).alias("movieId"),
            pl.col("rating").cast(pl.Float64),
            pl.lit(0).cast(pl.Int64).alias("timestamp"),
        ]).unique(subset=["userId", "movieId"], keep="first")
        ratings_df.write_csv(ratings_csv)

        if raw_path.exists():
            raw_path.unlink()

        console.print(
            f"[green]✓[/green] Successfully prepared [bold cyan]MyAnimeList[/bold cyan] at [dim]{target_dir}[/dim]"
        )
        return target_dir

    elif dataset in ("mal-57m", "mal-large"):
        anime_url = "https://raw.githubusercontent.com/Hernan4444/MyAnimeList-Database/master/data/anime.csv"
        raw_anime_path = target_dir / "anime.csv"
        raw_ratings_path = target_dir / "rating_complete.csv"

        if not movies_csv.exists():
            if not raw_anime_path.exists():
                console.print(f"[bold yellow]→[/bold yellow] Downloading MyAnimeList (57M) anime catalog metadata...")
                _download_file(anime_url, raw_anime_path)
            console.print("[dim]Converting MyAnimeList anime metadata to canonical movies.csv...[/dim]")
            raw_anime_df = pl.read_csv(raw_anime_path, ignore_errors=True)
            movies_df = raw_anime_df.select([
                pl.col("MAL_ID").cast(pl.Int64).alias("movieId"),
                pl.col("Name").alias("title"),
                pl.col("Genres").fill_null("Anime").str.replace_all(", ", "|").alias("genres"),
            ]).unique(subset=["movieId"], keep="first")
            movies_df.write_csv(movies_csv)
            if raw_anime_path.exists():
                raw_anime_path.unlink()

        if not ratings_csv.exists():
            if not raw_ratings_path.exists():
                console.print(f"[bold yellow]→[/bold yellow] Downloading MyAnimeList 57.6M ratings (~780 MB) from {url}...")
                _download_file(url, raw_ratings_path)

            console.print("[dim]Converting MyAnimeList 57M ratings to canonical format with Polars streaming...[/dim]")
            pl.scan_csv(raw_ratings_path).select([
                pl.col("user_id").cast(pl.Int64).alias("userId"),
                pl.col("anime_id").cast(pl.Int64).alias("movieId"),
                pl.col("rating").cast(pl.Float64).alias("rating"),
                pl.lit(0).cast(pl.Int64).alias("timestamp"),
            ]).sink_csv(ratings_csv)

            if raw_ratings_path.exists():
                raw_ratings_path.unlink()

        console.print(
            f"[green]✓[/green] Successfully prepared [bold cyan]MyAnimeList (57M)[/bold cyan] at [dim]{target_dir}[/dim]"
        )
        return target_dir

    elif dataset == "mal-7m":
        anime_url = "https://huggingface.co/datasets/jason1966/CooperUnion_anime-recommendations-database/resolve/main/anime.csv"
        raw_anime_path = target_dir / "raw_anime.csv"
        raw_ratings_path = target_dir / "raw_rating.csv"

        if not movies_csv.exists():
            if not raw_anime_path.exists():
                console.print(f"[bold yellow]→[/bold yellow] Downloading CooperUnion anime catalog metadata...")
                _download_file(anime_url, raw_anime_path)
            console.print("[dim]Converting CooperUnion anime catalog to canonical movies.csv...[/dim]")
            raw_anime_df = pl.read_csv(raw_anime_path, ignore_errors=True)
            movies_df = raw_anime_df.select([
                pl.col("anime_id").cast(pl.Int64).alias("movieId"),
                pl.col("name").alias("title"),
                pl.col("genre").fill_null("Anime").str.replace_all(", ", "|").alias("genres"),
            ]).unique(subset=["movieId"], keep="first")
            movies_df.write_csv(movies_csv)
            if raw_anime_path.exists():
                raw_anime_path.unlink()

        if not ratings_csv.exists():
            if not raw_ratings_path.exists():
                console.print(f"[bold yellow]→[/bold yellow] Downloading CooperUnion 7.8M ratings from {url}...")
                _download_file(url, raw_ratings_path)

            console.print("[dim]Converting CooperUnion ratings to canonical format...[/dim]")
            pl.scan_csv(raw_ratings_path).filter(
                pl.col("rating") > 0
            ).select([
                pl.col("user_id").cast(pl.Int64).alias("userId"),
                pl.col("anime_id").cast(pl.Int64).alias("movieId"),
                pl.col("rating").cast(pl.Float64).alias("rating"),
                pl.lit(0).cast(pl.Int64).alias("timestamp"),
            ]).sink_csv(ratings_csv)

            if raw_ratings_path.exists():
                raw_ratings_path.unlink()

        console.print(
            f"[green]✓[/green] Successfully prepared [bold cyan]MyAnimeList (7.8M CooperUnion)[/bold cyan] at [dim]{target_dir}[/dim]"
        )
        return target_dir

    elif dataset == "letterboxd":
        raw_path = target_dir / "train.parquet"
        if not raw_path.exists():
            console.print(f"[bold yellow]→[/bold yellow] Downloading Letterboxd 10M dataset from {url}...")
            _download_file(url, raw_path)

        console.print("[dim]Converting Letterboxd Parquet to canonical format...[/dim]")
        lb_df = pl.read_parquet(raw_path)
        movies_df = lb_df.select([
            pl.col("movie_id").cast(pl.Int64).alias("movieId"),
            pl.col("title").str.replace_all("-", " ").str.to_titlecase().alias("title"),
            pl.lit("Film").alias("genres")
        ]).unique(subset=["movieId"], keep="first")
        movies_df.write_csv(movies_csv)

        ratings_df = lb_df.select([
            pl.col("user").cast(pl.Int64).alias("userId"),
            pl.col("movie_id").cast(pl.Int64).alias("movieId"),
            (pl.col("rating").cast(pl.Float64) * 2.0).alias("rating"),
            pl.lit(0).cast(pl.Int64).alias("timestamp"),
        ]).unique(subset=["userId", "movieId"], keep="first")
        ratings_df.write_csv(ratings_csv)

        if raw_path.exists():
            raw_path.unlink()

        console.print(
            f"[green]✓[/green] Successfully prepared [bold cyan]Letterboxd[/bold cyan] at [dim]{target_dir}[/dim]"
        )
        return target_dir

    # Standard zip-based datasets (MovieLens ml-1m, ml-25m, ml-32m, ml-latest-small)
    zip_path = data_dir / f"{dataset}.zip"
    if not zip_path.exists():
        console.print(f"[bold yellow]→[/bold yellow] Downloading {dataset} from {url}...")
        _download_file(url, zip_path)
    else:
        console.print(f"[green]✓[/green] Found existing archive at [dim]{zip_path}[/dim]")

    console.print(f"[bold yellow]→[/bold yellow] Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if dataset == "ml-1m":
            console.print("[dim]Converting ML-1M .dat files to standard CSV...[/dim]")
            movies_raw = zf.read("ml-1m/movies.dat").decode("latin-1")
            movies_lines = movies_raw.strip().splitlines()
            movies_df = pl.DataFrame(
                [line.split("::", 2) for line in movies_lines if line],
                schema=["movieId", "title", "genres"],
                orient="row",
            ).with_columns(pl.col("movieId").cast(pl.Int64))
            movies_df.write_csv(movies_csv)

            ratings_raw = zf.read("ml-1m/ratings.dat").decode("latin-1")
            ratings_lines = ratings_raw.strip().splitlines()
            ratings_df = pl.DataFrame(
                [line.split("::") for line in ratings_lines if line],
                schema=["userId", "movieId", "rating", "timestamp"],
                orient="row",
            ).with_columns([
                pl.col("userId").cast(pl.Int64),
                pl.col("movieId").cast(pl.Int64),
                pl.col("rating").cast(pl.Float64),
                pl.col("timestamp").cast(pl.Int64),
            ])
            ratings_df.write_csv(ratings_csv)
        else:
            for member in zf.namelist():
                base_name = os.path.basename(member)
                if base_name in ("ratings.csv", "movies.csv", "README.txt"):
                    target_file = target_dir / base_name
                    with zf.open(member) as source, open(target_file, "wb") as target:
                        shutil.copyfileobj(source, target)

    if zip_path.exists():
        zip_path.unlink()

    console.print(
        f"[green]✓[/green] Successfully extracted [bold cyan]{dataset}[/bold cyan] to [dim]{target_dir}[/dim]"
    )
    return target_dir


def load_raw_data(
    data_dir: str | Path,
    scale_to_10: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Loads ratings.csv and movies.csv into Polars DataFrames.
    Scales 5-point ratings (0.5 - 5.0) to 10-point scale (1.0 - 10.0) if scale_to_10 is True.
    """
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {data_dir}. Expected ratings.csv and movies.csv."
        )

    # Fast read using Polars
    ratings_df = pl.read_csv(
        ratings_path,
        schema_overrides={
            "userId": pl.Int64,
            "movieId": pl.Int64,
            "rating": pl.Float64,
            "timestamp": pl.Int64,
        },
    )

    movies_df = pl.read_csv(
        movies_path,
        schema_overrides={
            "movieId": pl.Int64,
            "title": pl.String,
            "genres": pl.String,
        },
    )

    if scale_to_10:
        # Check current max rating: if max <= 5.0, multiply by 2
        max_rating = ratings_df["rating"].max()
        if max_rating is not None and max_rating <= 5.01:
            ratings_df = ratings_df.with_columns(
                (pl.col("rating") * 2.0).alias("rating")
            )

    # Deduplicate: if a user rated the same title multiple times, keep the latest rating
    initial_count = len(ratings_df)
    if "timestamp" in ratings_df.columns and (ratings_df["timestamp"].max() != ratings_df["timestamp"].min()):
        ratings_df = ratings_df.sort("timestamp").unique(subset=["userId", "movieId"], keep="last")
    else:
        ratings_df = ratings_df.unique(subset=["userId", "movieId"], keep="first")

    dedup_count = initial_count - len(ratings_df)
    if dedup_count > 0:
        console.print(
            f"[dim]Deduplicated {dedup_count:,} duplicate ratings (kept most recent rating per user-item pair)[/dim]"
        )

    return ratings_df, movies_df


def filter_cold_start(
    ratings_df: pl.DataFrame,
    movies_df: pl.DataFrame,
    min_user_ratings: int = 15,
    min_item_ratings: int = 30,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Filters out cold-start noise:
    - Exclude users with < min_user_ratings (default 15)
    - Exclude items with < min_item_ratings (default 30)
    
    Iterates until convergence to strictly satisfy both thresholds.
    """
    initial_ratings_count = len(ratings_df)
    initial_users_count = ratings_df["userId"].n_unique()
    initial_items_count = ratings_df["movieId"].n_unique()

    current_ratings = ratings_df
    iteration = 0
    while True:
        iteration += 1
        prev_len = len(current_ratings)
        current_ratings = (
            current_ratings
            .filter(pl.len().over("userId") >= min_user_ratings)
            .filter(pl.len().over("movieId") >= min_item_ratings)
        )
        if len(current_ratings) == prev_len:
            break

    # Keep only movies that remain in filtered ratings
    filtered_movies = movies_df.join(
        current_ratings.select("movieId").unique(), on="movieId", how="semi"
    )

    console.print(
        f"[dim]Cold-start filtering (min_user={min_user_ratings}, min_item={min_item_ratings}): "
        f"Ratings: {initial_ratings_count:,} → {len(current_ratings):,} | "
        f"Users: {initial_users_count:,} → {current_ratings['userId'].n_unique():,} | "
        f"Items: {initial_items_count:,} → {filtered_movies['movieId'].n_unique():,} "
        f"(converged in {iteration} passes)[/dim]"
    )

    return current_ratings, filtered_movies


def get_combined_data(
    sources: list[str] = ("ml-1m", "ml-latest-small", "mal-57m"),
    data_dir: str | Path = "data",
    min_user_ratings: int = 15,
    min_item_ratings: int = 30,
    scale_to_10: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Combines multiple rating cohorts/databases into a unified benchmarking dataset.

    Data Integrity & Deduplication Architecture:
    1. Namespaces User IDs across disparate databases (offsetting each source by 1,000,000,000 * i)
       to ensure independent user leniency (b_u) and zero synthetic identity collisions.
    2. Namespaces non-MovieLens item IDs (offset by 1,000,000 * (i+1)) while preserving MovieLens
       cross-era movie alignment.
    3. Sorts ratings by timestamp and deduplicates on (userId, movieId), keeping the latest rating.
    4. Executes iterative dual-threshold cold-start pruning.
    """
    console.print(
        f"[bold cyan]→ Ingesting Combined Multi-Cohort Benchmark[/bold cyan] from sources: {list(sources)}"
    )
    all_ratings = []
    all_movies = []

    for idx, src in enumerate(sources):
        src_dir = download_and_extract(src, data_dir=data_dir)
        r_df, m_df = load_raw_data(src_dir, scale_to_10=scale_to_10)

        # Namespace user IDs so users from cohort 1 and cohort 2 don't collide
        offset = idx * 1_000_000_000
        if offset > 0:
            r_df = r_df.with_columns((pl.col("userId") + offset).alias("userId"))

        # Namespace item IDs for non-MovieLens sources to avoid ID collision
        if src.startswith("mal") or src == "letterboxd":
            item_offset = (idx + 1) * 1_000_000
            r_df = r_df.with_columns((pl.col("movieId") + item_offset).alias("movieId"))
            m_df = m_df.with_columns((pl.col("movieId") + item_offset).alias("movieId"))

        all_ratings.append(r_df)
        all_movies.append(m_df)
        console.print(
            f"  [dim]• Cohort '{src}': {len(r_df):,} ratings across {m_df['movieId'].n_unique():,} titles[/dim]"
        )

    # Union ratings & movies
    combined_ratings = pl.concat(all_ratings)
    # Deduplicate ratings across interactions
    if "timestamp" in combined_ratings.columns and (combined_ratings["timestamp"].max() != combined_ratings["timestamp"].min()):
        combined_ratings = combined_ratings.sort("timestamp").unique(
            subset=["userId", "movieId"], keep="last"
        )
    else:
        combined_ratings = combined_ratings.unique(subset=["userId", "movieId"], keep="first")
    combined_movies = pl.concat(all_movies).unique(subset=["movieId"], keep="first")

    console.print(
        f"[dim]Combined raw total: {len(combined_ratings):,} ratings across "
        f"{combined_movies['movieId'].n_unique():,} unique titles[/dim]"
    )

    return filter_cold_start(
        combined_ratings,
        combined_movies,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
    )


def get_data(
    dataset: str = "ml-1m",
    data_dir: str | Path = "data",
    min_user_ratings: int = 15,
    min_item_ratings: int = 30,
    scale_to_10: bool = True,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Convenience pipeline to download (if needed), load, and filter dataset."""
    dataset = dataset.lower().strip()
    if dataset == "combined":
        return get_combined_data(
            sources=["ml-1m", "ml-latest-small", "mal-57m"],
            data_dir=data_dir,
            min_user_ratings=min_user_ratings,
            min_item_ratings=min_item_ratings,
            scale_to_10=scale_to_10,
        )

    target_dir = download_and_extract(dataset, data_dir=data_dir)
    ratings_df, movies_df = load_raw_data(target_dir, scale_to_10=scale_to_10)
    return filter_cold_start(
        ratings_df,
        movies_df,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniversalRatingEngine Data Loader")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-1m",
        choices=["ml-1m", "ml-25m", "ml-32m", "ml-latest-small", "mal", "mal-57m", "mal-large", "mal-7m", "letterboxd", "combined"],
        help="Dataset to download and extract (ml-32m, ml-1m, mal, mal-57m, letterboxd, etc.)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store datasets",
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=15,
        help="Minimum ratings required per user",
    )
    parser.add_argument(
        "--min-item-ratings",
        type=int,
        default=30,
        help="Minimum ratings required per movie",
    )
    args = parser.parse_args()

    console.print(f"[bold green]Starting Data Loader for {args.dataset}...[/bold green]")
    r_df, m_df = get_data(
        dataset=args.dataset,
        data_dir=args.data_dir,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings,
    )
    console.print(
        f"[bold green]Done![/bold green] Loaded {len(r_df):,} ratings and {len(m_df):,} movies."
    )
