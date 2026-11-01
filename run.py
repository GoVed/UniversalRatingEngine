#!/usr/bin/env python3
"""UniversalRatingEngine - Production-grade CLI for cross-genre quality calibration.

Commands:
    download    Download and extract benchmark datasets (ml-1m, ml-32m)
    fit         Train calibration pipeline across Model A, B, and C
    inspect     Inspect calibrated scores and rank shifts for any title
    report      Generate comprehensive Markdown report and terminal summary
    top         List top calibrated titles with optional genre filtering
    compare     Compare multiple titles side-by-side
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data_loader import download_and_extract
from src.engine import UniversalRatingEngine, format_display_title
from src.report import generate_markdown_report, print_terminal_summary

app = typer.Typer(
    name="UniversalRatingEngine",
    help="Cross-genre quality scoring and self-selection bias calibration pipeline.",
    add_completion=False,
)
console = Console()


def print_combined_disclaimer() -> None:
    """Prints methodology and integrity disclaimer for multi-cohort datasets."""
    disclaimer_text = (
        "[bold yellow]⚠ COMBINED MULTI-COHORT BENCHMARK DISCLAIMER[/bold yellow]\n\n"
        "• [bold]Cohort Namespacing[/bold]: User IDs are offset by database (+1,000,000,000) to ensure individual "
        "leniency (b_u) and variance (σ_u) are modeled without identity collisions.\n"
        "• [bold]Interaction Deduplication[/bold]: Repeated interactions by the same user on the same title "
        "are deduplicated by latest timestamp.\n"
        "• [bold]Longitudinal & Cultural Synthesis[/bold]: Model B's latent factor decomposition isolates intrinsic quality (b_i) "
        "across cohorts. However, merging cohorts spanning different decades (e.g. 2000 vs 2018) blends disparate cultural contexts "
        "and evolving audience demographics.\n"
        "• [bold]Dual-Threshold Pruning[/bold]: Iterative cold-start filtering (≥15 ratings/user, ≥30 ratings/movie) is strictly enforced."
    )
    console.print(Panel(disclaimer_text, title="Data Integrity Notice", border_style="yellow"))


@app.command()
def download(
    dataset: str = typer.Option("ml-1m", "--dataset", "-d", help="Dataset to download (ml-32m, ml-1m, mal, mal-57m, letterboxd, ml-latest-small)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Target directory for downloaded data"),
) -> None:
    """Download and extract MovieLens, MyAnimeList, or Letterboxd benchmark datasets."""
    console.print(f"[bold yellow]Downloading and preparing {dataset}...[/bold yellow]")
    download_and_extract(dataset=dataset, data_dir=data_dir)
    console.print(f"[bold green]✓ Dataset {dataset} ready for ingestion![/bold green]")


@app.command()
def fit(
    dataset: str = typer.Option("ml-1m", "--dataset", "-d", help="Dataset name (ml-32m, ml-1m, mal, mal-57m, letterboxd, ml-latest-small, combined)"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs for Model B (Biased MF)"),
    k: int = typer.Option(32, "--k", "-k", help="Latent factor dimension (20-50)"),
    lambda_reg: float = typer.Option(0.02, "--lambda-reg", "-l", help="L2 regularization parameter"),
    lr: float = typer.Option(0.01, "--lr", help="Learning rate for AdamW optimizer"),
    batch_size: int = typer.Option(32768, "--batch-size", "-b", help="Batch size for matrix factorization"),
    min_user_ratings: int = typer.Option(15, "--min-user", help="Cold-start threshold: min ratings per user"),
    min_item_ratings: int = typer.Option(30, "--min-item", help="Cold-start threshold: min ratings per item"),
    device: str = typer.Option("auto", "--device", help="Device for matrix factorization ('auto', 'cuda', 'cpu')"),
    data_dir: str = typer.Option("data", "--data-dir", help="Base directory containing datasets"),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir", help="Directory to save calibrated models"),
) -> None:
    """Fit calibration models (Model A: Z-Score, Model B: Biased MF, Model C: Bridge Users)."""
    if dataset.lower().strip() == "combined":
        print_combined_disclaimer()

    engine = UniversalRatingEngine(dataset=dataset, artifacts_dir=artifacts_dir, data_dir=data_dir)
    engine.fit(
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
        k=k,
        lambda_reg=lambda_reg,
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        verbose=True,
    )


@app.command()
def inspect(
    title: str = typer.Argument(..., help="Title to inspect (exact or fuzzy match)"),
    dataset: str = typer.Option("ml-1m", "--dataset", "-d", help="Dataset to search within (ml-32m, ml-1m, mal, letterboxd, ml-latest-small, combined)"),
    data_dir: str = typer.Option("data", "--data-dir", help="Base directory containing datasets"),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir", help="Directory where models are saved"),
    auto_fallback: bool = typer.Option(True, "--auto-fallback/--no-fallback", help="Automatically fall back to ml-32m if title only exists there"),
) -> None:
    """Inspect calibrated scores and audience shift diagnostics for any title."""
    if dataset.lower().strip() == "combined":
        print_combined_disclaimer()

    engine = UniversalRatingEngine(dataset=dataset, artifacts_dir=artifacts_dir, data_dir=data_dir)
    
    # Check if artifacts exist; if not, train them
    cal_file = Path(artifacts_dir) / dataset / "calibrated_scores.parquet"
    if not cal_file.exists():
        console.print(f"[yellow]Artifacts for {dataset} not found. Running fit first...[/yellow]")
        engine.fit()
    else:
        engine.load()

    res = engine.inspect(title)
    if res is None and auto_fallback and dataset not in ("ml-32m", "combined"):
        # Check if title exists in ml-32m
        alt_dataset = "ml-32m"
        alt_cal = Path(artifacts_dir) / alt_dataset / "calibrated_scores.parquet"
        alt_data = Path(data_dir) / alt_dataset / "movies.csv"

        if alt_cal.exists() or alt_data.exists():
            console.print(f"\n[cyan]Attempting inspection on fallback dataset: [bold]{alt_dataset}[/bold]...[/cyan]")
            alt_engine = UniversalRatingEngine(dataset=alt_dataset, artifacts_dir=artifacts_dir, data_dir=data_dir)
            if not alt_cal.exists():
                console.print(f"[yellow]Fitting {alt_dataset} engine...[/yellow]")
                alt_engine.fit(epochs=5, batch_size=65536)
            else:
                alt_engine.load()
            alt_engine.inspect(title)


@app.command()
def report(
    dataset: str = typer.Option("ml-1m", "--dataset", "-d", help="Dataset name (ml-32m, ml-1m, mal, mal-57m, letterboxd, ml-latest-small, combined)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save Markdown report"),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir", help="Directory containing model artifacts"),
    data_dir: str = typer.Option("data", "--data-dir", help="Base directory containing datasets"),
) -> None:
    """Generate Markdown calibration report and render terminal summary."""
    if dataset.lower().strip() == "combined":
        print_combined_disclaimer()

    engine = UniversalRatingEngine(dataset=dataset, artifacts_dir=artifacts_dir, data_dir=data_dir)
    engine.load()

    assert engine.calibrated_df is not None
    assert engine.genre_summary_df is not None

    if output is None:
        output = f"reports/{dataset}_calibration_report.md"

    # Terminal summary
    print_terminal_summary(engine.calibrated_df, engine.genre_summary_df)

    # Save Markdown report
    generate_markdown_report(
        calibrated_df=engine.calibrated_df,
        genre_summary_df=engine.genre_summary_df,
        dataset_name=dataset,
        output_path=output,
    )


@app.command()
def top(
    dataset: str = typer.Option("ml-1m", "--dataset", "-d", help="Dataset name (ml-32m, ml-1m, mal, mal-57m, letterboxd, ml-latest-small, combined)"),
    n: int = typer.Option(20, "-n", help="Number of titles to display"),
    genre: Optional[str] = typer.Option(None, "--genre", "-g", help="Filter by specific genre (e.g. Animation, Horror)"),
    by: str = typer.Option("model_b", "--by", help="Sort metric: model_b, model_a, model_c, raw, delta"),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir", help="Directory containing model artifacts"),
    data_dir: str = typer.Option("data", "--data-dir", help="Base directory containing datasets"),
) -> None:
    """List top calibrated movies with optional genre filter and sorting metric."""
    if dataset.lower().strip() == "combined":
        print_combined_disclaimer()
    engine = UniversalRatingEngine(dataset=dataset, artifacts_dir=artifacts_dir, data_dir=data_dir)
    engine.load()

    assert engine.calibrated_df is not None
    df = engine.calibrated_df

    if genre:
        df = df.filter(pl.col("genres").str.to_lowercase().str.contains(genre.lower()))
        title_prefix = f"Top {n} {genre} Titles"
    else:
        title_prefix = f"Top {n} Titles"

    sort_col_map = {
        "model_b": "score_model_b",
        "model_a": "score_model_a",
        "model_c": "score_model_c",
        "raw": "raw_avg",
        "delta": "calibration_delta",
    }
    col = sort_col_map.get(by, "score_model_b")
    df = df.sort(col, descending=(by != "delta")).head(n)

    t = Table(title=f"[bold cyan]{title_prefix} (Sorted by {col})[/bold cyan]", header_style="bold magenta")
    t.add_column("Rank", justify="right", style="dim")
    t.add_column("Title", style="bold white")
    t.add_column("Genres", style="dim")
    t.add_column("Votes", justify="right")
    t.add_column("Raw Avg", justify="right")
    t.add_column("Bayes", justify="right")
    t.add_column("Model A", justify="right")
    t.add_column("Model B", justify="right", style="bold green")
    t.add_column("Model C", justify="right")
    t.add_column("Delta", justify="right", style="yellow")
    t.add_column("Rank Shift", style="cyan")

    for idx, row in enumerate(df.iter_rows(named=True), start=1):
        clean_name = format_display_title(row["title"])
        t.add_row(
            str(idx),
            clean_name[:38],
            row["genres"][:20],
            f"{row['vote_count']:,}",
            f"{row['raw_avg']:.2f}",
            f"{row['bayesian_avg']:.2f}",
            f"{row['score_model_a']:.2f}",
            f"{row['score_model_b']:.2f}",
            f"{row['score_model_c']:.2f}",
            f"{row['calibration_delta']:+.2f}",
            row["rank_diagnostic"],
        )
    console.print(t)


@app.command()
def compare(
    titles: List[str] = typer.Argument(..., help="List of titles to compare"),
    dataset: str = typer.Option("ml-1m", "--dataset", "-d", help="Dataset name (ml-1m, ml-32m)"),
    artifacts_dir: str = typer.Option("artifacts", "--artifacts-dir", help="Directory containing model artifacts"),
    data_dir: str = typer.Option("data", "--data-dir", help="Base directory containing datasets"),
) -> None:
    """Compare multiple movie titles side-by-side."""
    engine = UniversalRatingEngine(dataset=dataset, artifacts_dir=artifacts_dir, data_dir=data_dir)
    engine.load()

    assert engine.calibrated_df is not None

    table = Table(title=f"[bold yellow]Head-to-Head Movie Calibration Comparison ({dataset.upper()})[/bold yellow]")
    table.add_column("Title", style="bold white")
    table.add_column("Genres", style="dim")
    table.add_column("Votes", justify="right")
    table.add_column("Raw Avg", justify="right")
    table.add_column("Bayes Avg", justify="right")
    table.add_column("Model A (Z)", justify="right")
    table.add_column("Model B (Latent)", justify="right", style="bold green")
    table.add_column("Model C (Bridge)", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Rank Diagnostic", style="cyan")

    for t_query in titles:
        matches = engine.search_title(t_query)
        if not matches:
            table.add_row(f"[red]{t_query} (Not found)[/red]", "-", "-", "-", "-", "-", "-", "-", "-", "-")
            continue
        mid = matches[0]["movieId"]
        row = engine.calibrated_df.filter(pl.col("movieId") == mid).to_dicts()[0]
        d_val = row["calibration_delta"]
        d_style = "bold red" if d_val < 0 else "bold green"
        table.add_row(
            row["title"][:35],
            row["genres"][:18],
            f"{row['vote_count']:,}",
            f"{row['raw_avg']:.2f}",
            f"{row['bayesian_avg']:.2f}",
            f"{row['score_model_a']:.2f}",
            f"{row['score_model_b']:.2f}",
            f"{row['score_model_c']:.2f}",
            f"[{d_style}]{d_val:+.2f}[/{d_style}]",
            row["rank_diagnostic"],
        )
    console.print(table)


if __name__ == "__main__":
    app()
