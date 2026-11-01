"""Analysis Report Generator for UniversalRatingEngine.

Generates comprehensive analytical reports:
- Top 25 Titles with the biggest negative calibration delta (enthusiast siloing)
- Top 25 Titles with the biggest positive calibration delta (mismatched audience drag)
- Cross-genre summary comparing inflation/deflation across genres
- Formats in Markdown and terminal Rich tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()


def generate_markdown_report(
    calibrated_df: pl.DataFrame,
    genre_summary_df: pl.DataFrame,
    dataset_name: str = "ml-1m",
    output_path: Optional[str | Path] = None,
) -> str:
    """Generates an exhaustive Markdown calibration report with tables and analysis."""
    # Top 25 biggest negative calibration delta (enthusiast inflation)
    # Sorted ascending by delta (most negative first)
    top_negative = (
        calibrated_df
        .sort("calibration_delta", descending=False)
        .head(25)
    )

    # Top 25 biggest positive calibration delta (harsh audience drag / hidden gems)
    # Sorted descending by delta (most positive first)
    top_positive = (
        calibrated_df
        .sort("calibration_delta", descending=True)
        .head(25)
    )

    total_titles = len(calibrated_df)
    total_votes = calibrated_df["vote_count"].sum()
    avg_raw = calibrated_df["raw_avg"].mean()
    avg_cal_b = calibrated_df["score_model_b"].mean()

    md = []
    md.append(f"# UniversalRatingEngine: Cross-Genre Quality Calibration Report")
    md.append(f"**Dataset**: `{dataset_name}` | **Titles Evaluated**: {total_titles:,} | **Total Ratings Evaluated**: {total_votes:,}\n")
    md.append(f"> [!NOTE]\n> Public review platforms (IMDb, MyAnimeList, Letterboxd) suffer from severe self-selection bias and contextual anchoring. Niche genres attract loyalist enthusiasts who inflate scores, while broad mainstream titles suffer score drag from mismatched audiences. This report presents calibrated scores produced by decomposing user leniency and item latent quality.\n")

    md.append("## Global Dataset Overview")
    md.append(f"- **Population Mean (Raw 10-point Scale)**: `{avg_raw:.2f} / 10.0`")
    md.append(f"- **Calibrated Latent Mean (Model B)**: `{avg_cal_b:.2f} / 10.0`")
    md.append(f"- **Calibration Delta Range**: `[{calibrated_df['calibration_delta'].min():.2f}, +{calibrated_df['calibration_delta'].max():.2f}]`")
    md.append(f"- **Maximum Rank Fall**: `{calibrated_df['rank_delta'].max():,}` ranks")
    md.append(f"- **Maximum Rank Rise**: `{abs(calibrated_df['rank_delta'].min()):,}` ranks\n")

    # Section 1: Top 25 Negative Delta
    md.append("## 1. Top 25 Titles with Biggest Negative Calibration Delta")
    md.append("*Titles artificially inflated by niche enthusiast siloing and sympathetic user bases.*\n")
    md.append("| # | Title | Genres | Votes | Raw Avg | IMDb Bayes | Model A (Z) | Model B (Latent) | Model C (Bridge) | Delta (Δ) | Rank Shift |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for idx, row in enumerate(top_negative.iter_rows(named=True), start=1):
        title = row["title"].replace("|", "/")
        genres = row["genres"].replace("|", ", ")
        votes = f"{row['vote_count']:,}"
        raw = f"{row['raw_avg']:.2f}"
        bayes = f"{row['bayesian_avg']:.2f}"
        mod_a = f"{row['score_model_a']:.2f}"
        mod_b = f"{row['score_model_b']:.2f}"
        mod_c = f"{row['score_model_c']:.2f}"
        delta = f"{row['calibration_delta']:+.2f}"
        rank_shift = row["rank_diagnostic"]
        md.append(f"| {idx} | **{title}** | {genres} | {votes} | {raw} | {bayes} | {mod_a} | **{mod_b}** | {mod_c} | `{delta}` | {rank_shift} |")

    md.append("\n---\n")

    # Section 2: Top 25 Positive Delta
    md.append("## 2. Top 25 Titles with Biggest Positive Calibration Delta")
    md.append("*Titles penalized by mismatched casual viewers, critical general audiences, or harsh rating environments.*\n")
    md.append("| # | Title | Genres | Votes | Raw Avg | IMDb Bayes | Model A (Z) | Model B (Latent) | Model C (Bridge) | Delta (Δ) | Rank Shift |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for idx, row in enumerate(top_positive.iter_rows(named=True), start=1):
        title = row["title"].replace("|", "/")
        genres = row["genres"].replace("|", ", ")
        votes = f"{row['vote_count']:,}"
        raw = f"{row['raw_avg']:.2f}"
        bayes = f"{row['bayesian_avg']:.2f}"
        mod_a = f"{row['score_model_a']:.2f}"
        mod_b = f"{row['score_model_b']:.2f}"
        mod_c = f"{row['score_model_c']:.2f}"
        delta = f"{row['calibration_delta']:+.2f}"
        rank_shift = row["rank_diagnostic"]
        md.append(f"| {idx} | **{title}** | {genres} | {votes} | {raw} | {bayes} | {mod_a} | **{mod_b}** | {mod_c} | `{delta}` | {rank_shift} |")

    md.append("\n---\n")

    # Section 3: Cross-Genre Summary
    md.append("## 3. Cross-Genre Calibration & Inflation Summary")
    md.append("*Comparison of average raw scores versus de-biased quality metrics across genres.*\n")
    md.append("| Genre | Titles | Total Votes | Raw Avg | Bayes Avg | Model A (Z) | Model B (Latent) | Model C (Bridge) | Cal. Delta (Δ) | Siloing Diagnosis |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")

    for row in genre_summary_df.iter_rows(named=True):
        genre = row["genres"]
        titles = f"{row['title_count']:,}"
        votes = f"{row['total_votes']:,}"
        raw = f"{row['avg_raw_rating']:.2f}"
        bayes = f"{row['avg_bayesian_rating']:.2f}"
        mod_a = f"{row['avg_model_a']:.2f}"
        mod_b = f"{row['avg_model_b']:.2f}"
        mod_c = f"{row['avg_model_c']:.2f}"
        delta = f"{row['avg_calibration_delta']:+.3f}"

        # Diagnosis tag
        d_val = row["avg_calibration_delta"]
        if d_val < -0.15:
            diag = "🔴 Severe Enthusiast Inflation"
        elif d_val < -0.05:
            diag = "🟠 Moderate Inflation"
        elif d_val <= 0.05:
            diag = "🟢 Well-Balanced Audience"
        elif d_val <= 0.15:
            diag = "🔵 Mismatched Audience Drag"
        else:
            diag = "🟣 Heavily Underrated / Dragged Down"

        md.append(f"| **{genre}** | {titles} | {votes} | {raw} | {bayes} | {mod_a} | {mod_b} | {mod_c} | `{delta}` | {diag} |")

    md.append("\n```mermaid\nflowchart TD\n")
    md.append('    Raw["Raw Platform Mean (IMDb/MAL)"] --> Bias["Enthusiast Bias & Leniency"]\n')
    md.append('    Raw --> Bayes["IMDb Bayesian Formula (Weighted Mean)"]\n')
    md.append('    Bayes -. "Fails on niche siloing" .-> Silo["Artificial Score Inflation"]\n')
    md.append('    Bias --> ModelA["Model A: User Z-Score Normalization"]\n')
    md.append('    Bias --> ModelB["Model B: Biased Matrix Factorization (Latent Factors)"]\n')
    md.append('    Bias --> ModelC["Model C: Bridge-User & Genre Deflation Multiplier"]\n')
    md.append('    ModelA --> Calibrated["De-biased Cross-Genre Universal Score"]\n')
    md.append('    ModelB --> Calibrated\n')
    md.append('    ModelC --> Calibrated\n')
    md.append("```\n")

    content = "\n".join(md)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        console.print(f"[green]✓[/green] Analysis report saved to [bold cyan]{out}[/bold cyan]")

    return content


def print_terminal_summary(
    calibrated_df: pl.DataFrame,
    genre_summary_df: pl.DataFrame,
) -> None:
    """Renders sleek Rich tables to the terminal."""
    # Top 10 Negative Delta Table
    t_neg = Table(
        title="[bold red]Top 10 Enthusiast-Inflated Titles (Biggest Negative Delta)[/bold red]",
        header_style="bold magenta",
        show_lines=False,
    )
    t_neg.add_column("Rank", justify="right", style="dim")
    t_neg.add_column("Title", style="bold white")
    t_neg.add_column("Genres", style="dim")
    t_neg.add_column("Raw Avg", justify="right")
    t_neg.add_column("Model B", justify="right", style="bold green")
    t_neg.add_column("Delta", justify="right", style="bold red")
    t_neg.add_column("Rank Shift", style="yellow")

    top_neg = calibrated_df.sort("calibration_delta", descending=False).head(10)
    for idx, row in enumerate(top_neg.iter_rows(named=True), start=1):
        t_neg.add_row(
            str(idx),
            row["title"][:38],
            row["genres"][:22],
            f"{row['raw_avg']:.2f}",
            f"{row['score_model_b']:.2f}",
            f"{row['calibration_delta']:+.2f}",
            row["rank_diagnostic"],
        )
    console.print(t_neg)

    # Top 10 Positive Delta Table
    t_pos = Table(
        title="[bold green]Top 10 Audience-Penalized Titles (Biggest Positive Delta)[/bold green]",
        header_style="bold cyan",
        show_lines=False,
    )
    t_pos.add_column("Rank", justify="right", style="dim")
    t_pos.add_column("Title", style="bold white")
    t_pos.add_column("Genres", style="dim")
    t_pos.add_column("Raw Avg", justify="right")
    t_pos.add_column("Model B", justify="right", style="bold green")
    t_pos.add_column("Delta", justify="right", style="bold cyan")
    t_pos.add_column("Rank Shift", style="green")

    top_pos = calibrated_df.sort("calibration_delta", descending=True).head(10)
    for idx, row in enumerate(top_pos.iter_rows(named=True), start=1):
        t_pos.add_row(
            str(idx),
            row["title"][:38],
            row["genres"][:22],
            f"{row['raw_avg']:.2f}",
            f"{row['score_model_b']:.2f}",
            f"{row['calibration_delta']:+.2f}",
            row["rank_diagnostic"],
        )
    console.print(t_pos)

    # Genre Inflation Table
    t_genre = Table(
        title="[bold yellow]Cross-Genre Inflation & Deflation Summary[/bold yellow]",
        header_style="bold yellow",
        show_lines=False,
    )
    t_genre.add_column("Genre", style="bold white")
    t_genre.add_column("Titles", justify="right")
    t_genre.add_column("Raw Avg", justify="right")
    t_genre.add_column("Model B", justify="right", style="bold green")
    t_genre.add_column("Cal. Delta", justify="right")
    t_genre.add_column("Bridge Mult (M_g)", justify="right")

    for row in genre_summary_df.iter_rows(named=True):
        delta = row["avg_calibration_delta"]
        style_delta = "bold red" if delta < -0.05 else ("bold green" if delta > 0.05 else "white")
        t_genre.add_row(
            row["genres"],
            f"{row['title_count']:,}",
            f"{row['avg_raw_rating']:.2f}",
            f"{row['avg_model_b']:.2f}",
            f"[{style_delta}]{delta:+.3f}[/{style_delta}]",
            f"{row['avg_model_c'] / row['avg_raw_rating']:.3f}",
        )
    console.print(t_genre)
