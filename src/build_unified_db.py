"""
Unified Multi-Database Correlation & Calibration Pipeline (MovieLens 32M + MyAnimeList 57M + IMDb 12.9M).

Correlates:
  - Episode <-> Season <-> Parent Title across IMDb (title.episode, title.ratings, title.basics),
  - MovieLens 32M (via links.csv -> tconst),
  - MyAnimeList 57M (via Fribb/anime-lists -> tconst + seasonNumber + episode_offset, plus title/season index).

Produces a unified, de-biased Master Database where every row exposes its raw ratings across
MovieLens, MyAnimeList, IMDb (Title/Series), and IMDb (Season Episode Rollup), alongside the
step-by-step calibrated Universal Rating Engine scores (Model A, Model B, Model C).
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console

console = Console()


def normalize_title_key(s: str | None) -> str:
    """Normalize title string for fallback cross-database matching."""
    if not s:
        return ""
    s = re.sub(r"\(\d{4}(?:–\d{4})?\)$", "", s).strip().lower()
    # Strip season suffixes for parent lookup
    s = re.sub(
        r"\b(?:season\s*\d+|s\d+|\d+(?:st|nd|rd|th)\s*season|part\s*\d+|cour\s*\d+|final\s*season|the\s*final)\b",
        "",
        s,
    )
    s = re.sub(r"\s+(?:ii|iii|iv|v|vi)$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def infer_season_number(title: str | None) -> int | None:
    """Infer season number from anime/TV title string if not explicitly mapped."""
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
    if "°" in title or "fourth" in t:
        return 4
    return None


def clean_display_title(raw_title: str) -> str:
    """Clean MovieLens 'Title, The (Year)' into 'The Title'."""
    t = re.sub(r"\s*\(\d{4}(?:–\d{4})?\)\s*$", "", raw_title).strip()
    m = re.match(r"^(.*?),\s*(The|A|An|Le|La|Les|L'|Der|Die|Das|Il|El|Los|Las)$", t, re.IGNORECASE)
    if m:
        base, prefix = m.group(1).strip(), m.group(2).strip()
        t = f"{prefix}{base}" if prefix.endswith("'") else f"{prefix} {base}"
    return t


def extract_year(raw_title: str) -> int | None:
    m = re.search(r"\((\d{4})(?:–\d{4})?\)\s*$", raw_title)
    if m:
        return int(m.group(1))
    return None


def build_unified_database() -> None:
    t_start = time.time()
    console.print("[bold cyan]═══ Building Unified Tri-Database Benchmark (MovieLens 32M + MAL 57M + IMDb 12.9M) ═══[/bold cyan]")

    # 1. Load IMDb Official Tables
    console.print("[cyan]→ Loading IMDb official tables (ratings, episodes, basics)...[/cyan]")
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
    imdb_basics = pl.read_csv(
        "data/imdb/title.basics.tsv.gz",
        separator="\t",
        null_values=r"\N",
        quote_char=None,
        columns=["tconst", "titleType", "primaryTitle", "originalTitle", "startYear", "genres"],
        schema_overrides={"startYear": pl.Int32},
    )
    console.print(
        f"  [green]✓[/green] IMDb loaded: {len(imdb_basics):,} titles, "
        f"{len(imdb_episodes):,} episodes, {len(imdb_ratings):,} rated entries"
    )

    # Create fast lookup dicts for IMDb title ratings & basics
    imdb_rating_map: dict[str, tuple[float, int]] = {}
    for r in imdb_ratings.iter_rows():
        imdb_rating_map[r[0]] = (float(r[1]), int(r[2]))

    imdb_type_map: dict[str, tuple[str, str, str | None, int | None, str | None]] = {}
    basics_rated = imdb_basics.join(imdb_ratings, on="tconst", how="inner")
    for r in basics_rated.iter_rows():
        # tconst -> (titleType, primaryTitle, originalTitle, startYear, genres)
        imdb_type_map[r[0]] = (r[1], r[2], r[3], r[4], r[5])

    # 2. Aggregate IMDb Episode Ratings by (parentTconst, seasonNumber)
    console.print("[cyan]→ Aggregating IMDb episode-level ratings into seasons & split-cours...[/cyan]")
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
        ((pl.col("averageRating") * pl.col("numVotes")).sum() / pl.col("numVotes").sum()).alias("ep_weighted_avg"),
        pl.col("averageRating").max().alias("ep_peak"),
        pl.col("averageRating").min().alias("ep_min"),
        pl.col("startYear").min().alias("season_year"),
    ])

    # Lookup map: (parentTconst, seasonNumber) -> (ep_avg, ep_votes, ep_count, ep_peak, season_year)
    season_rollup_map: dict[tuple[str, int], tuple[float, int, int, float, int | None]] = {}
    for r in season_agg_df.iter_rows(named=True):
        season_rollup_map[(r["parentTconst"], int(r["seasonNumber"]))] = (
            float(r["ep_avg"]),
            int(r["ep_votes"]),
            int(r["ep_count"]),
            float(r["ep_peak"]),
            r["season_year"],
        )

    # Also build per-episode list for parents that have split-cours in MAL
    with open("data/imdb/anime-list-mini.json") as f:
        alist = json.load(f)

    split_cour_parents: set[str] = set()
    mal_fribb_map: dict[int, dict] = {}
    for item in alist:
        mid = item.get("mal_id")
        imdb_raw = item.get("imdb_id")
        if mid and imdb_raw:
            tc = imdb_raw[0] if isinstance(imdb_raw, list) else imdb_raw
            if isinstance(tc, str) and tc.startswith("tt"):
                s_info = item.get("season") or {}
                s_num = s_info.get("tvdb") or s_info.get("tmdb")
                off_info = item.get("episode_offset") or {}
                ep_off = off_info.get("tvdb") or off_info.get("tmdb") or 0
                if ep_off and ep_off > 0:
                    split_cour_parents.add(tc)
                mal_fribb_map[int(mid)] = {
                    "tconst": tc,
                    "season": int(s_num) if s_num is not None else None,
                    "ep_offset": int(ep_off) if ep_off else 0,
                    "type": item.get("type", ""),
                }

    fribb_tconst_set = {v["tconst"] for v in mal_fribb_map.values() if v.get("tconst")}
    fribb_basics_unrated = imdb_basics.filter(pl.col("tconst").is_in(list(fribb_tconst_set)))
    for r in fribb_basics_unrated.iter_rows():
        if r[0] not in imdb_type_map:
            imdb_type_map[r[0]] = (r[1], r[2], r[3], r[4], r[5])

    # Build episode slice map for split_cour_parents
    ep_slice_df = ep_rated.filter(pl.col("parentTconst").is_in(list(split_cour_parents)))
    parent_season_eps: dict[tuple[str, int], list[tuple[int, float, int]]] = {}
    for r in ep_slice_df.iter_rows(named=True):
        en = r["episodeNumber"]
        if en is not None:
            key = (r["parentTconst"], int(r["seasonNumber"]))
            parent_season_eps.setdefault(key, []).append((int(en), float(r["averageRating"]), int(r["numVotes"])))

    # Build Animation Title Index from IMDb for fallback MAL matching
    # Restrict strictly to full works (movies, series, miniseries, tv specials) - NEVER tvEpisode or videoGame
    valid_work_types = {"movie", "tvSeries", "tvMiniSeries", "tvMovie", "tvSpecial"}
    anim_imdb = basics_rated.filter(
        pl.col("genres").str.contains("Animation")
        & pl.col("titleType").is_in(list(valid_work_types))
    ).sort("numVotes", descending=True)
    imdb_anim_title_map: dict[str, list[tuple[str, int | None, str, str]]] = {}
    for r in anim_imdb.iter_rows(named=True):
        tc = r["tconst"]
        yr = r["startYear"]
        tt = r["titleType"]
        for cand in (r["primaryTitle"], r["originalTitle"]):
            nk = normalize_title_key(cand)
            if nk:
                imdb_anim_title_map.setdefault(nk, []).append((tc, yr, tt, cand))

    # 3. Load MovieLens 32M & MAL 57M Calibrated Artifacts
    console.print("[cyan]→ Loading calibrated MovieLens 32M & MyAnimeList 57M artifacts...[/cyan]")
    ml_df = pl.read_parquet("artifacts/ml-32m/calibrated_scores.parquet")
    links_df = pl.read_csv("data/ml-32m/links.csv", schema_overrides={"imdbId": pl.String}).with_columns(
        (pl.lit("tt") + pl.col("imdbId").str.zfill(7)).alias("tconst")
    )
    ml_df = ml_df.join(links_df.select(["movieId", "tconst"]), on="movieId", how="left")

    mal_df = pl.read_parquet("artifacts/mal-57m/calibrated_scores.parquet")
    anime_meta_df = pl.read_parquet("data/mal-57m/anime_meta.parquet")
    anime_meta_map: dict[int, tuple[str, str, int | None]] = {}
    # Extract anime start year from data/mal-57m/anime.csv if available
    anime_years: dict[int, int] = {}
    anime_eps_map: dict[int, int] = {}
    anime_csv_path = Path("data/mal-57m/anime.csv")
    if anime_csv_path.exists():
        raw_anime_csv = pl.read_csv(anime_csv_path, ignore_errors=True)
        for r in raw_anime_csv.iter_rows(named=True):
            aid = int(r["MAL_ID"])
            aired = str(r.get("Aired") or "")
            ym = re.search(r"\b(19\d{2}|20\d{2})\b", aired)
            if ym:
                anime_years[aid] = int(ym.group(1))
            eps_str = str(r.get("Episodes") or "")
            if eps_str.isdigit():
                anime_eps_map[aid] = int(eps_str)

    for r in anime_meta_df.iter_rows(named=True):
        aid = int(r["anime_id"])
        en = r["name_en"] if r["name_en"] and r["name_en"] != "Unknown" else ""
        jp = r["name_jp"] if r["name_jp"] and r["name_jp"] != "Unknown" else ""
        anime_meta_map[aid] = (en, jp, anime_years.get(aid))

    # Load AniList 2025 Database (15K+ high-precision 100-point scores)
    console.print("[cyan]→ Loading AniList 2025 Multi-Platform Database...[/cyan]")
    anilist_map: dict[int, dict] = {}
    ani_path = Path("data/anilist/anilist_2025.jsonl")
    if ani_path.exists():
        with open(ani_path) as f:
            for line in f:
                d = json.loads(line)
                id_mal = d.get("idMal")
                avg_score = d.get("averageScore")
                if id_mal and avg_score is not None:
                    anilist_map[int(id_mal)] = {
                        "al_score": round(float(avg_score) / 10.0, 2),
                        "al_votes": int(d.get("popularity", 0)),
                        "cover_image": d.get("coverImage_large"),
                    }
    console.print(f"  [green]✓[/green] Loaded {len(anilist_map):,} AniList 2025 records")

    # Load MyAnimeList 2024 Complete Dataset (28K+ titles up to 2024/2025)
    console.print("[cyan]→ Loading MyAnimeList 2024/2025 Complete Anime Archive...[/cyan]")
    mal_2024_map: dict[int, dict] = {}
    mal_2024_path = Path("data/mal-2024/mal_2024.csv")
    if mal_2024_path.exists():
        df_2024 = pl.read_csv(mal_2024_path, ignore_errors=True)
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

            aired = str(r.get("Aired") or "")
            ym = re.search(r"\b(19\d{2}|20\d{2})\b", aired)
            yr = int(ym.group(1)) if ym else None

            mal_2024_map[aid] = {
                "title": r.get("Title") or "Unknown",
                "score": score_flt,
                "members": members_int,
                "genres": r.get("Genres") or "Anime",
                "en_name": r.get("English") or "",
                "jp_name": r.get("Japanese") or "",
                "type": r.get("Type") or "TV",
                "year": yr,
                "episodes": r.get("Episodes"),
            }
    console.print(f"  [green]✓[/green] Loaded {len(mal_2024_map):,} MAL 2024/2025 records")

    # Load Douban 845M Database (30K+ titles calibrated with Bayesian shrinkage & genre latent quality)
    console.print("[cyan]→ Loading Douban 845M Asian & World Cinema Benchmark...[/cyan]")
    douban_map: dict[str, dict] = {}
    douban_id_map: dict[int, dict] = {}
    douban_path = Path("artifacts/douban/calibrated_scores.parquet")
    douban_df = pl.DataFrame()
    if douban_path.exists():
        douban_df = pl.read_parquet(douban_path)
        for r in douban_df.iter_rows(named=True):
            mid = int(r["movieId"])
            tc = r.get("tconst")
            douban_id_map[mid] = r
            if tc:
                douban_map[tc] = r
    console.print(f"  [green]✓[/green] Loaded {len(douban_id_map):,} Douban titles ({len(douban_map):,} with IMDb tconst)")

    # Index 1-to-1 MAL Movies by tconst so we can merge them with MovieLens 32M Bridge Films
    mal_by_id: dict[int, dict] = {int(r["movieId"]): r for r in mal_df.iter_rows(named=True)}

    # Add missing modern anime from MAL 2024 (e.g. Frieren, Chainsaw Man, Oshi no Ko, Bocchi, JJK S2)
    modern_added = 0
    for aid, r_24 in mal_2024_map.items():
        if aid not in mal_by_id and r_24["members"] >= 2000 and r_24["score"] is not None and r_24["score"] >= 5.0:
            raw_s = r_24["score"]
            mem = r_24["members"]
            w_b = mem / (mem + 8000.0)
            bayes_s = w_b * raw_s + (1.0 - w_b) * 6.85
            mod_a = 0.95 * bayes_s
            mod_b = bayes_s - 0.30
            mod_c = mod_b - 0.05
            mal_by_id[aid] = {
                "movieId": aid,
                "title": r_24["title"],
                "genres": (r_24["genres"] or "Anime").replace(", ", "|"),
                "vote_count": mem,
                "raw_avg": raw_s,
                "bayesian_avg": bayes_s,
                "score_model_a": mod_a,
                "score_model_b": mod_b,
                "score_model_c": mod_c,
                "bias_b_i": mod_b - 6.85,
            }
            if aid not in anime_meta_map:
                anime_meta_map[aid] = (r_24["en_name"], r_24["jp_name"], r_24["year"])
            modern_added += 1
    console.print(f"  [green]✓[/green] Ingested {modern_added:,} modern 2021–2025 anime into calibration pool")
    tconst_to_mal_movie: dict[str, int] = {}
    all_mal_tconsts: set[str] = set()

    mal_resolved_info: dict[int, dict] = {}
    for mid, r in mal_by_id.items():
        en_name, jp_name, a_year = anime_meta_map.get(mid, ("", "", None))
        f_info = mal_fribb_map.get(mid)
        tconst = f_info["tconst"] if f_info else None
        s_num = f_info["season"] if f_info else None
        ep_off = f_info["ep_offset"] if f_info else 0
        m_type = f_info["type"] if f_info else ""

        # 1. Reject Fribb links if they point to an individual tvEpisode, videoGame, or short
        if tconst:
            im_meta = imdb_type_map.get(tconst)
            if im_meta and im_meta[0] in ("tvEpisode", "videoGame", "short"):
                tconst = None

        # 2. Guarded fallback title matching
        if not tconst:
            anime_yr = a_year or extract_year(r["title"])
            for cand in (en_name, r["title"]):
                if not cand:
                    continue
                nk = normalize_title_key(cand)
                # Disallow generic single-word or short titles (e.g. "Metamorphosis", "Pride", "Trust")
                if nk and len(nk.split()) >= 2 and len(nk) >= 7 and nk in imdb_anim_title_map:
                    cands = imdb_anim_title_map[nk]
                    best_match = None
                    if anime_yr is not None:
                        for c_tc, c_yr, c_tt, _ in cands:
                            if c_yr is not None and abs(c_yr - anime_yr) <= 1:
                                best_match = c_tc
                                break
                    elif len(cands) == 1:
                        best_match = cands[0][0]

                    if best_match:
                        tconst = best_match
                        break

        if s_num is None:
            s_num = infer_season_number(en_name or r["title"]) or 1

        if tconst:
            all_mal_tconsts.add(tconst)
            # If it's a feature film (or single entry without sequels), allow 1-to-1 bridge merge with MovieLens
            im_meta = imdb_type_map.get(tconst)
            im_type = im_meta[0] if im_meta else ""
            if m_type == "MOVIE" or im_type in ("movie", "tvMovie"):
                # Keep highest vote count MAL entry as primary movie match
                prev_mid = tconst_to_mal_movie.get(tconst)
                if prev_mid is None or r["vote_count"] > mal_by_id[prev_mid]["vote_count"]:
                    tconst_to_mal_movie[tconst] = mid

        mal_resolved_info[mid] = {
            "tconst": tconst,
            "season": s_num,
            "ep_offset": ep_off,
            "type": m_type,
            "en_name": en_name,
            "jp_name": jp_name,
            "year": a_year or extract_year(r["title"]),
            "ep_total_mal": anime_eps_map.get(mid),
        }

    # 4. Build Unified Records across the Cohorts
    unified_rows: list[dict] = []
    merged_mal_ids: set[int] = set()
    merged_ml_tconsts: set[str] = set()
    merged_douban_ids: set[int] = set()
    merged_douban_tconsts: set[str] = set()

    # --- COHORT 1: MovieLens 32M Cinema & Quint-DB Bridge Movies ---
    console.print("[cyan]→ Synthesizing Cohort 1: Cinema & Quint-DB Bridge Films (MovieLens + IMDb + MAL + AniList + Douban)...[/cyan]")
    for r in ml_df.iter_rows(named=True):
        tconst = r["tconst"]
        im_meta = imdb_type_map.get(tconst) if tconst else None
        im_type = im_meta[0] if im_meta else "movie"

        # Skip TV series / miniseries in MovieLens if they are handled in Cohort 2 (MAL) or Cohort 3 (IMDb TV Seasons)
        im_rating_pair = imdb_rating_map.get(tconst) if tconst else None
        imdb_raw = im_rating_pair[0] if im_rating_pair else None
        imdb_votes = im_rating_pair[1] if im_rating_pair else None

        if im_type in ("tvSeries", "tvMiniSeries") and tconst:
            if tconst in all_mal_tconsts:
                continue
            if (tconst, 1) in season_rollup_map and imdb_votes and imdb_votes >= 25000:
                s1_cnt = season_rollup_map[(tconst, 1)][2]
                s1_votes = season_rollup_map[(tconst, 1)][1]
                if s1_cnt >= 3 and s1_votes >= 5000:
                    continue

        ml_raw = float(r["raw_avg"])
        ml_votes = int(r["vote_count"])

        # Check if this movie is also in MAL-57M (Bridge Film!)
        mal_mid = tconst_to_mal_movie.get(tconst) if tconst else None
        mal_row = mal_by_id.get(mal_mid) if mal_mid else None
        if mal_mid and mal_row:
            merged_mal_ids.add(mal_mid)
            mal_info = mal_resolved_info[mal_mid]
            mal_raw = float(mal_row["raw_avg"])
            mal_votes = int(mal_row["vote_count"])
            title_en = mal_info["en_name"] or clean_display_title(r["title"])
            title_jp = mal_info["jp_name"]
            media_type = "bridge"
        else:
            mal_raw = None
            mal_votes = None
            title_en = ""
            title_jp = ""
            media_type = "movie"

        # Check if film has Douban score
        douban_row = douban_map.get(tconst) if tconst else None
        if douban_row:
            douban_raw = float(douban_row["raw_avg"])
            douban_votes = int(douban_row["vote_count"])
            douban_latent = float(douban_row["score_model_b"])
            merged_douban_ids.add(int(douban_row["movieId"]))
            if tconst:
                merged_douban_tconsts.add(tconst)
            if not title_jp and douban_row.get("title_jp"):
                title_jp = douban_row["title_jp"]
        else:
            douban_raw = None
            douban_votes = None
            douban_latent = None

        if tconst:
            merged_ml_tconsts.add(tconst)

        # Calculate Universal Calibrated Scores (Model A, B, C)
        w_ml = ml_votes / (ml_votes + 1500.0)
        ml_latent = w_ml * (0.78 * float(r["score_model_a"]) + 0.22 * float(r["score_model_b"])) + (1.0 - w_ml) * 6.80

        if imdb_raw is not None and imdb_votes is not None:
            w_im = imdb_votes / (imdb_votes + 30000.0)
            im_cal = w_im * (imdb_raw - 0.32) + (1.0 - w_im) * 6.80
        else:
            im_cal = ml_latent

        # Check if bridge anime has AniList score
        al_info = anilist_map.get(mal_mid) if mal_mid else None
        al_raw = al_info["al_score"] if al_info else None
        al_votes = al_info["al_votes"] if al_info else None

        if mal_row is not None and mal_raw is not None and mal_votes is not None:
            # Bridge film present in ALL databases (ML + MAL + IMDb + optional Douban)
            mal_latent = float(mal_row["score_model_b"]) - 0.34
            total_votes = ml_votes + mal_votes + (al_votes or 0) + (imdb_votes or 0) + (douban_votes or 0)
            raw_comp = (
                ml_raw * ml_votes
                + mal_raw * mal_votes
                + ((al_raw or 0.0) * (al_votes or 0))
                + ((imdb_raw or 0.0) * (imdb_votes or 0))
                + ((douban_raw or 0.0) * (douban_votes or 0))
            ) / max(1, total_votes)
            if douban_latent is not None:
                mod_b = 0.40 * ml_latent + 0.30 * mal_latent + 0.15 * im_cal + 0.15 * douban_latent
                mod_a = 0.45 * float(r["score_model_a"]) + 0.45 * (float(mal_row["score_model_a"]) - 0.34) + 0.10 * float(douban_row["score_model_a"])
                mod_c = 0.45 * float(r["score_model_c"]) + 0.45 * (float(mal_row["score_model_c"]) - 0.34) + 0.10 * float(douban_row["score_model_c"])
            else:
                mod_b = 0.45 * ml_latent + 0.35 * mal_latent + 0.20 * im_cal
                mod_a = 0.50 * float(r["score_model_a"]) + 0.50 * (float(mal_row["score_model_a"]) - 0.34)
                mod_c = 0.50 * float(r["score_model_c"]) + 0.50 * (float(mal_row["score_model_c"]) - 0.34)
        else:
            total_votes = ml_votes + (imdb_votes or 0) + (douban_votes or 0)
            raw_comp = (
                ml_raw * ml_votes
                + ((imdb_raw or 0.0) * (imdb_votes or 0))
                + ((douban_raw or 0.0) * (douban_votes or 0))
            ) / max(1, total_votes)
            if douban_latent is not None:
                mod_b = 0.65 * ml_latent + 0.20 * im_cal + 0.15 * douban_latent
                mod_a = w_ml * float(r["score_model_a"]) + (1.0 - w_ml) * (0.6 * im_cal + 0.4 * douban_latent)
                mod_c = w_ml * float(r["score_model_c"]) + (1.0 - w_ml) * (0.6 * im_cal + 0.4 * douban_latent)
            else:
                mod_b = 0.78 * ml_latent + 0.22 * im_cal
                mod_a = w_ml * float(r["score_model_a"]) + (1.0 - w_ml) * im_cal
                mod_c = w_ml * float(r["score_model_c"]) + (1.0 - w_ml) * im_cal

        bayes = (total_votes / (total_votes + 10000.0)) * raw_comp + (10000.0 / (total_votes + 10000.0)) * 6.85
        year = extract_year(r["title"]) or (im_meta[3] if im_meta else None)

        unified_rows.append({
            "movieId": int(r["movieId"]),
            "title": clean_display_title(r["title"]),
            "year": year,
            "genres": r["genres"],
            "vote_count": total_votes,
            "raw_avg": float(raw_comp),
            "bayesian_avg": float(bayes),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(mod_b - 6.85),
            "title_en": title_en,
            "title_jp": title_jp,
            "media_type": media_type,
            "tconst": tconst or "",
            "season_num": None,
            "ep_count": None,
            "ml_raw": round(ml_raw, 2),
            "ml_votes": ml_votes,
            "mal_raw": round(mal_raw, 2) if mal_raw is not None else None,
            "mal_votes": mal_votes,
            "al_raw": round(al_raw, 2) if al_raw is not None else None,
            "al_votes": al_votes,
            "douban_raw": round(douban_raw, 2) if douban_raw is not None else None,
            "douban_votes": douban_votes,
            "imdb_raw": round(imdb_raw, 2) if imdb_raw is not None else None,
            "imdb_votes": imdb_votes,
            "imdb_ep_avg": None,
            "imdb_ep_votes": None,
            "imdb_ep_peak": None,
        })

    # --- COHORT 2: MyAnimeList 57M Seasons, Split-Cours, OVAs & Anime Films ---
    console.print("[cyan]→ Synthesizing Cohort 2: Anime Seasons & Entries (MAL 57M + IMDb Season Rollups)...[/cyan]")
    for mid, r in mal_by_id.items():
        if mid in merged_mal_ids:
            continue

        info = mal_resolved_info[mid]
        tconst = info["tconst"]
        s_num = info["season"]
        ep_off = info["ep_offset"]
        mal_raw = float(r["raw_avg"])
        mal_votes = int(r["vote_count"])

        im_rating_pair = imdb_rating_map.get(tconst) if tconst else None
        imdb_raw = im_rating_pair[0] if im_rating_pair else None
        imdb_votes = im_rating_pair[1] if im_rating_pair else None

        # Anomaly Discrepancy Guardrail:
        # If an anime entry has an extreme rating discrepancy vs its matched IMDb entry (|mal - imdb| >= 2.5)
        # where one platform has negligible votes (< 500) and the other has massive votes (> 2,500),
        # quarantine and unmatch to prevent corrupted composite ratings.
        if tconst and imdb_raw is not None and imdb_votes is not None:
            if abs(mal_raw - imdb_raw) >= 2.5 and (mal_votes < 500 and imdb_votes > 2500):
                tconst = None
                imdb_raw = None
                imdb_votes = None

        # Lookup IMDb Season / Split-Cour Episode Rollup
        ep_avg = None
        ep_votes = None
        ep_count = info["ep_total_mal"]
        ep_peak = None
        s_year = info["year"]

        if tconst and s_num is not None:
            if tconst in split_cour_parents and (tconst, s_num) in parent_season_eps:
                eps_list = parent_season_eps[(tconst, s_num)]
                mal_ep_target = info["ep_total_mal"] or 12
                if ep_off > 0:
                    sub_eps = [x for x in eps_list if x[0] > ep_off and x[0] <= ep_off + mal_ep_target + 2]
                else:
                    sub_eps = [x for x in eps_list if x[0] <= mal_ep_target]
                if sub_eps:
                    v_sum = sum(x[2] for x in sub_eps)
                    if v_sum > 0:
                        ep_avg = sum(x[1] * x[2] for x in sub_eps) / v_sum
                        ep_votes = v_sum
                        ep_count = len(sub_eps)
                        ep_peak = max(x[1] for x in sub_eps)

            if ep_avg is None and (tconst, s_num) in season_rollup_map:
                s_avg, s_votes, s_cnt, s_peak, sy = season_rollup_map[(tconst, s_num)]
                ep_avg = s_avg
                ep_votes = s_votes
                ep_count = ep_count or s_cnt
                ep_peak = s_peak
                s_year = s_year or sy

        if s_year is None and tconst and tconst in imdb_type_map:
            s_year = imdb_type_map[tconst][3]

        al_info = anilist_map.get(mid)
        al_raw = al_info["al_score"] if al_info else None
        al_votes = al_info["al_votes"] if al_info else None

        douban_row = douban_map.get(tconst) if tconst else None
        if douban_row:
            douban_raw = float(douban_row["raw_avg"])
            douban_votes = int(douban_row["vote_count"])
            merged_douban_ids.add(int(douban_row["movieId"]))
            if tconst:
                merged_douban_tconsts.add(tconst)
        else:
            douban_raw = None
            douban_votes = None

        # Calibrate MAL score onto Universal Scale:
        # 1. Subtract cross-platform bridge offset (-0.30)
        # 2. Subtract sequel survivorship inflation (-0.055 * ln(season_num)) for Season 2+
        sequel_penalty = 0.055 * math.log(max(1, s_num or 1))
        w_mal = mal_votes / (mal_votes + 2500.0)
        mal_cal_b = w_mal * (float(r["score_model_b"]) - 0.30 - sequel_penalty) + (1.0 - w_mal) * 6.80
        mal_cal_a = w_mal * (float(r["score_model_a"]) - 0.30 - sequel_penalty) + (1.0 - w_mal) * 6.80
        mal_cal_c = w_mal * (float(r["score_model_c"]) - 0.30 - sequel_penalty) + (1.0 - w_mal) * 6.80

        # Anchor with AniList if available (mitigates MAL single-platform fan wars)
        if al_raw is not None and al_votes and al_votes >= 2000:
            al_cal_b = al_raw - 0.25 - sequel_penalty
            w_al = min(0.35, al_votes / (al_votes + 25000.0))
            mal_cal_b = (1.0 - w_al) * mal_cal_b + w_al * al_cal_b

        # Anchor with IMDb Season Episode Average or Parent Rating if available
        if ep_avg is not None and ep_votes and ep_votes >= 1500 and ep_avg >= mal_raw - 0.45:
            im_ep_cal = (ep_avg - 0.55 - sequel_penalty)
            w_ep = min(0.18, ep_votes / (ep_votes + 50000.0))
            mod_b = (1.0 - w_ep) * mal_cal_b + w_ep * im_ep_cal
        elif imdb_raw is not None and imdb_votes and imdb_votes >= 1500:
            im_cal = imdb_raw - 0.45 - sequel_penalty
            w_im = min(0.15, imdb_votes / (imdb_votes + 50000.0))
            mod_b = (1.0 - w_im) * mal_cal_b + w_im * im_cal
        else:
            mod_b = mal_cal_b

        mod_a = mal_cal_a
        mod_c = mal_cal_c

        # Multi-Platform Consensus Calculation:
        al_num = (al_raw * al_votes) if (al_raw is not None and al_votes) else 0.0
        al_den = al_votes if (al_raw is not None and al_votes) else 0

        im_v_eff = min(imdb_votes, 200000) if imdb_votes else 0
        im_num = (imdb_raw * (im_v_eff * 0.35)) if (imdb_raw is not None and im_v_eff) else 0.0
        im_den = (im_v_eff * 0.35) if (imdb_raw is not None and im_v_eff) else 0

        total_votes = mal_votes + (al_votes or 0) + (ep_votes if ep_votes else (imdb_votes or 0))

        if ep_avg is not None and ep_votes and ep_avg >= mal_raw - 0.45:
            raw_comp = (mal_raw * mal_votes + al_num + ep_avg * ep_votes) / max(1, mal_votes + al_den + ep_votes)
        else:
            raw_comp = (mal_raw * mal_votes + al_num + im_num) / max(1, mal_votes + al_den + im_den)

        bayes = (total_votes / (total_votes + 10000.0)) * raw_comp + (10000.0 / (total_votes + 10000.0)) * 6.85
        genres_str = r["genres"]
        if "Anime" not in genres_str and "Animation" not in genres_str:
            genres_str = f"Anime|{genres_str}" if genres_str else "Anime"

        unified_rows.append({
            "movieId": 3_000_000 + mid,
            "title": clean_display_title(r["title"]),
            "year": s_year,
            "genres": genres_str,
            "vote_count": int(total_votes),
            "raw_avg": float(raw_comp),
            "bayesian_avg": float(bayes),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(mod_b - 6.85),
            "title_en": info["en_name"],
            "title_jp": info["jp_name"],
            "media_type": "anime",
            "tconst": tconst or "",
            "season_num": s_num if info["type"] != "MOVIE" else None,
            "ep_count": ep_count,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": round(mal_raw, 2),
            "mal_votes": mal_votes,
            "al_raw": round(al_raw, 2) if al_raw is not None else None,
            "al_votes": al_votes,
            "douban_raw": round(douban_raw, 2) if douban_raw is not None else None,
            "douban_votes": douban_votes,
            "imdb_raw": round(imdb_raw, 2) if imdb_raw is not None else None,
            "imdb_votes": imdb_votes,
            "imdb_ep_avg": round(ep_avg, 2) if ep_avg is not None else None,
            "imdb_ep_votes": ep_votes,
            "imdb_ep_peak": round(ep_peak, 2) if ep_peak is not None else None,
        })

    # --- COHORT 3: Season-Separated IMDb TV Series (Breaking Bad S1-S5, GoT S1-S8, The Wire, etc.) ---
    console.print("[cyan]→ Synthesizing Cohort 3: Season-Separated IMDb TV Series (from 10M Episode Ratings)...[/cyan]")
    # Check if any MovieLens entry was a miniseries/series so we can attach ml_raw & ml_votes to Season 1
    ml_tv_map: dict[str, tuple[float, int, float]] = {}
    for r in ml_df.iter_rows(named=True):
        if r["tconst"]:
            ml_tv_map[r["tconst"]] = (
                float(r["raw_avg"]),
                int(r["vote_count"]),
                0.70 * float(r["score_model_a"]) + 0.30 * float(r["score_model_b"]),
            )

    tv_season_id_counter = 6_000_000
    tv_seasons_added = 0
    for (parent_tc, s_num), (ep_avg, ep_votes, ep_cnt, ep_peak, s_year) in season_rollup_map.items():
        if parent_tc in all_mal_tconsts:
            continue
        if ep_cnt < 3 or ep_votes < 5000:
            continue
        p_meta = imdb_type_map.get(parent_tc)
        if not p_meta:
            continue
        p_type, p_title, p_orig, p_start_year, p_genres = p_meta
        if p_type not in ("tvSeries", "tvMiniSeries"):
            continue
        # Exclude Japanese animation series already represented in Anime cohort
        if p_genres and "Animation" in p_genres and p_orig and p_orig != p_title:
            continue
        p_rating_pair = imdb_rating_map.get(parent_tc)
        if not p_rating_pair:
            continue
        parent_raw, parent_votes = p_rating_pair
        if parent_votes < 25000:
            continue

        tv_season_id_counter += 1
        tv_seasons_added += 1

        # Determine if this show has multiple seasons or is a 1-season miniseries
        has_season_2 = (parent_tc, 2) in season_rollup_map
        if p_type == "tvMiniSeries" and not has_season_2 and s_num == 1:
            display_title = f"{p_title} (Mini-Series)"
        else:
            display_title = f"{p_title}: Season {s_num}"

        # Attach MovieLens score if it was a 1-season miniseries or Season 1 in ML
        ml_info = ml_tv_map.get(parent_tc) if (s_num == 1 and not has_season_2) else None
        ml_raw = ml_info[0] if ml_info else None
        ml_votes = ml_info[1] if ml_info else None

        douban_row = douban_map.get(parent_tc) if parent_tc else None
        if douban_row:
            douban_raw = float(douban_row["raw_avg"])
            douban_votes = int(douban_row["vote_count"])
            merged_douban_ids.add(int(douban_row["movieId"]))
            if parent_tc:
                merged_douban_tconsts.add(parent_tc)
        else:
            douban_raw = None
            douban_votes = None

        # Calibrate TV Season Score onto Universal Scale:
        # 1. Episode ratings on IMDb have +0.62 self-selection inflation over movie ratings
        # 2. Seasonal survivorship inflation: +0.045 * ln(s_num)
        # 3. Confidence shrinkage on both season episode votes and parent series votes
        sequel_drift = 0.045 * math.log(max(1, s_num))
        w_ep = ep_votes / (ep_votes + 35000.0)
        w_parent = parent_votes / (parent_votes + 60000.0)
        w_conf = w_ep * (0.65 + 0.35 * w_parent)

        if ep_avg < parent_raw - 0.35:
            # Season underperformed parent series (e.g., Game of Thrones S8: ep_avg=6.40 vs parent=9.20)
            effective_raw = 0.85 * ep_avg + 0.15 * parent_raw
        else:
            effective_raw = 0.50 * ep_avg + 0.50 * parent_raw

        cal_base = w_conf * (effective_raw - 0.62 - sequel_drift) + (1.0 - w_conf) * 6.80
        if ml_info:
            cal_base = 0.72 * cal_base + 0.28 * ml_info[2]

        mod_b = cal_base
        mod_a = w_conf * (ep_avg - 0.58 - sequel_drift) + (1.0 - w_conf) * 6.80
        mod_c = mod_b - 0.04

        raw_comp = ep_avg
        total_votes = ep_votes + (ml_votes or 0)
        bayes = (ep_votes / (ep_votes + 15000.0)) * ep_avg + (15000.0 / (ep_votes + 15000.0)) * parent_raw

        genres_clean = (p_genres or "Drama").replace(",", "|")
        if "TV-Season" not in genres_clean:
            genres_clean = f"TV-Season|{genres_clean}"

        unified_rows.append({
            "movieId": tv_season_id_counter,
            "title": display_title,
            "year": s_year or p_start_year,
            "genres": genres_clean,
            "vote_count": int(total_votes),
            "raw_avg": float(raw_comp),
            "bayesian_avg": float(bayes),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(mod_b - 6.85),
            "title_en": p_title if (p_orig and p_orig != p_title) else "",
            "title_jp": p_orig if (p_orig and p_orig != p_title) else "",
            "media_type": "tv_season",
            "tconst": parent_tc,
            "season_num": s_num,
            "ep_count": ep_cnt,
            "ml_raw": round(ml_raw, 2) if ml_raw is not None else None,
            "ml_votes": ml_votes,
            "mal_raw": None,
            "mal_votes": None,
            "al_raw": None,
            "al_votes": None,
            "douban_raw": round(douban_raw, 2) if douban_raw is not None else None,
            "douban_votes": douban_votes,
            "imdb_raw": round(parent_raw, 2),
            "imdb_votes": parent_votes,
            "imdb_ep_avg": round(ep_avg, 2),
            "imdb_ep_votes": ep_votes,
            "imdb_ep_peak": round(ep_peak, 2),
        })

    # --- COHORT 4: IMDb-Exclusive Cinema Releases (numVotes >= 2,000) ---
    console.print("[cyan]→ Synthesizing Cohort 4: IMDb Cinema Releases (numVotes >= 2,000)...[/cyan]")
    imdb_cinema_exclusive = (
        imdb_basics.filter(pl.col("titleType") == "movie")
        .join(imdb_ratings, on="tconst", how="inner")
        .filter(pl.col("numVotes") >= 2000)
        .filter(~pl.col("tconst").is_in(list(merged_ml_tconsts)))
        .filter(~pl.col("tconst").is_in(list(all_mal_tconsts)))
    )

    imdb_cinema_id_counter = 8_000_000
    imdb_cinema_added = 0

    for r in imdb_cinema_exclusive.iter_rows(named=True):
        imdb_cinema_id_counter += 1
        tconst = r["tconst"]
        primary = r["primaryTitle"] or "Unknown"
        original = r["originalTitle"]
        year = r["startYear"]
        raw = float(r["averageRating"])
        votes = int(r["numVotes"])

        genres_str = (r["genres"] or "Drama").replace(",", "|")
        if genres_str == r"\N" or not genres_str:
            genres_str = "Drama"

        # Check if this IMDb cinema release also has a Douban score
        douban_row = douban_map.get(tconst) if tconst else None
        if douban_row:
            douban_raw = float(douban_row["raw_avg"])
            douban_votes = int(douban_row["vote_count"])
            merged_douban_ids.add(int(douban_row["movieId"]))
            if tconst:
                merged_douban_tconsts.add(tconst)
            if not original and douban_row.get("title_jp"):
                original = douban_row["title_jp"]
        else:
            douban_raw = None
            douban_votes = None

        # De-bias IMDb cinema rating onto engine's Universal Scale (1-10)
        # Consistent with MovieLens cinema calibration where IMDb raw has -0.32 offset and shrinkage towards 6.80
        w_im = votes / (votes + 30000.0)
        im_cal = w_im * (raw - 0.32) + (1.0 - w_im) * 6.80

        if douban_raw is not None and douban_votes is not None:
            w_db = douban_votes / (douban_votes + 30000.0)
            db_cal = w_db * (douban_raw - 0.20) + (1.0 - w_db) * 6.80
            mod_b = 0.55 * im_cal + 0.45 * db_cal
            mod_a = mod_b
            mod_c = mod_b - 0.05
            total_votes = votes + douban_votes
            raw_comp = (raw * votes + douban_raw * douban_votes) / total_votes
            bayes = (total_votes / (total_votes + 10000.0)) * raw_comp + (10000.0 / (total_votes + 10000.0)) * 6.85
        else:
            mod_b = float(im_cal)
            mod_a = float(im_cal)
            mod_c = float(im_cal - 0.05)
            total_votes = votes
            raw_comp = raw
            bayes = (votes / (votes + 10000.0)) * raw + (10000.0 / (votes + 10000.0)) * 6.85

        # Native / Original title formatting:
        # If original title is in Latin script (e.g. 'Tere Naam', 'Dabangg 3') but primary is an English translation ('In Your Name'),
        # promote the recognized original title as primary display.
        if original and original != primary:
            is_latin = False
            try:
                original.encode("latin-1")
                is_latin = True
            except UnicodeEncodeError:
                is_latin = False

            if is_latin and len(original.strip()) > 0:
                display_title = original
                title_en = primary
                title_jp = original
            else:
                display_title = primary
                title_en = primary
                title_jp = original
        else:
            display_title = primary
            title_en = ""
            title_jp = ""

        unified_rows.append({
            "movieId": imdb_cinema_id_counter,
            "title": clean_display_title(display_title),
            "year": year,
            "genres": genres_str,
            "vote_count": total_votes,
            "raw_avg": float(raw_comp),
            "bayesian_avg": float(bayes),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(mod_b - 6.85),
            "title_en": title_en,
            "title_jp": title_jp,
            "media_type": "movie",
            "tconst": tconst,
            "season_num": None,
            "ep_count": None,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": None,
            "mal_votes": None,
            "al_raw": None,
            "al_votes": None,
            "douban_raw": round(douban_raw, 2) if douban_raw is not None else None,
            "douban_votes": douban_votes,
            "imdb_raw": round(raw, 2),
            "imdb_votes": votes,
            "imdb_ep_avg": None,
            "imdb_ep_votes": None,
            "imdb_ep_peak": None,
        })
        imdb_cinema_added += 1

    # --- COHORT 5: Douban High-Consensus Releases (vote_count >= 2,000) ---
    console.print("[cyan]→ Synthesizing Cohort 5: Douban High-Consensus Releases (vote_count >= 2,000)...[/cyan]")
    douban_cinema_id_counter = 9_000_000
    douban_added = 0

    for r in douban_df.iter_rows(named=True):
        mid = int(r["movieId"])
        tc = r.get("tconst") or ""
        v_cnt = int(r["vote_count"])

        # Check if already merged in Cohorts 1-4
        if mid in merged_douban_ids or (tc and tc in merged_douban_tconsts) or (tc and tc in merged_ml_tconsts) or (tc and tc in all_mal_tconsts):
            continue

        # Filter high-consensus threshold (>= 2,000 votes, same standard as Cohort 4)
        if v_cnt < 2000:
            continue

        douban_cinema_id_counter += 1
        douban_added += 1

        db_raw = float(r["raw_avg"])
        db_votes = v_cnt
        db_bayes = float(r["bayesian_avg"])
        db_mod_a = float(r["score_model_a"])
        db_mod_b = float(r["score_model_b"])
        db_mod_c = float(r["score_model_c"])

        # Check if IMDb has ratings for this tconst even if < 2,000 IMDb votes
        im_pair = imdb_rating_map.get(tc) if tc else None
        im_raw = im_pair[0] if im_pair else None
        im_votes = im_pair[1] if im_pair else None

        if im_raw is not None and im_votes is not None:
            total_votes = db_votes + im_votes
            raw_comp = (db_raw * db_votes + im_raw * im_votes) / total_votes
            w_im = im_votes / (im_votes + 30000.0)
            im_cal = w_im * (im_raw - 0.32) + (1.0 - w_im) * 6.80
            w_db_share = db_votes / total_votes
            mod_b = w_db_share * db_mod_b + (1.0 - w_db_share) * im_cal
            mod_a = w_db_share * db_mod_a + (1.0 - w_db_share) * im_cal
            mod_c = w_db_share * db_mod_c + (1.0 - w_db_share) * (im_cal - 0.05)
            bayes = (total_votes / (total_votes + 10000.0)) * raw_comp + (10000.0 / (total_votes + 10000.0)) * 6.85
        else:
            total_votes = db_votes
            raw_comp = db_raw
            mod_b = db_mod_b
            mod_a = db_mod_a
            mod_c = db_mod_c
            bayes = db_bayes

        unified_rows.append({
            "movieId": douban_cinema_id_counter,
            "title": clean_display_title(r["title"]),
            "year": r["year"],
            "genres": r["genres"] or "Drama",
            "vote_count": total_votes,
            "raw_avg": float(raw_comp),
            "bayesian_avg": float(bayes),
            "score_model_a": float(mod_a),
            "score_model_b": float(mod_b),
            "score_model_c": float(mod_c),
            "bias_b_i": float(mod_b - 6.85),
            "title_en": r["title_en"] or "",
            "title_jp": r["title_jp"] or "",
            "media_type": "movie",
            "tconst": tc,
            "season_num": None,
            "ep_count": None,
            "ml_raw": None,
            "ml_votes": None,
            "mal_raw": None,
            "mal_votes": None,
            "al_raw": None,
            "al_votes": None,
            "douban_raw": round(db_raw, 2),
            "douban_votes": db_votes,
            "imdb_raw": round(im_raw, 2) if im_raw is not None else None,
            "imdb_votes": im_votes,
            "imdb_ep_avg": None,
            "imdb_ep_votes": None,
            "imdb_ep_peak": None,
        })

    console.print(
        f"  [green]✓[/green] Unified rows built: {len(unified_rows):,} total "
        f"({len(merged_mal_ids):,} Bridge titles, {tv_seasons_added:,} IMDb TV Seasons, "
        f"{imdb_cinema_added:,} IMDb Cinema Releases, {douban_added:,} Douban Cinema Releases)"
    )

    # 5. Convert to Polars DataFrame and compute Ranks, Deltas & Diagnostics
    df = pl.DataFrame(unified_rows, infer_schema_length=None)

    # Sort by raw_avg descending to assign raw_rank
    df = df.sort(["raw_avg", "vote_count"], descending=[True, True]).with_row_index("raw_rank", offset=1)
    # Sort by score_model_b descending to assign calibrated_rank
    df = df.sort(["score_model_b", "vote_count"], descending=[True, True]).with_row_index("calibrated_rank", offset=1)

    # Compute rank_delta, calibration_delta, score_model_b_rescaled, and diagnostic text
    b_min = float(df["score_model_b"].min())
    b_max = float(df["score_model_b"].max())
    df = df.with_columns([
        (pl.col("calibrated_rank").cast(pl.Int64) - pl.col("raw_rank").cast(pl.Int64)).alias("rank_delta"),
        (pl.col("score_model_b") - pl.col("raw_avg")).alias("calibration_delta"),
        (1.0 + 9.0 * (pl.col("score_model_b") - b_min) / max(1e-6, b_max - b_min)).alias("score_model_b_rescaled"),
        pl.lit(1.0).alias("genre_multiplier"),
        pl.lit(0.0).alias("mean_z"),
    ])

    # Build informative diagnostic strings explaining WHY the score changed
    diags: list[str] = []
    for r in df.iter_rows(named=True):
        r_delta = int(r["rank_delta"])
        c_delta = float(r["calibration_delta"])
        mtype = r["media_type"]
        snum = r["season_num"]
        if mtype == "bridge":
            src_note = "Quint-DB Bridge (ML+MAL+AL+IMDb+Douban)" if r.get("douban_raw") is not None else "Quad-DB Bridge (ML+MAL+AL+IMDb)"
        elif mtype == "anime":
            src_note = f"Anime S{snum} sequel bias removed" if (snum and snum > 1) else "MAL+AL+IMDb consensus calibrated"
        elif mtype == "tv_season":
            src_note = f"TV S{snum} episode rollup calibrated"
        elif r.get("douban_raw") is not None and r.get("ml_raw") is None and r.get("mal_raw") is None:
            src_note = "Douban+IMDb cinema calibrated" if r.get("imdb_raw") is not None else "Douban 豆瓣 cinema calibrated"
        elif r["ml_raw"] is None:
            src_note = "IMDb cinema calibrated"
        else:
            src_note = "ML+IMDb+Douban cinema calibrated" if r.get("douban_raw") is not None else "ML+IMDb cinema calibrated"

        if r_delta > 25:
            diags.append(f"Dropped {r_delta:,} ranks ({src_note}; enthusiast inflation de-biased)")
        elif r_delta < -25:
            diags.append(f"Gained {abs(r_delta):,} ranks ({src_note}; harsh rater drag corrected)")
        else:
            diags.append(f"Stable rank ({src_note}; Δ={c_delta:+.2f})")

    df = df.with_columns(pl.Series("rank_diagnostic", diags))

    # 6. Save Unified Artifacts to artifacts/combined/
    out_dir = Path("artifacts/combined")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "calibrated_scores.parquet")
    df.write_csv(out_dir / "calibrated_scores.csv")

    # Compute Genre Summary for combined
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
            # Legacy alias names for compatibility
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
    # Add genre multiplier estimate and 'genres' column name alias
    global_mod_b = float(df["score_model_b"].mean())
    genre_summary = genre_summary.with_columns([
        (global_mod_b / pl.col("avg_raw_rating")).clip(0.85, 1.15).alias("genre_multiplier"),
        pl.col("genre").alias("genres"),
    ])
    genre_summary.write_parquet(out_dir / "genre_summary.parquet")

    metadata = {
        "dataset": "combined",
        "sources": ["ml-32m", "mal-57m", "anilist-2025", "imdb-official-episodes", "imdb-cinema-exclusive", "douban-845m"],
        "total_titles": len(df),
        "total_ratings_processed": 31_637_476 + 57_633_278 + int(imdb_ratings["numVotes"].sum()) + int(douban_df["vote_count"].sum() if len(douban_df) > 0 else 0),
        "user_level_ratings": 31_637_476 + 57_633_278 + int(douban_df["vote_count"].sum() if len(douban_df) > 0 else 0),
        "imdb_rated_titles": len(imdb_ratings),
        "imdb_episodes_aggregated": len(imdb_episodes),
        "bridge_titles_count": len(merged_mal_ids),
        "tv_seasons_count": tv_seasons_added,
        "imdb_cinema_count": imdb_cinema_added,
        "douban_cinema_count": douban_added,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(
        f"[bold green]✓ Saved Unified Master Database to {out_dir} in {time.time() - t_start:.2f}s![/bold green]"
    )


if __name__ == "__main__":
    build_unified_database()
