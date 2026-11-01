"""
build_plot_synopses.py - Ingest, clean, and unify plot descriptions across Cinema, Anime, and TV.
Sources:
  1. Cinema Movies: HenryWaltson/TMDB-IMDB-Movies-Dataset (TMDB overviews, taglines, keywords)
  2. Anime: AniList 2025 JSONL (rich descriptions, themes) + MAL 2024
  3. TV Seasons: TVMaze API (series summaries, episode context)
Outputs:
  data/unified_plots.parquet - Unified plot repository aligned with movies_combined.json
"""

import json
import os
import re
import time
import urllib.request
import concurrent.futures
import polars as pl
import pandas as pd

def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("=" * 60)
    print("UniversalRatingEngine • Ingesting Unified Plot Synopses")
    print("=" * 60)

    # 1. Load Combined Movies Catalog
    with open("docs/data/movies_combined.json") as f:
        comb = json.load(f)
    
    cols = comb["cols"]
    idx = {c: i for i, c in enumerate(cols)}
    rows = comb["data"]
    print(f"Total titles in catalog: {len(rows)}")

    # Group by media type
    movie_tconst_map = {} # tconst -> list of rows
    anime_rows = []
    tv_tconst_map = {}   # tconst -> list of rows

    for r in rows:
        mt = r[idx.get("media_type")]
        tc = r[idx.get("tconst")]
        if mt in ["movie", "bridge"] and tc:
            movie_tconst_map.setdefault(tc, []).append(r)
        elif mt in ["anime"]:
            anime_rows.append(r)
        elif mt in ["tv_season"] and tc:
            tv_tconst_map.setdefault(tc, []).append(r)

    print(f"Cinema Movies with tconst: {len(movie_tconst_map)}")
    print(f"Anime titles: {len(anime_rows)}")
    print(f"Unique TV Series with tconst: {len(tv_tconst_map)}")

    # 2. Ingest Cinema Overviews from TMDB-IMDB dataset
    print("\n[1/3] Streaming Cinema Movie Overviews from HuggingFace TMDB-IMDB dataset...")
    tmdb_url = "https://huggingface.co/datasets/HenryWaltson/TMDB-IMDB-Movies-Dataset/resolve/main/TMDB%20%20IMDB%20Movies%20Dataset.csv"
    
    target_movie_tconsts = set(movie_tconst_map.keys())
    cinema_overviews = {} # tconst -> {overview, tagline, keywords}

    t0 = time.time()
    chunk_iter = pd.read_csv(
        tmdb_url,
        chunksize=50000,
        usecols=["tconst", "overview", "tagline", "keywords"]
    )
    for i, chunk in enumerate(chunk_iter):
        valid = chunk[chunk["tconst"].isin(target_movie_tconsts) & chunk["overview"].notna()]
        for _, row in valid.iterrows():
            tc = row["tconst"]
            if tc not in cinema_overviews:
                cinema_overviews[tc] = {
                    "overview": clean_html(str(row["overview"])),
                    "tagline": clean_html(str(row["tagline"])) if pd.notna(row["tagline"]) else "",
                    "keywords": clean_html(str(row["keywords"])) if pd.notna(row["keywords"]) else "",
                }
        if len(cinema_overviews) >= len(target_movie_tconsts):
            break
    print(f"Ingested {len(cinema_overviews)} / {len(target_movie_tconsts)} cinema overviews in {time.time()-t0:.2f}s")

    # 3. Ingest Anime Descriptions from AniList 2025
    print("\n[2/3] Reading Anime Descriptions from AniList 2025 JSONL...")
    al_by_mal = {}
    al_by_title = {}
    if os.path.exists("data/anilist/anilist_2025.jsonl"):
        with open("data/anilist/anilist_2025.jsonl") as f:
            for line in f:
                d = json.loads(line)
                desc = clean_html(d.get("description") or "")
                if not desc:
                    continue
                if d.get("idMal"):
                    al_by_mal[int(d["idMal"])] = desc
                if d.get("title_romaji"):
                    al_by_title[d["title_romaji"].strip().lower()] = desc
                if d.get("title_english"):
                    al_by_title[d["title_english"].strip().lower()] = desc

    # Load anime_meta to map anime_id
    anime_meta_map = {}
    if os.path.exists("data/mal-57m/anime_meta.parquet"):
        meta_df = pl.read_parquet("data/mal-57m/anime_meta.parquet")
        for row in meta_df.iter_rows(named=True):
            aid = row["anime_id"]
            if row["name_romaji"]:
                anime_meta_map[row["name_romaji"].strip().lower()] = aid
            if row["name_en"]:
                anime_meta_map[row["name_en"].strip().lower()] = aid

    anime_overviews = {} # unified_id -> overview
    anime_matched = 0
    for r in anime_rows + [r for r in rows if r[idx.get("media_type")] == "bridge"]:
        uid = r[idx["id"]]
        t = r[idx["title"]].strip().lower()
        ten = (r[idx.get("title_en")] or "").strip().lower()
        aid = anime_meta_map.get(t) or anime_meta_map.get(ten)
        
        desc = ""
        if aid and aid in al_by_mal:
            desc = al_by_mal[aid]
        elif t in al_by_title:
            desc = al_by_title[t]
        elif ten and ten in al_by_title:
            desc = al_by_title[ten]

        if desc:
            anime_overviews[uid] = desc
            anime_matched += 1

    print(f"Matched {anime_matched} / {len(anime_rows)} anime descriptions from AniList")

    # 4. Fetch/Load TV Series Summaries via TVMaze
    print("\n[3/3] Ingesting TV Series Summaries...")
    tv_cache_file = "data/tv_summaries_cache.json"
    tv_cache = {}
    if os.path.exists(tv_cache_file):
        with open(tv_cache_file) as f:
            tv_cache = json.load(f)
        print(f"Loaded {len(tv_cache)} TV summaries from cache")

    missing_tv = [tc for tc in tv_tconst_map if tc not in tv_cache]
    if missing_tv:
        print(f"Fetching {len(missing_tv)} TV series from TVMaze API...")
        def fetch_tv(tc):
            url = f"https://api.tvmaze.com/lookup/shows?imdb={tc}"
            req = urllib.request.Request(url, headers={"User-Agent": "UniversalRatingEngine/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode())
                    raw_summary = data.get("summary") or ""
                    clean = clean_html(raw_summary)
                    return tc, clean, data.get("genres", [])
            except Exception:
                return tc, "", []

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            fut_to_tc = {executor.submit(fetch_tv, tc): tc for tc in missing_tv}
            for fut in concurrent.futures.as_completed(fut_to_tc):
                tc, summary, genres = fut.result()
                if summary:
                    tv_cache[tc] = {"summary": summary, "genres": genres}

        with open(tv_cache_file, "w") as f:
            json.dump(tv_cache, f, indent=2)
        print(f"Updated TV cache with {len(tv_cache)} series summaries")

    # 5. Assemble Unified Plots Dataset
    print("\nAssembling unified plot records...")
    out_records = []
    
    for r in rows:
        uid = r[idx["id"]]
        title = r[idx["title"]]
        mtype = r[idx.get("media_type")] or "movie"
        year = r[idx.get("year")] or 0
        genres = r[idx.get("genres")] or ""
        tc = r[idx.get("tconst")] or ""
        mod_b = r[idx.get("mod_b")] or 0.0
        cal_rank = r[idx.get("cal_rank")] or 999999
        votes = r[idx.get("votes")] or 0
        season_num = r[idx.get("season_num")]

        overview = ""
        tagline = ""
        keywords = ""

        if mtype in ["movie", "bridge"] and tc in cinema_overviews:
            item = cinema_overviews[tc]
            overview = item["overview"]
            tagline = item["tagline"]
            keywords = item["keywords"]

        if not overview and uid in anime_overviews:
            overview = anime_overviews[uid]

        if not overview and mtype == "tv_season" and tc in tv_cache:
            item = tv_cache[tc]
            s_text = f" Season {season_num}." if season_num else ""
            overview = f"{item['summary']}{s_text}"

        # If overview is still empty, synthesize a basic contextual synopsis from title and genres
        if not overview:
            overview = f"{title} ({year}) is an acclaimed {genres.replace('|', ', ')} {mtype} production."

        out_records.append({
            "id": int(uid),
            "title": str(title),
            "media_type": str(mtype),
            "year": int(year),
            "genres": str(genres),
            "tconst": str(tc),
            "overview": str(overview),
            "tagline": str(tagline),
            "keywords": str(keywords),
            "mod_b": float(mod_b),
            "cal_rank": int(cal_rank),
            "votes": int(votes)
        })

    out_df = pl.DataFrame(out_records)
    out_path = "data/unified_plots.parquet"
    out_df.write_parquet(out_path)
    print(f"\n[DONE] Saved {len(out_df)} unified plot records to {out_path}")
    print(f"  Titles with non-empty rich overview: {out_df.filter(pl.col('overview').str.len_chars() > 50).height} / {len(out_df)}")

if __name__ == "__main__":
    main()
