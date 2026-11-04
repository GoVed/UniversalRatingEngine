"""
unify_semantic_tags.py - Classify and unify narrative & thematic tags across all Cinema, Anime, and TV titles using all-MiniLM-L6-v2.
Outputs:
  Enriched movies_*.json and vibe_titles.json with 100% complete Universal Tags across all datasets.
"""

import json
import os
import time
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 22 Universal Canonical Tags
TAG_TAXONOMY = {
    "Mind-Bending & Twists": "Shocking plot twists, unreliable narrators, mind-bending revelations, reality-warping mysteries, and deception.",
    "Slow Burn & Climax": "Patient deliberate build-up, gradual unraveling of clues, atmospheric pacing, escalating into an explosive grand finale.",
    "Time Travel & Paradox": "Time loops, timelines, alternate reality, temporal paradoxes, past and future collision, and causal loops.",
    "Psychological Warfare": "Intense mind games, intellectual cat-and-mouse duels, manipulation, paranoia, psychological thriller, and mental breakdown.",
    "Cerebral & Philosophical": "Existential themes, moral dilemmas, philosophical inquiries into human condition, ethics, and consciousness.",
    "Dark Mystery & Whodunit": "Detective crime investigations, murder mysteries, cryptic conspiracies, noir sleuths, and hidden secrets.",
    "Existential Dread & Cosmic": "Atmospheric dread, bleak cosmic insignificance, grim reality, isolation, existential terror, and psychological void.",
    "Cyberpunk & Dystopia": "High-tech low-life, artificial intelligence, transhumanism, surveillance state, totalitarian oppression, and neon megacorps.",
    "Post-Apocalyptic Survival": "Ruined civilization, wasteland survival, societal collapse, aftermath of apocalypse, and desperate endurance.",
    "High Adrenaline & Action": "Relentless action, explosive combat, high-stakes chases, adrenaline-pumping thrills, and visceral martial arts.",
    "Found Family & Camaraderie": "Bonds of unlikely companions, team loyalty, ragtag groups forming a family, brotherhood, and deep friendship.",
    "Cozy & Healing": "Wholesome comforting everyday life, gentle humor, heartwarming charm, and healing low-stakes warmth.",
    "Melancholy & Bittersweet": "Poignant grief, nostalgic longing, heartbreaking loss, emotional catharsis, sadness, and tragic love.",
    "Coming of Age": "Adolescent growth, school youth, loss of innocence, self-discovery, and personal maturity.",
    "Revenge & Obsession": "Destructive vengeance, personal obsession, vendettas, moral descent, retribution, and unrelenting grudge.",
    "Epic Fantasy & Myth": "High fantasy worlds, magic, mythical beasts, legendary quests, kingdoms, and supernatural lore.",
    "Political Intrigue & Strategy": "Power struggles, war tactics, courtly diplomacy, factions, betrayal, and tactical maneuvering.",
    "Romance & Heartbreak": "Passionate romance, heartbreak, emotional intimacy, forbidden love, and relationship dynamics.",
    "Crime & Underworld": "Mafia syndicates, heists, underworld gangsters, police corruption, drug trade, and criminal syndicates.",
    "Dark Comedy & Satire": "Cynical irony, absurdist humor, biting social satire, eccentric farce, and dark comedy.",
    "Space Odyssey & Sci-Fi": "Interstellar journey, galaxy exploration, future science, space fleets, and extraterrestrial alien encounters.",
    "Supernatural & Occult": "Ghosts, curses, spirits, demons, esoteric rituals, dark folklore, and otherworldly occult entities."
}

def main():
    print("=" * 60)
    print("UniversalRatingEngine • Neural Tag Unification Pipeline")
    print("=" * 60)

    # 1. Load Unified Plots
    plots_df = pl.read_parquet("data/unified_plots.parquet")
    print(f"Loaded {len(plots_df)} total plot records.")

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading transformer model on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    model.to(device)

    def embed_batch(texts, max_len=192):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_emb = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            return F.normalize(sum_emb / sum_mask, p=2, dim=1)

    # 3. Compute Embeddings for the 22 Canonical Tags
    tag_names = list(TAG_TAXONOMY.keys())
    tag_descriptions = list(TAG_TAXONOMY.values())
    tag_vectors = embed_batch(tag_descriptions)
    print(f"Computed reference embeddings for {len(tag_names)} universal tags.")

    # 4. Process all Titles in Batches from unified_plots
    titles_data = plots_df.to_dicts()
    batch_size = 256
    id_to_unified_tags = {}
    tc_to_unified_tags = {}
    title_yr_to_unified_tags = {}

    print(f"Classifying {len(titles_data)} unified plot titles in batches of {batch_size}...")
    t0 = time.time()

    for i in range(0, len(titles_data), batch_size):
        batch = titles_data[i:i + batch_size]
        batch_texts = []
        for r in batch:
            tl = f" {r['tagline']}" if r.get("tagline") else ""
            kw = f" {r['keywords']}" if r.get("keywords") else ""
            ov = r["overview"][:300] if r.get("overview") else ""
            txt = f"{r['title']}. {r['genres']}.{tl}{kw}. {ov}"
            batch_texts.append(txt)

        title_embs = embed_batch(batch_texts)
        sim_matrix = torch.mm(title_embs, tag_vectors.t()) # shape: (batch_size, 22)

        for b_idx in range(len(batch)):
            row = batch[b_idx]
            sims = sim_matrix[b_idx]
            topk = torch.topk(sims, k=3)
            
            top_tags = []
            for rank_i in range(3):
                tag_idx = topk.indices[rank_i].item()
                score = topk.values[rank_i].item()
                if rank_i == 0 or score >= 0.22:
                    top_tags.append(tag_names[tag_idx])

            id_to_unified_tags[row["id"]] = top_tags
            if row.get("tconst"):
                tc_to_unified_tags[row["tconst"]] = top_tags
            if row.get("title") and row.get("year"):
                title_yr_to_unified_tags[(row["title"].lower(), row["year"])] = top_tags

        done = min(i + batch_size, len(titles_data))
        rate = done / (time.time() - t0)
        print(f"  [{done}/{len(titles_data)}] ({done/len(titles_data)*100:.1f}%) - {rate:.1f} titles/sec", end="\r")

    print(f"\nUnified plots classification complete in {time.time()-t0:.2f}s")

    # 5. Update docs/data/vibe_titles.json
    vibe_titles_path = "docs/data/vibe_titles.json"
    if os.path.exists(vibe_titles_path):
        with open(vibe_titles_path) as f:
            v_titles = json.load(f)
        for item in v_titles:
            uid = item["id"]
            if uid in id_to_unified_tags:
                item["tags"] = id_to_unified_tags[uid]
        with open(vibe_titles_path, "w") as f:
            json.dump(v_titles, f, separators=(',', ':'))
        print(f"Updated {vibe_titles_path} with unified tags.")

    # 6. Update and Enrich All Datasets (including standalone)
    datasets = ["combined", "imdb", "mal-57m", "anilist", "ml-32m", "douban"]
    for ds in datasets:
        file_path = f"docs/data/movies_{ds}.json"
        if not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            data = json.load(f)
        
        cols = data["cols"]
        idx_id = cols.index("id")
        idx_tc = cols.index("tconst") if "tconst" in cols else None
        idx_title = cols.index("title")
        idx_year = cols.index("year")
        idx_genres = cols.index("genres")
        
        if "tags" not in cols:
            cols.append("tags")
            data["cols"] = cols
            idx_tags = len(cols) - 1
            for row in data["data"]:
                row.append([])
        else:
            idx_tags = cols.index("tags")

        # Pass 1: Map from unified precomputed tags
        unmatched_rows = []
        for row in data["data"]:
            uid = row[idx_id]
            tc = row[idx_tc] if idx_tc is not None else None
            ty = (str(row[idx_title]).lower(), row[idx_year])
            
            tags = (
                id_to_unified_tags.get(uid) or 
                id_to_unified_tags.get(3000000 + uid if isinstance(uid, int) else None) or
                tc_to_unified_tags.get(tc) or 
                title_yr_to_unified_tags.get(ty)
            )
            if tags:
                row[idx_tags] = tags
            else:
                unmatched_rows.append(row)

        print(f"\nDataset [{ds}]: {len(data['data']) - len(unmatched_rows)} matched via catalog. {len(unmatched_rows)} titles need neural classification.")

        # Pass 2: Classify unmatched titles in batches
        if unmatched_rows:
            t_start = time.time()
            for b_start in range(0, len(unmatched_rows), batch_size):
                b_slice = unmatched_rows[b_start:b_start + batch_size]
                b_texts = []
                for r in b_slice:
                    ge = str(r[idx_genres]).replace("|", ", ")
                    txt = f"{r[idx_title]} ({r[idx_year]}). Genres: {ge}."
                    b_texts.append(txt)

                embs = embed_batch(b_texts)
                sims = torch.mm(embs, tag_vectors.t())
                for b_i in range(len(b_slice)):
                    row = b_slice[b_i]
                    topk = torch.topk(sims[b_i], k=3)
                    top_tags = []
                    for rank_i in range(3):
                        tag_idx = topk.indices[rank_i].item()
                        score = topk.values[rank_i].item()
                        if rank_i == 0 or score >= 0.20:
                            top_tags.append(tag_names[tag_idx])
                    row[idx_tags] = top_tags

                done = min(b_start + batch_size, len(unmatched_rows))
                rate = done / (time.time() - t_start)
                print(f"  [{ds}] Classified {done}/{len(unmatched_rows)} ({done/len(unmatched_rows)*100:.1f}%) - {rate:.1f} titles/sec", end="\r")

            print(f"\n  [{ds}] Neural classification complete in {time.time()-t_start:.2f}s")

        with open(file_path, "w") as f:
            json.dump(data, f, separators=(',', ':'))
        print(f"Saved {file_path} with 100% unified tags.")

    # 7. Tag Distribution Statistics
    tag_counts = {}
    with open("docs/data/movies_combined.json") as f:
        comb_data = json.load(f)
    tags_idx = comb_data["cols"].index("tags")
    for r in comb_data["data"]:
        for t in r[tags_idx]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    print("\nUniversal Tag Distribution in Master Catalog:")
    for t, c in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t:30s}: {c:,} titles")

    print("\n[SUCCESS] All datasets updated with 100% Unified Universal Tags!")

if __name__ == "__main__":
    main()
