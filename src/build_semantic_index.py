"""
build_semantic_index.py - Generate 384-dim semantic embeddings and quantized Int8 index.
Outputs:
  docs/data/vibe_embeddings.bin - Int8 binary matrix (N x 384)
  docs/data/vibe_titles.json - Aligned title metadata and narrative tags
"""

import json
import os
import time
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from extract_vibe_tags import tag_text

def main():
    print("=" * 60)
    print("UniversalRatingEngine • Semantic Index Generator")
    print("=" * 60)

    # 1. Load Unified Plots
    plots_df = pl.read_parquet("data/unified_plots.parquet")
    print(f"Loaded {len(plots_df)} plot records.")

    # 2. Filter & Sort Top Titles to Embed
    # Include all titles with votes >= 500, or top calibrated ranks
    # Ensure representative coverage across Cinema, Anime, and TV
    filtered = plots_df.filter(
        (pl.col("votes") >= 300) | (pl.col("cal_rank") <= 5000)
    ).sort("cal_rank")

    # Take top 8,000 titles
    target_df = filtered.head(8000)
    print(f"Selected {len(target_df)} titles for the in-browser semantic index.")

    # 3. Extract Narrative Tags and Prepare Documents
    print("Extracting narrative tags and preparing documents...")
    titles_meta = []
    doc_texts = []

    for r in target_df.iter_rows(named=True):
        full_text = f"{r['overview']} {r['tagline']} {r['keywords']}"
        tags = tag_text(full_text, r["genres"])
        if not tags:
            # Fallback genre-based tag
            g_first = r["genres"].split("|")[0].replace("Anime", "").strip() or "Drama"
            tags = [g_first]

        titles_meta.append({
            "id": r["id"],
            "title": r["title"],
            "year": r["year"],
            "media_type": r["media_type"],
            "genres": r["genres"],
            "mod_b": round(float(r["mod_b"]), 2),
            "cal_rank": int(r["cal_rank"]),
            "tags": tags[:3]
        })

        tag_str = ", ".join(tags)
        kw_str = f" Keywords: {r['keywords']}." if r.get("keywords") else ""
        tl_str = f" Tagline: {r['tagline']}." if r.get("tagline") else ""
        
        # Synthesize rich document
        doc = f"{r['title']} ({r['year']}). {r['media_type']}. Genres: {r['genres']}. Themes: {tag_str}.{tl_str}{kw_str} Plot: {r['overview'][:450]}"
        doc_texts.append(doc)

    # 4. Generate Embeddings using all-MiniLM-L6-v2
    print(f"\nLoading transformer model (sentence-transformers/all-MiniLM-L6-v2)...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using compute device: {device}")

    batch_size = 128
    all_embeddings = []
    t0 = time.time()

    print(f"Embedding {len(doc_texts)} documents in batches of {batch_size}...")
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            norm_emb = F.normalize(sum_emb / sum_mask, p=2, dim=1)
            all_embeddings.append(norm_emb.cpu().numpy())

        done = min(i + batch_size, len(doc_texts))
        rate = done / (time.time() - t0)
        print(f"  [{done}/{len(doc_texts)}] ({done/len(doc_texts)*100:.1f}%) - {rate:.1f} docs/sec", end="\r")

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    print(f"\nGenerated matrix shape: {embeddings_matrix.shape} in {time.time()-t0:.2f}s")

    # 5. Quantize to Int8 (-128 to 127) for compact browser delivery
    # Unit vectors have components in [-1.0, 1.0]. Scaling by 127 preserves high fidelity
    int8_matrix = np.clip(embeddings_matrix * 127.0, -128, 127).astype(np.int8)

    # 6. Save Artifacts to docs/data/
    bin_path = "docs/data/vibe_embeddings.bin"
    int8_matrix.tofile(bin_path)
    print(f"Saved quantized vector binary to {bin_path} ({len(int8_matrix) * 384 / 1e6:.2f} MB)")

    json_path = "docs/data/vibe_titles.json"
    with open(json_path, "w") as f:
        json.dump(titles_meta, f, separators=(',', ':'))
    print(f"Saved metadata index to {json_path} ({os.path.getsize(json_path) / 1e6:.2f} MB)")

    # 7. Also enrich movies_combined.json with the tags dictionary so tags render everywhere
    print("\nEnriching docs/data/movies_combined.json with narrative tags...")
    id_to_tags = {m["id"]: m["tags"] for m in titles_meta}
    
    with open("docs/data/movies_combined.json") as f:
        comb = json.load(f)
    
    cols = comb["cols"]
    idx_id = cols.index("id")
    
    if "tags" not in cols:
        cols.append("tags")
        comb["cols"] = cols
        for row in comb["data"]:
            uid = row[idx_id]
            row.append(id_to_tags.get(uid, []))
    else:
        idx_tags = cols.index("tags")
        for row in comb["data"]:
            uid = row[idx_id]
            if uid in id_to_tags:
                row[idx_tags] = id_to_tags[uid]

    with open("docs/data/movies_combined.json", "w") as f:
        json.dump(comb, f, separators=(',', ':'))
    print("Updated docs/data/movies_combined.json successfully.")

    print("\n[SUCCESS] Semantic Vibe Index successfully generated!")

if __name__ == "__main__":
    main()
