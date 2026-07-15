# similarity.py: cosine similarity analysis for ReMIND
# This script evaluates semantic cosine between idea_wake and idea_dream outputs. 
# Embeddings are normalized before similarity computation to improve numerical stability. 
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def main() -> None:
    ap = argparse.ArgumentParser(description="Cosine similarity analysis for ReMIND ideas (util.cos_sim)")
    ap.add_argument("jsonl", type=str, help="path to remind_sweep_*.jsonl")
    ap.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformer model name",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="reports/idea_similarity.csv",
        help="output CSV path",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[similarity] loading: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"[similarity] loaded {len(records)} records")

    # ---- extract valid pairs ----
    rows = []
    for r in records:
        res = r.get("result", r)  # Use result for sweep runs; for a single run, use r itself

        idea_wake = (res.get("idea_wake") or "").strip()
        idea_dream = (res.get("idea_dream") or "").strip()

        if not idea_wake or not idea_dream:
            continue  # skip empty

        sweep = r.get("sweep", {})

        rows.append(
            {
                "run_id": r.get("run_id"),
                "pair": tuple(sweep.get("pair", [])),
                "template_id": sweep.get("template_id"),
                "word_limit": sweep.get("word_limit"),
                "temp_dream": sweep.get("temp_dream"),
                "seed_dream": sweep.get("seed_dream"),
                "idea_wake": idea_wake,
                "idea_dream": idea_dream,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[similarity] valid idea pairs: {len(df)}")

    if len(df) == 0:
        print("[similarity] no valid rows found. exiting.")
        return

    # ---- embedding ----
    print(f"[similarity] loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # util.cos_sim expects torch tensors; therefore we set convert_to_tensor=True
    emb_wake = model.encode(
        df["idea_wake"].tolist(),
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,   # improves numerical stability for cosine similarity
    )
    emb_dream = model.encode(
        df["idea_dream"].tolist(),
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # ---- cosine similarity (pairwise diagonal) ----
    sims = util.cos_sim(emb_wake, emb_dream).diagonal()

    # move to CPU and cast to float before storing in pandas
    df["cosine_similarity"] = sims.cpu().numpy().astype(np.float32)

    # ---- save ----
    df.to_csv(out_path, index=False)
    print(f"[similarity] wrote CSV: {out_path}")

    # ---- quick summary ----
    print("\n[similarity] summary")
    print(f"  mean = {df['cosine_similarity'].mean():.3f}")
    print(f"  std  = {df['cosine_similarity'].std():.3f}")
    print(f"  min  = {df['cosine_similarity'].min():.3f}")
    print(f"  max  = {df['cosine_similarity'].max():.3f}")


if __name__ == "__main__":
    main()