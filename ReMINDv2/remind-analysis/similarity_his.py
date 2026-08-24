# similarity_his.py: cosine similarity histogram overlay for ReMIND (wake-level)
# - wake-dream similarity using wakeout vs dreamout (run-level)
# - wake-wake similarity using wakeout only (negative control; within same condition group)
# - automatic output naming from remind_sweep_xxx.jsonl -> similarity_wake_xxx.csv + similarity_his_xxx.png
# Usage:
#   python similarity_his.py logs/remind_sweep_XXXX.jsonl
# Output:  similarity_wake_XXXX.csv, similarity_his_XXXX.png
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_sweep_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Assumes sweep.py-style JSONL. Returns empty/default fields if missing."""
    sweep = r.get("sweep", {}) or {}
    pair = sweep.get("pair")
    if isinstance(pair, list):
        pair = tuple(pair)
    return {
        "run_id": r.get("run_id"),
        "pair": pair,
        "template_id": sweep.get("template_id"),
        "word_limit": sweep.get("word_limit"),
        "temp_dream": sweep.get("temp_dream"),
        "seed_dream": sweep.get("seed_dream"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Histogram overlay of wake–dream vs wake–wake cosine similarity (ReMIND).")
    ap.add_argument("jsonl", type=str, help="path to remind_sweep_*.jsonl")
    ap.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformer model name",
    )
    ap.add_argument("--out-csv", type=str, default="", help="output CSV (default: reports/similarity_wake_<suffix>.csv)")
    ap.add_argument("--plot-out", type=str, default="", help="output PNG (default: reports/similarity_his_<suffix>.png)")
    ap.add_argument("--bins", type=int, default=30, help="histogram bins")
    ap.add_argument("--min-temp-dream", type=float, default=None, metavar="T",
                    help="skip records with temp_dream < T (e.g. 1.0 to exclude control runs)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).expanduser()

    # auto naming: remind_sweep_xxx.jsonl -> similarity_wake_xxx.csv / similarity_his_xxx.png
    stem = jsonl_path.stem
    suffix = stem.replace("remind_sweep_", "")

    out_csv = Path(args.out_csv).expanduser() if args.out_csv else Path("reports") / f"similarity_wake_{suffix}.csv"
    plot_path = Path(args.plot_out).expanduser() if args.plot_out else Path("reports") / f"similarity_his_{suffix}.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[similarity_his] loading: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"[similarity_his] loaded {len(records)} records")

    # ---- extract run-level wake–dream rows ----
    rows: List[Dict[str, Any]] = []
    for r in records:
        res = r.get("result") or r

        wakeout = (res.get("wakeout") or "").strip()
        dreamout = (res.get("dreamout") or "").strip()
        if not wakeout or not dreamout:
            continue

        meta = get_sweep_fields(r)
        if args.min_temp_dream is not None and (meta["temp_dream"] is None or float(meta["temp_dream"]) < args.min_temp_dream):
            continue
        rows.append(
            {
                **meta,
                "wake_out": wakeout,
                "dream_out": dreamout,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[similarity_his] valid wake–dream pairs: {len(df)}")
    if len(df) == 0:
        print("[similarity_his] no valid rows. exit.")
        return

    # ---- embedding ----
    print(f"[similarity_his] loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    emb_wake = model.encode(
        df["wake_out"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb_dream = model.encode(
        df["dream_out"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    sims_wd = util.cos_sim(emb_wake, emb_dream).diagonal().cpu().numpy().astype(np.float32)
    df["wake_dream_similarity"] = sims_wd

    # ---- build wake–wake negative control distribution ----
    # Group by (pair, template_id, word_limit): this defines the "same task prompt family".
    wake_rows: List[Tuple[Tuple[Any, Any, Any], str]] = []
    for r in records:
        res = r.get("result") or r
        wakeout = (res.get("wakeout") or "").strip()
        if not wakeout:
            continue

        meta = get_sweep_fields(r)
        if args.min_temp_dream is not None and (meta["temp_dream"] is None or float(meta["temp_dream"]) < args.min_temp_dream):
            continue
        key = (meta["pair"], meta["template_id"], meta["word_limit"])
        if key[0] is None or key[1] is None or key[2] is None:
            continue
        wake_rows.append((key, wakeout))

    ww_vals_all: List[float] = []
    if wake_rows:
        # collect per-group wakeouts
        group_map: Dict[Tuple[Any, Any, Any], List[str]] = {}
        for key, w in wake_rows:
            group_map.setdefault(key, []).append(w)

        for key, ws in group_map.items():
            uniq = list(pd.unique(ws))
            if len(uniq) == 1:
                # All identical -> similarity defined as 1
                ww_vals_all.append(1.0)
                continue

            # Variability exists: compute pairwise cosine similarities among unique wakeouts
            emb = model.encode(uniq, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
            m = util.cos_sim(emb, emb).cpu().numpy().astype(float)
            iu = np.triu_indices_from(m, k=1)
            vals = m[iu]
            # vals may be empty in weird edge cases, but normally not.
            if vals.size == 0:
                ww_vals_all.append(1.0)
            else:
                ww_vals_all.extend(vals.tolist())

    ww_vals = np.array(ww_vals_all, dtype=float)
    if ww_vals.size > 0:
        print("\n[similarity_his] wake–wake summary (negative control)")
        print(f"  n_vals = {ww_vals.size}")
        print(f"  mean  = {float(np.mean(ww_vals)):.3f}")
        print(f"  std   = {float(np.std(ww_vals)):.3f}")
        print(f"  min   = {float(np.min(ww_vals)):.3f}")
        print(f"  max   = {float(np.max(ww_vals)):.3f}")
    else:
        print("\n[similarity_his] wake–wake: no values (insufficient wakeouts).")

    # ---- save wake-wake similarity values ----
    ww_df = pd.DataFrame({
        "wake_wake_similarity": ww_vals
    })
    ww_df.to_csv(out_csv, index=False)

    print(f"[similarity_his] wrote wake-wake similarity CSV: {out_csv}")

    # ---- histogram overlay ----
    plt.figure()
    plt.hist(df["wake_dream_similarity"].values, bins=args.bins, density=True, alpha=0.6, label="Wake–Dream (wakeout vs dreamout)")
    if ww_vals.size > 0:
        ww_counts, ww_edges = np.histogram(ww_vals, bins=args.bins, density=True)
        ww_counts = ww_counts / 5
        plt.stairs(ww_counts, ww_edges, fill=True, alpha=0.6, label="Wake–Wake (neg ctrl, density×0.2)")
        plt.xlim(0.3, 1.1) 

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"\n[similarity_his] wrote histogram: {plot_path}")

if __name__ == "__main__":
    main()