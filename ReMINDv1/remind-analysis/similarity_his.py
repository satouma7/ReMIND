# similarity_his.py: cosine similarity analysis for ReMIND (idea-level)
# - wake-dream similarity (idea_wake vs idea_dream)
# - wake-wake similarity using wake_out (negative control; within same condition group)
# - histogram overlay
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
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
    ap = argparse.ArgumentParser(description="Similarity analysis for ReMIND (offline).")
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
        default="reports/idea_similarity_his.csv",
        help="run-level output CSV (wake-dream)",
    )
    ap.add_argument(
        "--wake-wake-out",
        type=str,
        default="reports/wake_wake_similarity.csv",
        help="wake-wake output CSV (negative control)",
    )
    ap.add_argument(
        "--tail-thr",
        type=float,
        default=0.3,
        help="threshold for tail extraction (cos_sim < tail-thr)",
    )
    ap.add_argument(
        "--tail-out",
        type=str,
        default="reports/tail_ids.csv",
        help="tail output CSV path",
    )
    ap.add_argument(
        "--plot-out",
        type=str,
        default="reports/cosine_hist_overlay.png",
        help="histogram overlay PNG path",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=30,
        help="histogram bins",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl).expanduser()
    out_path = Path(args.out).expanduser()
    ww_path = Path(args.wake_wake_out).expanduser()
    tail_path = Path(args.tail_out).expanduser()
    plot_path = Path(args.plot_out).expanduser()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ww_path.parent.mkdir(parents=True, exist_ok=True)
    tail_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[similarity_his] loading: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"[similarity_his] loaded {len(records)} records")

    # ---- extract wake-dream rows ----
    rows: List[Dict[str, Any]] = []
    for r in records:
        res = r.get("result", r)  # Use result for sweep runs; for a single run, use r itself

        idea_wake = (res.get("idea_wake") or "").strip()
        idea_dream = (res.get("idea_dream") or "").strip()
        if not idea_wake or not idea_dream:
            continue

        meta = get_sweep_fields(r)
        rows.append(
            {
                **meta,
                "idea_wake": idea_wake,
                "idea_dream": idea_dream,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[similarity_his] valid idea pairs (wake-dream): {len(df)}")
    if len(df) == 0:
        print("[similarity_his] no valid rows. exit.")
        return

    # ---- embedding ----
    print(f"[similarity_his] loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    emb_wake = model.encode(df["idea_wake"].tolist(), convert_to_tensor=True, show_progress_bar=True)
    emb_dream = model.encode(df["idea_dream"].tolist(), convert_to_tensor=True, show_progress_bar=True)

    # util.cos_sim returns matrix; take diagonal
    sim_mat = util.cos_sim(emb_wake, emb_dream)
    sims = sim_mat.diag().cpu().numpy().astype(float)

    df["cosine_similarity"] = sims

    # ---- save run-level ----
    df.to_csv(out_path, index=False)
    print(f"[similarity_his] wrote run-level CSV: {out_path}")

    print("\n[similarity_his] run-level summary (wake-dream)")
    print(f"  mean = {df['cosine_similarity'].mean():.3f}")
    print(f"  std  = {df['cosine_similarity'].std():.3f}")
    print(f"  min  = {df['cosine_similarity'].min():.3f}")
    print(f"  max  = {df['cosine_similarity'].max():.3f}")

    # ---- tail extraction ----
    thr = float(args.tail_thr)
    tail_df = df[df["cosine_similarity"] < thr].copy().sort_values("cosine_similarity", ascending=True)

    # Save only (run_id + conditions + cosine_similarity) so we can fetch the full text later from the JSONL.
    tail_cols = ["run_id", "pair", "template_id", "word_limit", "temp_dream", "seed_dream", "cosine_similarity"]
    tail_df_out = tail_df[tail_cols]
    tail_df_out.to_csv(tail_path, index=False)
    tail_rate = (len(tail_df_out) / len(df)) * 100.0

    print("\n[similarity_his] tail extraction")
    print(f"  threshold = {thr:.3f}")
    print(f"  tail n    = {len(tail_df_out)} / {len(df)} ({tail_rate:.2f}%)")
    print(f"  wrote CSV = {tail_path}")

    # Wake–Wake similarity (negative control):
    # Group by (pair, template_id, word_limit), then compare unique wakeout texts within each group.
    # (wakeout is longer than ideas, but we use it here as the requested negative-control signal.)
    wake_rows: List[Dict[str, Any]] = []
    for r in records:
        res = r.get("result", r)
        wakeout = (res.get("wakeout") or "").strip()
        if not wakeout:
            continue

        meta = get_sweep_fields(r)
        # only groups that have enough info
        if meta["pair"] is None or meta["template_id"] is None or meta["word_limit"] is None:
            continue

        wake_rows.append(
            {
                **meta,
                "wakeout": wakeout,
            }
        )

    wdf = pd.DataFrame(wake_rows)
    # Some runs may be missing; still okay.
    if len(wdf) > 0:
        # For each group, get unique wakeouts; compute pairwise sims among them; summarize mean.
        ww_records: List[Dict[str, Any]] = []
        groups = wdf.groupby(["pair", "template_id", "word_limit"], dropna=False)

        for (pair, tid, wl), g in groups:
            uniq = g["wakeout"].dropna().unique().tolist()
            if len(uniq) < 2:
                continue

            emb = model.encode(uniq, convert_to_tensor=True, show_progress_bar=False)
            m = util.cos_sim(emb, emb).cpu().numpy().astype(float)

            # take upper triangle excluding diagonal
            iu = np.triu_indices_from(m, k=1)
            vals = m[iu]
            ww_records.append(
                {
                    "pair": pair,
                    "template_id": tid,
                    "word_limit": wl,
                    "n_unique_wake": len(uniq),
                    "wake_wake_mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "wake_wake_min": float(np.min(vals)) if len(vals) else np.nan,
                    "wake_wake_max": float(np.max(vals)) if len(vals) else np.nan,
                }
            )

        ww_df = pd.DataFrame(ww_records)
    else:
        ww_df = pd.DataFrame(columns=["pair", "template_id", "word_limit", "n_unique_wake", "wake_wake_mean"])

    ww_df.to_csv(ww_path, index=False)
    print(f"\n[similarity_his] wrote wake-wake CSV: {ww_path}")

    if len(ww_df) > 0:
        print("\n[similarity_his] wake-wake summary (negative control)")
        print(f"  groups total = {len(wdf.groupby(['pair','template_id','word_limit']))}")
        print(f"  groups with >=2 unique wakes = {len(ww_df)}")
        print(f"  mean = {ww_df['wake_wake_mean'].mean():.3f}")
        print(f"  std  = {ww_df['wake_wake_mean'].std():.3f}")
        print(f"  min  = {ww_df['wake_wake_mean'].min():.3f}")
        print(f"  max  = {ww_df['wake_wake_mean'].max():.3f}")

    # ---- histogram overlay ----
    plt.figure()
    plt.hist(df["cosine_similarity"].values, bins=args.bins, density=True, alpha=0.6, label="Wake–Dream")

    # wake-wake may have few groups; plot only if exists
    if len(ww_df) > 0 and ww_df["wake_wake_mean"].notna().any():
        # wake–wake (negative control): 1/10 density 
        ww_vals = ww_df["wake_wake_mean"].dropna().values
        ww_counts, ww_edges = np.histogram(
            ww_vals,
            bins=args.bins,
            density=True,
        )
        ww_counts = ww_counts / 10.0
        plt.stairs(
            ww_counts,
            ww_edges,
            fill=True,
            alpha=0.6,
            label="Wake–Wake (neg ctrl, density×0.1)",
        )

    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"\n[similarity_his] wrote histogram: {plot_path}")


if __name__ == "__main__":
    main()