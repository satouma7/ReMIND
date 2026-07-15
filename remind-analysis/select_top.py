#!/usr/bin/env python3
# select_top.py: Select top rewake candidates from review JSONL for evaluate.py
#
# Strategy (per theme):
#   1. novelty >= 4  -> if count >= min_count, use as-is
#   2. novelty >= 3  -> fallback if step 1 yields < min_count
#   3. all status=ok -> final fallback
#
# Usage:
#   python select_top.py logs/remind_review_1ooo_core_rewake_gpt5mini.jsonl
#   python select_top.py logs/remind_review_ogo_*_core_rewake_gpt5mini.jsonl --min-count 30
#   python select_top.py logs/remind_review_*_rewake_gpt5mini.jsonl --theme time_space --dry-run
#
# Output:
#   reports/<cond>/top_rewakeout_<cond>_<theme>.csv  (one file per theme)

from __future__ import annotations
import argparse
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

THEME_KEYWORDS = {
    "time_space":        "time",
    "aperiodic_craft":   "aperiodic",
    "periodic_tarot":    "tarot",
    # social pairs
    "loss_of_life_purpose": "loss of life",
    "urban_loneliness":     "urban",
    "knowledge_lifespan":   "lifespan",
}


def load_rewake_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("meta") or {}).get("status") != "ok":
                continue
            if str(r.get("target", "")).lower() != "rewake":
                continue
            blob = (r.get("reviews") or {}).get("openai") or {}
            novelty = blob.get("novelty")
            if novelty is None:
                continue
            text = (r.get("text") or r.get("rewakeout") or "").strip()
            if not text:
                continue
            sweep = r.get("sweep") or {}
            records.append({
                "run_id":     r.get("run_id"),
                "pair":       sweep.get("pair"),
                "temp_dream": sweep.get("temp_dream"),
                "alignment":  blob.get("alignment"),
                "coherence":  blob.get("coherence"),
                "novelty":    int(novelty),
                "text":       text,
            })
    return records


def select_for_theme(records: List[Dict], keyword: str, min_count: int) -> tuple[List[Dict], int]:
    """Returns (selected_records, threshold_used)."""
    subset = [r for r in records if keyword in str(r["pair"]).lower()]
    for threshold in [4, 3, 0]:
        chosen = [r for r in subset if r["novelty"] >= threshold] if threshold > 0 else subset
        chosen_sorted = sorted(chosen, key=lambda r: -r["novelty"])
        if len(chosen_sorted) >= min_count or threshold == 0:
            return chosen_sorted, threshold
    return subset, 0


def infer_cond(path: Path) -> str:
    name = path.name
    name = name.replace("remind_review_", "").replace("_rewake_gpt5mini.jsonl", "")
    # strip timestamp: ogo_20260420T013428Z_core -> ogo_core
    import re
    name = re.sub(r"_\d{8}T\d{6}Z", "", name)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="Select top rewake candidates for evaluate.py")
    ap.add_argument("jsonl", nargs="+", help="remind_review_*_rewake_gpt5mini.jsonl file(s)")
    ap.add_argument("--min-count", type=int, default=30, help="minimum records per theme (default: 30)")
    ap.add_argument("--theme", type=str, default="", help="limit to specific theme (e.g. time_space)")
    ap.add_argument("--reports", type=str, default="reports", help="output base dir (default: reports/)")
    ap.add_argument("--dry-run", action="store_true", help="show counts only, no file output")
    args = ap.parse_args()

    # expand globs
    paths: List[Path] = []
    for pat in args.jsonl:
        expanded = glob.glob(pat)
        paths.extend(Path(p) for p in (expanded if expanded else [pat]))

    themes = {k: v for k, v in THEME_KEYWORDS.items()
              if not args.theme or k == args.theme}

    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[warn] not found: {path}")
            continue

        cond = infer_cond(path)
        records = load_rewake_records(path)

        print(f"\n=== {cond} (total ok+rewake: {len(records)}) ===")

        for theme_label, keyword in themes.items():
            chosen, threshold = select_for_theme(records, keyword, args.min_count)
            thr_str = f">={threshold}" if threshold > 0 else "all(ok)"
            print(f"  {theme_label}: {len(chosen)}件  threshold={thr_str}")

            if args.dry_run or not chosen:
                continue

            # output dir: reports/<cond>/
            out_dir = Path(args.reports) / cond
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"top_rewakeout_{cond}_{theme_label}.csv"

            df = pd.DataFrame(chosen)[["run_id", "alignment", "coherence", "novelty", "temp_dream", "text"]]
            df.insert(1, "cond", cond)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"    -> {out_path}")


if __name__ == "__main__":
    main()
