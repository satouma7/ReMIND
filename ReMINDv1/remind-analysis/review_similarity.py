# review_similarity.py: merge idea-level cosine similarity CSV with OpenAI review JSONL
# Output: idea_similarity_with_review.csv (adds prompt, rewakeout, alignment/coherence/novelty/sum_score)
# Usage:
#   python review_similarity.py \
#     --idea-csv reports/idea_similarity.csv \
#     --review-jsonl logs/remind_review_20251227T072006Z.jsonl \
#     --out reports/idea_similarity_with_review.csv
# Notes:
# - Assumes idea_similarity.csv has at least: run_id, cosine_similarity
# - idea_similarity.csv is typically produced by similarity2.py and therefore includes ONLY runs where
#   both idea_wake and idea_dream are non-empty (runs with empty extracted ideas are excluded upstream).
# - This script merges by run_id, effectively keeping the intersection of:
#     {runs present in idea_similarity.csv} ∩ {runs present in review JSONL with "openai" scores}
#   As a result, reviewed runs that lack cosine_similarity (i.e., absent from idea_similarity.csv)
#   will not appear in the output.
# - review jsonl lines contain: run_id, prompt, rewakeout, reviews.openai.{alignment,coherence,novelty}
# - If multiple reviewers exist, this script prioritizes "openai" by default.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def pick_reviewer(block: Dict[str, Any], prefer: str = "openai") -> Optional[Dict[str, Any]]:
    if not isinstance(block, dict):
        return None
    if prefer in block and isinstance(block[prefer], dict):
        return block[prefer]
    for v in block.values():
        if isinstance(v, dict):
            return v
    return None

def main() -> None:
    ap = argparse.ArgumentParser(description="Merge idea similarity CSV with external review JSONL.")
    ap.add_argument("--idea-csv", required=True, type=str, help="path to idea_similarity.csv")
    ap.add_argument("--review-jsonl", required=True, type=str, help="path to remind_review_*.jsonl")
    ap.add_argument("--out", default="reports/idea_similarity_with_review.csv", type=str, help="output CSV path")
    ap.add_argument("--reviewer", default="openai", type=str, help="preferred reviewer key (default: openai)")
    ap.add_argument(
        "--keep-long-text",
        action="store_true",
        help="keep full prompt/rewakeout text (default: truncate to 500/1500 chars for CSV readability)",
    )
    args = ap.parse_args()

    idea_path = Path(args.idea_csv).expanduser()
    review_path = Path(args.review_jsonl).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not idea_path.exists():
        raise FileNotFoundError(f"idea CSV not found: {idea_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"review JSONL not found: {review_path}")

    print(f"[merge] loading idea csv   : {idea_path}")
    idea_df = pd.read_csv(idea_path)
    if "run_id" not in idea_df.columns:
        raise ValueError("idea CSV must contain 'run_id' column")
    if "cosine_similarity" not in idea_df.columns:
        raise ValueError("idea CSV must contain 'cosine_similarity' column")

    # normalize run_id type for merge
    idea_df["run_id"] = pd.to_numeric(idea_df["run_id"], errors="coerce").astype("Int64")

    print(f"[merge] loading review jsonl: {review_path}")
    rev_records = load_jsonl(review_path)
    print(f"[merge] loaded review records: {len(rev_records)}")

    # Build lookup by run_id
    rows: List[Dict[str, Any]] = []
    missing_review = 0
    ok_review = 0

    for r in rev_records:
        run_id = r.get("run_id")
        try:
            run_id_int = int(run_id)
        except Exception:
            continue

        if r.get("meta", {}).get("status") != "ok":
            # keep it (so we can see missing), but scores will be NaN
            status = r.get("meta", {}).get("status", "unknown")
        else:
            status = "ok"

        reviewer_obj = pick_reviewer((r.get("reviews") or {}), prefer=args.reviewer)

        alignment = coherence = novelty = None
        model_name = None
        short_rationale = None
        if reviewer_obj:
            model_name = reviewer_obj.get("model")
            alignment = reviewer_obj.get("alignment")
            coherence = reviewer_obj.get("coherence")
            novelty = reviewer_obj.get("novelty")
            short_rationale = reviewer_obj.get("short_rationale")

        if status == "ok" and reviewer_obj:
            ok_review += 1
        else:
            missing_review += 1

        prompt = r.get("prompt", "")
        rewakeout = r.get("rewakeout", "")

        if not args.keep_long_text:
            # keep CSV readable (Excel-friendly)
            if isinstance(prompt, str) and len(prompt) > 500:
                prompt = prompt[:500] + "…"
            if isinstance(rewakeout, str) and len(rewakeout) > 1500:
                rewakeout = rewakeout[:1500] + "…"

        sum_score = None
        try:
            if alignment is not None and coherence is not None and novelty is not None:
                sum_score = int(alignment) + int(coherence) + int(novelty)
        except Exception:
            sum_score = None

        rows.append(
            {
                "run_id": run_id_int,
                "review_status": status,
                "reviewer": args.reviewer,
                "review_model": model_name,
                "alignment": alignment,
                "coherence": coherence,
                "novelty": novelty,
                "sum_score": sum_score,
                "short_rationale": short_rationale,
                "prompt": prompt,
                "rewakeout": rewakeout,
            }
        )

    rev_df = pd.DataFrame(rows)
    if rev_df.empty:
        raise RuntimeError("No usable rows parsed from review JSONL (run_id missing?)")

    # Merge
    merged = idea_df.merge(rev_df, on="run_id", how="left")

    # Report
    print(f"[merge] idea rows : {len(idea_df)}")
    print(f"[merge] review ok : {ok_review}  missing/err: {missing_review}")
    print(f"[merge] merged    : {len(merged)}")

    # Save
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[merge] wrote: {out_path}")

    # Quick summaries
    scored = merged.dropna(subset=["alignment", "coherence", "novelty", "sum_score", "cosine_similarity"])
    print("\n[merge] summary (rows with both cosine + review scores)")
    print(f"  n = {len(scored)}")
    if len(scored) > 0:
        print(f"  sum_score mean={scored['sum_score'].mean():.2f}  min={scored['sum_score'].min():.0f}  max={scored['sum_score'].max():.0f}")
        print(f"  cosine    mean={scored['cosine_similarity'].mean():.3f}  min={scored['cosine_similarity'].min():.3f}  max={scored['cosine_similarity'].max():.3f}")

        # Example: distribution for high scores
        hi = scored[scored["sum_score"] >= 14]
        if len(hi) > 0:
            print(f"\n  sum_score>=14: n={len(hi)}  cosine mean={hi['cosine_similarity'].mean():.3f}  min={hi['cosine_similarity'].min():.3f}  max={hi['cosine_similarity'].max():.3f}")


if __name__ == "__main__":
    main()