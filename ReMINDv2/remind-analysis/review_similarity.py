# review_similarity.py: merge review JSONL with cosine similarity CSV
# Output: reports/review_similarity_XXXX.csv
# Usage:
#   python review_similarity.py --review remind_review_XXXX.jsonl --similarity similarity_XXXX.csv
# Notes:
# - review JSONL (v2): has "target" and "text"
# - legacy review JSONL: may have "rewakeout" and no "target"
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_in_default_dir(p: str, default_dir: str) -> Path:
    """
    If p exists as-is -> use it.
    Else, try default_dir/p (e.g., logs/foo.jsonl or reports/bar.csv).
    """
    path = Path(p).expanduser()
    if path.exists():
        return path

    alt = Path(default_dir) / path
    if alt.exists():
        return alt

    # If user passed bare filename but it's not in default_dir either, raise original for clarity
    raise FileNotFoundError(str(path))


def infer_suffix_from_review_path(review_path: Path) -> str:
    """
    logs/remind_review_XXXX.jsonl -> XXXX
    logs/remind_review_XXXX_rewake.jsonl -> XXXX_rewake
    """
    stem = review_path.stem  # remind_review_XXXX...
    if stem.startswith("remind_review_"):
        return stem.replace("remind_review_", "", 1)
    return stem


# -------------------------
# Review parsing
# -------------------------
def pick_reviewer(block: Dict[str, Any], prefer: str = "openai") -> Optional[Dict[str, Any]]:
    if not isinstance(block, dict):
        return None
    if prefer in block and isinstance(block[prefer], dict):
        return block[prefer]
    for v in block.values():
        if isinstance(v, dict):
            return v
    return None


def extract_target_and_text(r: Dict[str, Any]) -> tuple[str, str]:
    """
    v2: target + text
    legacy: rewakeout (no target)
    """
    tgt = r.get("target")
    txt = r.get("text")

    if isinstance(tgt, str) and isinstance(txt, str):
        return tgt, txt

    # legacy
    rw = r.get("rewakeout")
    if isinstance(rw, str):
        return "rewake", rw

    # last resort (avoid crash)
    return "unknown", ""


def truncate(s: Any, n: int) -> Any:
    if not isinstance(s, str):
        return s
    s2 = s.strip()
    if len(s2) <= n:
        return s2
    return s2[:n] + "…"


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Merge external review JSONL with similarity CSV.")

    # requested UX
    ap.add_argument("--review", required=True, type=str, help="review JSONL (default dir: logs/)")
    ap.add_argument("--similarity", required=True, type=str, help="similarity CSV (default dir: reports/)")

    # backward-compatible aliases (optional, safe)
    ap.add_argument("--review-jsonl", dest="review_jsonl", default="", type=str, help=argparse.SUPPRESS)
    ap.add_argument("--similarity_csv", dest="similarity_csv", default="", type=str, help=argparse.SUPPRESS)

    ap.add_argument("--out", default="", type=str, help="output CSV path (default: reports/review_similarity_XXXX.csv)")
    ap.add_argument("--reviewer", default="openai", type=str, help="preferred reviewer key (default: openai)")
    ap.add_argument(
        "--keep-long-text",
        action="store_true",
        help="keep full prompt/text (default: truncate for CSV readability)",
    )
    args = ap.parse_args()

    # allow old flags if user still uses them
    review_arg = args.review_jsonl or args.review
    sim_arg = args.similarity_csv or args.similarity

    review_path = resolve_in_default_dir(review_arg, "logs")
    similarity_path = resolve_in_default_dir(sim_arg, "reports")

    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        suffix = infer_suffix_from_review_path(review_path)
        out_path = Path("reports") / f"review_similarity_{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[merge] review    : {review_path}")
    print(f"[merge] similarity: {similarity_path}")
    print(f"[merge] out      : {out_path}")
    print(f"[merge] reviewer : {args.reviewer}")

    # load similarity
    sim_df = pd.read_csv(similarity_path)
    if "run_id" not in sim_df.columns:
        raise ValueError("similarity CSV must contain 'run_id' column")
    if "cosine_similarity" not in sim_df.columns:
        raise ValueError("similarity CSV must contain 'cosine_similarity' column")

    sim_df["run_id"] = pd.to_numeric(sim_df["run_id"], errors="coerce").astype("Int64")

    # load review
    rev_records = load_jsonl(review_path)
    print(f"[merge] loaded review records: {len(rev_records)}")

    rows: List[Dict[str, Any]] = []
    ok_review = 0
    missing_review = 0

    for r in rev_records:
        run_id = r.get("run_id")
        try:
            run_id_int = int(run_id)
        except Exception:
            continue

        status = (r.get("meta") or {}).get("status", "unknown")
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
        target, text = extract_target_and_text(r)

        if not args.keep_long_text:
            prompt = truncate(prompt, 500)
            text = truncate(text, 1500)
            short_rationale = truncate(short_rationale, 600)

        sum_score = None
        try:
            if alignment is not None and coherence is not None and novelty is not None:
                sum_score = int(alignment) + int(coherence) + int(novelty)
        except Exception:
            sum_score = None

        rows.append(
            {
                "run_id": run_id_int,
                "target": target,
                "review_status": status,
                "reviewer": args.reviewer,
                "review_model": model_name,
                "alignment": alignment,
                "coherence": coherence,
                "novelty": novelty,
                "sum_score": sum_score,
                "short_rationale": short_rationale,
                "prompt": prompt,
                "text": text,
            }
        )

    rev_df = pd.DataFrame(rows)
    if rev_df.empty:
        raise RuntimeError("No usable rows parsed from review JSONL (run_id missing?)")

    # merge
    merged = sim_df.merge(rev_df, on="run_id", how="left")

    print(f"[merge] similarity rows : {len(sim_df)}")
    print(f"[merge] review ok       : {ok_review}  missing/err: {missing_review}")
    print(f"[merge] merged         : {len(merged)}")

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[merge] wrote: {out_path}")

    scored = merged.dropna(subset=["alignment", "coherence", "novelty", "sum_score", "cosine_similarity"])
    print("\n[merge] summary (rows with both cosine + review scores)")
    print(f"  n = {len(scored)}")
    if len(scored) > 0:
        print(f"  sum_score mean={scored['sum_score'].mean():.2f}  min={scored['sum_score'].min():.0f}  max={scored['sum_score'].max():.0f}")
        print(f"  cosine    mean={scored['cosine_similarity'].mean():.3f}  min={scored['cosine_similarity'].min():.3f}  max={scored['cosine_similarity'].max():.3f}")


if __name__ == "__main__":
    main()