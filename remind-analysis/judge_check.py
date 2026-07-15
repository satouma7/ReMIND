# judge_check.py
# Check judgedream score distribution and non-empty idea counts in remind_sweep JSONL.
#
# Usage:
#   python judge_check.py remind_sweep_XXXX.jsonl
#
# Notes:
# - Default folder is logs/ (same convention as other scripts)
# - Output is printed to terminal only.

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_input_path(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.exists():
        return p
    alt = Path("logs") / p.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Input JSONL not found: {p} (also tried {alt})")


def is_nonempty_text(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def safe_get(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize judgedream score distribution and non-empty idea counts from remind_sweep JSONL."
    )
    ap.add_argument(
        "input",
        help="remind_sweep_XXXX.jsonl (default folder: logs/)",
    )
    ap.add_argument(
        "--topk-reasons",
        type=int,
        default=10,
        help="show top-K rewake_skipped_reason (default: 10)",
    )
    args = ap.parse_args()

    in_path = resolve_input_path(args.input)

    n_total = 0
    n_ok = 0
    n_not_ok = 0

    # judgedream
    score_counter = Counter()
    n_judgedream_missing = 0
    n_score_missing = 0

    # idea fields
    n_judgedream_idea_nonempty = 0
    n_idea_dream_nonempty = 0  # sometimes you may want this too

    # rewake status
    n_rewake_null = 0
    skip_reason_counter = Counter()

    # sanity checks
    bad_json = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            status = safe_get(obj, "meta", "status")
            if status == "ok":
                n_ok += 1
            else:
                n_not_ok += 1

            # Prefer reading from obj["result"], but tolerate missing
            res = obj.get("result", obj)

            jd = safe_get(res, "judgedream")
            if not isinstance(jd, dict):
                n_judgedream_missing += 1
            else:
                score = jd.get("score")
                if score is None:
                    n_score_missing += 1
                else:
                    # normalize to int bucket when possible
                    try:
                        score_int = int(score)
                        score_counter[score_int] += 1
                    except Exception:
                        score_counter[str(score)] += 1  # fallback bucket

                idea = jd.get("idea")
                if is_nonempty_text(idea):
                    n_judgedream_idea_nonempty += 1

            # also count idea_dream if present (often mirrors judgedream.idea logic)
            idea_dream = safe_get(res, "idea_dream")
            if is_nonempty_text(idea_dream):
                n_idea_dream_nonempty += 1

            # rewake tracking
            rewakeout = safe_get(res, "rewakeout")
            if rewakeout is None:
                n_rewake_null += 1
                reason = safe_get(res, "rewake_skipped_reason")
                if is_nonempty_text(reason):
                    skip_reason_counter[reason.strip()] += 1

    # ---- Print report ----
    print(f"[judge_check] input : {in_path}")
    print(f"[judge_check] total : {n_total}  (bad_json={bad_json})")
    print(f"[judge_check] status: ok={n_ok}  not_ok/other={n_not_ok}")
    print()

    print("[judgedream] score distribution")
    if sum(score_counter.values()) == 0:
        print("  (no judgedream.score found)")
    else:
        # show 1..5 explicitly when present
        for s in [1, 2, 3, 4, 5]:
            if s in score_counter:
                print(f"  score={s}: {score_counter[s]}")
        # show any other buckets
        others = {k: v for k, v in score_counter.items() if k not in {1, 2, 3, 4, 5}}
        for k, v in sorted(others.items(), key=lambda x: (-x[1], str(x[0]))):
            print(f"  score={k}: {v}")
    print(f"  judgedream missing: {n_judgedream_missing}")
    print(f"  score missing     : {n_score_missing}")
    print()

    print("[idea] non-empty counts")
    print(f"  judgedream.idea non-empty: {n_judgedream_idea_nonempty} / {n_total}")
    print(f"  idea_dream     non-empty: {n_idea_dream_nonempty} / {n_total}")
    print()

    print("[rewake]")
    print(f"  rewakeout is null: {n_rewake_null} / {n_total}")
    if len(skip_reason_counter) > 0:
        print(f"  top {args.topk_reasons} rewake_skipped_reason:")
        for reason, cnt in skip_reason_counter.most_common(args.topk_reasons):
            print(f"    {cnt:>5}  {reason}")
    else:
        print("  rewake_skipped_reason: (none found)")


if __name__ == "__main__":
    main()