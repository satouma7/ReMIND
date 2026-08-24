#!/usr/bin/env python3
# extract_low_judgedream.py
# Extract dream outputs whose judgedream.score is in {1,2,3} from ReMIND sweep logs.
# Usage example:
#   python extract_low_judgedream.py logs/remind_sweep_1ooo_maxt_core.jsonl

import argparse
import json
from pathlib import Path

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_jsonl", help="Input sweep jsonl, e.g., logs/remind_sweep_ooo_..._core.jsonl")
    ap.add_argument("--out", default=None, help="Output jsonl path (default: <input>.judgedream_1-3.jsonl)")
    ap.add_argument("--scores", default="1,2,3", help="Comma-separated judgedream scores to extract (default: 1,2,3)")
    ap.add_argument("--limit", type=int, default=0, help="Max number of records to write (0 = no limit)")
    ap.add_argument("--with-wake", action="store_true", help="Also include wakeout/judgewake in output")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    scores = [int(s.strip()) for s in args.scores.split(",") if s.strip()]
    scores_sorted = sorted(scores)

    if scores_sorted == list(range(scores_sorted[0], scores_sorted[-1] + 1)):
        score_str = f"{scores_sorted[0]}-{scores_sorted[-1]}"
    else:
        score_str = "-".join(str(s) for s in scores_sorted)

    out_path = (
        Path(args.out)
        if args.out
        else in_path.parent / f"{in_path.stem}_judgedream_{score_str}.jsonl"
    )

    n_total = 0
    n_written = 0
    n_missing = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1

            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_missing += 1
                continue

            res = r.get("result") or {}
            judgedream = res.get("judgedream") or {}
            score = judgedream.get("score", None)

            if score is None:
                continue
            try:
                score_int = int(score)
            except Exception:
                continue

            if score_int not in scores:
                continue

            dreamout = res.get("dreamout")
            if not isinstance(dreamout, str) or not dreamout.strip():
                # Sometimes missing/empty: still write metadata, but mark it.
                dreamout = ""

            rec = {
                "run_id": r.get("run_id"),
                "llm_code": safe_get(r, "meta", "llm_code"),
                "ts_utc": safe_get(r, "meta", "ts_utc"),
                "status": safe_get(r, "meta", "status"),
                "pair": res.get("pair"),
                "template_id": res.get("template_id"),
                "word_limit": res.get("word_limit"),
                "prompt": res.get("prompt"),
                "params": res.get("params"),
                "judgedream": judgedream,
                "dreamout": dreamout,
                "rewake_skipped_reason": res.get("rewake_skipped_reason"),
            }

            if args.with_wake:
                rec["wakeout"] = res.get("wakeout")
                rec["judgewake"] = res.get("judgewake")

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

            if args.limit and n_written >= args.limit:
                break

    print(f"[done] input={in_path}")
    print(f"[done] output={out_path}")
    print(f"[stats] total_lines={n_total} written={n_written} bad_json={n_missing}")
    print(f"[stats] extracted_scores={sorted(scores)}")

if __name__ == "__main__":
    main()