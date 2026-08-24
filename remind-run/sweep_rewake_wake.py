# sweep_rewake_wake.py: augment an existing sweep log with REWAKE_WAKE
# (idea_wake -> rewakeout_wake), without re-running WAKE/DREAM/JUDGE.
# Only the llm_wake server needs to be running (dream/judge models are not touched).
# Reads an existing sweep_*.jsonl and writes a NEW file with rewakeout_wake /
# rewake_wake_skipped_reason added to each ok record's result. The input file
# is never modified.
# Usage:
#   python sweep_rewake_wake.py --in logs/remind_sweep_qoq_TIMESTAMP_core.jsonl
#   python sweep_rewake_wake.py --in logs/remind_sweep_qoq_TIMESTAMP_core.jsonl --out logs/custom_name.jsonl
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

from ensure_tmux import ensure_tmux
from wake import wake


def compute_rewake_wake(
    *, idea_wake: str, score_wake: int, word_limit: int, score_threshold: int,
    llm_wake: str, temp_rewake: float, seed_rewake: int, max_tokens_rewake: int,
) -> tuple[Optional[str], Optional[str]]:
    if score_wake >= score_threshold and idea_wake:
        prompt_idea_wake = (
            f"Propose the following idea to the user within {word_limit} words.\n"
            f"IDEA:\n{idea_wake}\n"
        )
        rewakeout_wake = wake(
            prompt_idea_wake,
            llm=llm_wake,
            max_tokens=max_tokens_rewake,
            temperature=temp_rewake,
            seed=seed_rewake,
        )
        return rewakeout_wake, None
    return None, f"score_wake={score_wake}, idea_wake_empty={not bool(idea_wake)}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Augment an existing sweep log with REWAKE_WAKE (idea_wake -> rewakeout_wake)."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Existing sweep_*.jsonl to read")
    ap.add_argument("--out", dest="out_path", default="", help="Output path (default: <in>_rewake_wake.jsonl)")
    ap.add_argument("--score-threshold", type=int, default=None,
                     help="Override score_threshold (default: use each record's own params.score_threshold)")
    ap.add_argument("--min-temp-dream", type=float, default=1.0,
                     help="Skip records with temp_dream below this (default: 1.0, excludes control runs, "
                          "matching the temp_dream>=1.0 filter used elsewhere in this repo)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path.with_name(
        in_path.stem + "_rewake_wake" + in_path.suffix
    )
    if out_path.resolve() == in_path.resolve():
        raise ValueError("--out must differ from --in (the input log is never modified)")

    records = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    required_llms = sorted({
        rec["result"]["params"]["llm_wake"]
        for rec in records
        if rec.get("result") and rec.get("meta", {}).get("status") == "ok"
    })
    if required_llms:
        ensure_tmux(required=required_llms, stop_unused=True)

    ok = skipped = failed = excluded = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in records:
            r = rec.get("result")
            if not r or rec.get("meta", {}).get("status") != "ok":
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            params = r["params"]
            temp_dream = rec.get("sweep", {}).get("temp_dream", params.get("temp_dream"))
            if temp_dream is not None and temp_dream < args.min_temp_dream:
                r["rewakeout_wake"] = None
                r["rewake_wake_skipped_reason"] = f"excluded: temp_dream={temp_dream} < {args.min_temp_dream} (control run)"
                excluded += 1
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            score_threshold = (
                args.score_threshold if args.score_threshold is not None else params["score_threshold"]
            )
            idea_wake = r.get("idea_wake") or ""
            score_wake = int((r.get("judgewake") or {}).get("score", 0) or 0)

            try:
                rewakeout_wake, skip_reason = compute_rewake_wake(
                    idea_wake=idea_wake,
                    score_wake=score_wake,
                    word_limit=r["word_limit"],
                    score_threshold=score_threshold,
                    llm_wake=params["llm_wake"],
                    temp_rewake=params["temp_rewake"],
                    seed_rewake=params["seed_rewake"],
                    max_tokens_rewake=params["max_tokens_rewake"],
                )
                r["rewakeout_wake"] = rewakeout_wake
                r["rewake_wake_skipped_reason"] = skip_reason
                params["rewake_wake"] = True
                params["rewake_wake_score_threshold"] = score_threshold
                if rewakeout_wake:
                    ok += 1
                else:
                    skipped += 1
            except Exception as e:
                r["rewakeout_wake"] = None
                r["rewake_wake_skipped_reason"] = f"error: {type(e).__name__}: {e}"
                failed += 1

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[sweep_rewake_wake] in={in_path}")
    print(f"[sweep_rewake_wake] out={out_path}")
    print(f"[sweep_rewake_wake] total={len(records)} ok={ok} skipped={skipped} excluded={excluded} failed={failed}")


if __name__ == "__main__":
    main()
