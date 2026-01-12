# sweep.py: parameter sweep runner for ReMIND v1.0
# Runs a minimal sweep (405 runs) over (pair, template_id, word_limit, temp_dream, seed_dream).
# Each run is appended as one JSON record per line to: logs/remind_sweep_<UTCtimestamp>.jsonl
from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable
from ensure_tmux import ensure_tmux
from remind import run_remind, DEFAULT_PAIRS, DEFAULT_LLM

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def iter_conditions() -> Iterable[dict]:
    # ---- minimal sweep set (405 runs) ----
    # pairs: 3
    # template_id: 3
    # word_limit: 75, 150, 300
    # temp_dream: 1, 3, 10
    # seed_dream: 0..4 (5)
    word_limits = [75, 150, 300]
    template_ids = [0, 1, 2]
    temp_dreams = [1.0, 3.0, 10.0]
    seed_dreams = list(range(0, 5))

    for pair in DEFAULT_PAIRS:
        for template_id in template_ids:
            for word_limit in word_limits:
                for temp_dream in temp_dreams:
                    for seed_dream in seed_dreams:
                        yield {
                            "pair": pair,
                            "template_id": template_id,
                            "word_limit": word_limit,
                            "temp_dream": temp_dream,
                            "seed_dream": seed_dream,
                        }

def main() -> None:
    ensure_tmux()
    out_dir = Path("logs")
    out_path = out_dir / f"remind_sweep_{utc_ts()}.jsonl"

    fixed = dict(
        llm_wake  = "gemma27b", # override DEFAULT_LLM["wake"]
        llm_dream = DEFAULT_LLM["dream"],
        llm_judge = "oss120b", # override DEFAULT_LLM["judge"]
        temp_wake=0.6,
        temp_judge=0.0,
        temp_rewake=0.6,
        seed_wake=0,
        seed_judge=0,
        seed_rewake=0,
        score_threshold=4,
        verbose=False,
        max_tokens_judge=200,
    )

    total = 0
    ok = 0
    failed = 0
    start = time.time()

    print(f"[sweep] output -> {out_path}")

    for i, cond in enumerate(iter_conditions(), start=1):
        total += 1
        record = {
            "run_id": i,
            "sweep": {
                "pair": cond["pair"],
                "template_id": cond["template_id"],
                "word_limit": cond["word_limit"],
                "temp_dream": cond["temp_dream"],
                "seed_dream": cond["seed_dream"],
            },
            "meta": {
                "ts_utc": utc_ts(),
                "status": "init",
            },
        }

        try:
            result = run_remind(
                pair=cond["pair"],
                template_id=cond["template_id"],
                word_limit=cond["word_limit"],
                temp_dream=cond["temp_dream"],
                seed_dream=cond["seed_dream"],
                **fixed,
            )

            # merge: keep sweep/meta separate, store full result under "result"
            record["result"] = result
            record["meta"]["status"] = "ok"
            ok += 1

        except Exception as e:
            record["meta"]["status"] = "error"
            record["meta"]["error_type"] = type(e).__name__
            record["meta"]["error"] = str(e)
            failed += 1

        jsonl_append(out_path, record)

        # progress print
        if i % 10 == 0:
            elapsed = time.time() - start
            print(f"[sweep] {i}/{405}  ok={ok}  failed={failed}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"[sweep] done. total={total} ok={ok} failed={failed} elapsed={elapsed:.1f}s")

if __name__ == "__main__":
    main()