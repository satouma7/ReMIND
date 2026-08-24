# sweep_dream.py: parameter sweep runner for dream module in ReMIND v1.0
# Runs a minimal sweep (405 runs) over (pair, template_id, word_limit, temp_dream, seed_dream).
# Each run is appended as one JSON record per line to: logs/dream_sweep_<llm>_<UTCtimestamp>.jsonl
# LLM selection (1-letter code): see LLM_CODE_MAP in sweep.py
# Usage:
#   python sweep_dream.py --llm g
#   python sweep_dream.py --llm o --test
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Any, Optional
from ensure_tmux import ensure_tmux
from remind import CORE_PAIRS
from dream import dream
from config import LLM_CODE_MAP

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

    for pair in CORE_PAIRS:
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

def build_prompts(pair: tuple[str, str], word_limit: int = 150) -> list[str]:
    a, b = pair
    return [
        f'Compare the meaning of "{a}" and "{b}" within {word_limit} words.',
        f'Describe the unexpected relationship between "{a}" and "{b}" within {word_limit} words.',
        f'Propose a new idea about the relationship between "{a}" and "{b}" within {word_limit} words.',
    ]

def run_dream(
    *,
    pair: tuple[str, str],
    template_id: int = 2,
    word_limit: int = 150,
    max_tokens_dream: Optional[int] = None,
    llm_dream: str = "oss120b",
    temp_dream: float = 10.0,
    seed_dream: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:

    prompts = build_prompts(pair, word_limit=word_limit)
    if not (0 <= template_id < len(prompts)):
        raise ValueError(f"template_id must be 0..{len(prompts)-1}, got {template_id}")
    prompt = prompts[template_id]

    if max_tokens_dream is None:
        max_tokens_dream = word_limit + 50

    if verbose:
        print(prompt)

    # ---- DREAM ----
    dreamout = dream(
        prompt,
        llm=llm_dream,
        max_tokens=max_tokens_dream,
        temperature=temp_dream,
        seed=seed_dream,
    )
    if verbose:
        print(f"\n=== DREAM (LLM={llm_dream}:temp={temp_dream}:seed={seed_dream}) ===")
        print(dreamout)

    return {
        "pair": pair,
        "template_id": template_id,
        "word_limit": word_limit,
        "prompt": prompt,
        "params": {
            "llm_dream": llm_dream,
            "temp_dream": temp_dream,
            "seed_dream": seed_dream,
            "max_tokens_dream": max_tokens_dream,
        },
        "dreamout": dreamout,
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="ReMIND dream-only sweep runner.")
    ap.add_argument("--llm", type=str, default="g",
                    help=f"1-letter code for dream LLM (default: g = gemma4_31b). "
                         f"Allowed: {sorted(LLM_CODE_MAP.keys())}")
    ap.add_argument("--test", action="store_true",
                    help="Run a tiny sweep (1 condition)")
    args = ap.parse_args()

    if args.llm not in LLM_CODE_MAP:
        ap.error(f"Unknown --llm '{args.llm}'. Allowed: {sorted(LLM_CODE_MAP.keys())}")

    llm_dream = LLM_CODE_MAP[args.llm]

    ensure_tmux(required=[llm_dream])
    out_path = Path("logs") / f"dream_sweep_{args.llm}_{utc_ts()}.jsonl"

    conditions = list(iter_conditions())
    if args.test:
        conditions = conditions[:1]
    expected = len(conditions)

    print(f"[sweep_dream] llm={args.llm} ({llm_dream}) test={args.test} expected={expected}")
    print(f"[sweep_dream] output -> {out_path}")

    total = ok = failed = 0
    start = time.time()

    for i, cond in enumerate(conditions, start=1):
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
                "llm_dream": llm_dream,
            },
        }

        try:
            result = run_dream(
                pair=cond["pair"],
                template_id=cond["template_id"],
                word_limit=cond["word_limit"],
                temp_dream=cond["temp_dream"],
                seed_dream=cond["seed_dream"],
                llm_dream=llm_dream,
            )
            record["result"] = result
            record["meta"]["status"] = "ok"
            ok += 1

        except Exception as e:
            record["meta"]["status"] = "error"
            record["meta"]["error_type"] = type(e).__name__
            record["meta"]["error"] = str(e)
            failed += 1

        jsonl_append(out_path, record)

        if i % 10 == 0:
            elapsed = time.time() - start
            print(f"[sweep_dream] {i}/{expected}  ok={ok}  failed={failed}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"[dream] done. total={total} ok={ok} failed={failed} elapsed={elapsed:.1f}s")

if __name__ == "__main__":
    main()
