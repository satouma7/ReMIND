# sweep.py: parameter sweep runner for ReMIND v1.0 (core/social split)
# - core (default) : CORE_PAIRS  x templates [0,1,2]
# - social         : SOCIAL_PAIRS x templates [3]
# LLM selection (3-letter code: wake/dream/judge): see LLM_CODE_MAP in config.py
#   e.g. ooo, oto, ogo
# Usage:
#   Single sweep:
#     python sweep.py --llm oto
#   Batch sweep (input order, shared timestamp, stop_unused=True):
#     python sweep.py --batch ooo ogo oog oto
#   Fast sanity test (3 runs per sweep):
#     python sweep.py --batch ooo ogo --test
# Output:
#   logs/remind_sweep_<llmcode>_<UTCtimestamp>_<topic>.jsonl
#   (batch uses the SAME timestamp for all codes)
# Behavior:
#   - Batch prints a plan summary before running
#   - Errors in one code do NOT stop the batch (A-plan)
#   - Output files are OVERWRITTEN at sweep start
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Dict, List
from ensure_tmux import ensure_tmux
from remind import run_remind
from remind import CORE_PAIRS, SOCIAL_PAIRS, SOCIAL_PAIRS_MAIN
from config import LLM_CODE_MAP

# ---------- utils ----------
def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def jsonl_append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def truncate_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8"):
        pass


# ---------- LLM code parsing ----------


def parse_llm_code(code: str) -> Dict[str, str]:
    code = (code or "").strip()
    if code == "":
        code = "ooo"
    if len(code) != 3:
        raise ValueError(f"--llm must be 3 letters (e.g., ooo, oto, ono). got: {code}")

    try:
        return {
            "llm_code": code,
            "llm_wake": LLM_CODE_MAP[code[0]],
            "llm_dream": LLM_CODE_MAP[code[1]],
            "llm_judge": LLM_CODE_MAP[code[2]],
        }
    except KeyError as e:
        raise ValueError(
            f"Unknown llm code '{e.args[0]}'. Allowed: {sorted(LLM_CODE_MAP.keys())}"
        ) from None


# ---------- sweep condition generator ----------

def iter_conditions(topic: str, *, test: bool = False, all_social: bool = False) -> Iterable[dict]:
    word_limits = [75, 150, 300]
    temp_dreams = [1.0, 3.0, 10.0]#[0, 0.3, 0.6, 1.0, 3.0, 10.0]
    seed_dreams = list(range(0, 5))

    if topic == "core":
        pairs = CORE_PAIRS
        template_ids = [0, 1, 2]
    elif topic == "social":
        pairs = SOCIAL_PAIRS if all_social else SOCIAL_PAIRS_MAIN
        template_ids = [3]
        temp_dreams = [1.0, 3.0, 10.0]
    else:
        raise ValueError(f"topic must be 'core' or 'social', got: {topic}")

    if test:
        pairs = list(pairs)[:3]
        template_ids = [min(template_ids)]
        word_limits = [min(word_limits)]
        temp_dreams = [max(temp_dreams)]
        seed_dreams = [min(seed_dreams)]

    for pair in pairs:
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


# ---------- single sweep runner ----------

def run_one_sweep(*, topic: str, llm_code: str, out_path: Path,
                  test: bool, stop_unused: bool, all_social: bool = False) -> dict:

    sel = parse_llm_code(llm_code)

    fixed = dict(
        llm_wake=sel["llm_wake"],
        llm_dream=sel["llm_dream"],
        llm_judge=sel["llm_judge"],
        temp_wake=0.0,#0.6
        temp_judge=0.0,
        temp_rewake=0.0,#0.6
        seed_wake=0,
        seed_judge=0,
        seed_rewake=0,
        score_threshold=4,
        verbose=False,
        max_tokens_judge=200,
    )

    required_llms = sorted({fixed["llm_wake"], fixed["llm_dream"], fixed["llm_judge"]})
    ensure_tmux(required=required_llms, stop_unused=stop_unused)

    truncate_file(out_path)

    ok = failed = total = 0
    start = time.time()

    expected = sum(1 for _ in iter_conditions(topic, test=test, all_social=all_social))

    print(f"[sweep] topic={topic} llm={llm_code} test={test} expected={expected}")
    print(f"[sweep] output={out_path}")

    for i, cond in enumerate(iter_conditions(topic, test=test, all_social=all_social), start=1):
        total += 1
        record = {
            "run_id": i,
            "sweep": cond,
            "meta": {
                "ts_utc": utc_ts(),
                "status": "init",
                "llm_code": llm_code,
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
            record["result"] = result
            record["meta"]["status"] = "ok"
            ok += 1

        except Exception as e:
            record["result"] = None
            record["meta"]["status"] = "error"
            record["meta"]["error_type"] = type(e).__name__
            record["meta"]["error"] = str(e)
            failed += 1

        jsonl_append(out_path, record)

    elapsed = time.time() - start
    print(f"[sweep] done. total={total} ok={ok} failed={failed} elapsed={elapsed:.1f}s")

    return {
        "llm_code": llm_code,
        "ok": ok,
        "failed": failed,
        "elapsed": elapsed,
        "out": str(out_path),
    }


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="ReMIND sweep runner.")
    ap.add_argument("--topic", choices=["core", "social"], default="core")
    ap.add_argument("--llm", type=str, default="ooo",
                    help="3-letter code for single sweep")
    ap.add_argument("--batch", nargs="*", default=[],
                    help="Run multiple sweeps sequentially")
    ap.add_argument("--test", action="store_true",
                    help="Run a tiny sweep (3 runs)")
    ap.add_argument("--all-social", action="store_true",
                    help="Use all 10 SOCIAL_PAIRS (default: primary 3 only)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    codes: List[str] = args.batch if args.batch else [args.llm]
    batch_ts = utc_ts()

    # ---- plan summary ----
    print("[batch] plan:")
    for c in codes:
        s = parse_llm_code(c)
        print(f"  - {c}: wake={s['llm_wake']} dream={s['llm_dream']} judge={s['llm_judge']}")
    print(f"[batch] topic={args.topic} test={args.test} ts={batch_ts}\n")

    summaries = []

    for c in codes:
        out_path = Path(args.out).expanduser() if args.out else (
            Path("logs") / f"remind_sweep_{c}_{batch_ts}_{args.topic}.jsonl"
        )

        try:
            summaries.append(
                run_one_sweep(
                    topic=args.topic,
                    llm_code=c,
                    out_path=out_path,
                    test=args.test,
                    stop_unused=True,
                    all_social=args.all_social,
                )
            )
        except Exception as e:
            print(f"[batch] ERROR on {c}: {type(e).__name__}: {e}")
            summaries.append({
                "llm_code": c,
                "ok": 0,
                "failed": -1,
                "elapsed": 0.0,
                "out": str(out_path),
                "error": str(e),
            })

    print("\n[batch] summary:")
    for s in summaries:
        print(f"  - {s['llm_code']}: ok={s.get('ok')} failed={s.get('failed')} out={s.get('out')}")


if __name__ == "__main__":
    main()