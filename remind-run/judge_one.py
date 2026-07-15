# judge_one.py
# Feed a single text to the judge and dump {score, idea, raw}.
# Usage: 
#   python judge_one.py --wrap --llm oss120b --temp 1 --seed 0 --max-tokens 200 --text '...dream_out...'
#   python judge_one.py --wrap --file dreamout_316.txt
import argparse
import json
import sys
from pathlib import Path

# judge.py 側に judge_raw を足した前提
from judge import judge_raw

def parse_two_lines(raw: str) -> dict:
    """
    Robust-ish parser for:
      SCORE: <int>
      IDEA: <text or EMPTY>
    """
    out = {"score": None, "idea": "", "raw": raw}
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip() != ""]
    # 余計な行が混ざることがあるので、SCORE/IDEA を含む行を探索
    score_line = next((ln for ln in lines if ln.startswith("SCORE:")), "")
    idea_line  = next((ln for ln in lines if ln.startswith("IDEA:")), "")

    if score_line:
        try:
            out["score"] = int(score_line.split(":", 1)[1].strip())
        except Exception:
            out["score"] = None

    if idea_line:
        idea = idea_line.split(":", 1)[1].strip()
        out["idea"] = "" if idea.upper() == "EMPTY" else idea

    return out

def read_text(args) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    # stdin
    return sys.stdin.read()

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--text", help="Direct input text")
    src.add_argument("--file", help="Path to a text file containing dream_out")
    ap.add_argument("--llm", default="oss120b")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrap", action="store_true",
                    help="Wrap input with ===BEGIN===/===END=== like sweep does")
    ap.add_argument("--out", default=None, help="Write JSON to this path (optional)")
    args = ap.parse_args()

    text = read_text(args).strip()
    if not text:
        raise SystemExit("Empty input text.")

    judge_input = f"===BEGIN===\n{text}\n===END===" if args.wrap else text

    raw = judge_raw(
        judge_input,
        llm=args.llm,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        seed=args.seed,
    )

    parsed = parse_two_lines(raw)
    payload = {
        "score": parsed["score"],
        "idea": parsed["idea"],
        "raw": parsed["raw"],
        "llm": args.llm,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "seed": args.seed,
    }

    j = json.dumps(payload, ensure_ascii=False, indent=2)
    print(j)

    if args.out:
        Path(args.out).write_text(j + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()