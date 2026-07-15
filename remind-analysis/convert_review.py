# convert_review.py
# Convert legacy ReMIND review JSONL files to unified v2 format.
# - rewakeout -> text
# - add target="rewake" if missing
# - keep all review scores intact
# Usage:
#   python convert_review.py remind_review_XXXX.jsonl
import json
import argparse
from pathlib import Path

def auto_output_path(in_path: Path) -> Path:
    """
    logs/foo.jsonl -> logs/foo_rewake.jsonl
    """
    stem = in_path.stem
    if stem.endswith("_rewake"):
        return in_path  # already converted
    return in_path.with_name(stem + "_rewake.jsonl")

def reorder_record(obj: dict) -> dict:
    # run_id, target を先頭固定。それ以外は元の順序で続ける。
    run_id = obj.get("run_id")
    target = obj.get("target")

    out = {}
    if run_id is not None:
        out["run_id"] = run_id
    if target is not None:
        out["target"] = target

    for k, v in obj.items():
        if k in ("run_id", "target"):
            continue
        out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert legacy review JSONL to v2 format")
    ap.add_argument(
        "input",
        help="input review jsonl (default folder: logs/)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        alt = Path("logs") / in_path
        if alt.exists():
            in_path = alt
        else:
            raise FileNotFoundError(in_path)

    out_path = auto_output_path(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            obj = json.loads(line)

            # already new format
            if "target" in obj and "text" in obj:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            # legacy rewake format
            if "rewakeout" in obj:
                obj["target"] = "rewake"
                obj["text"] = obj.pop("rewakeout")
                obj = reorder_record(obj)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                converted += 1
            else:
                # unknown format → skip silently
                skipped += 1

    print(f"[convert] input : {in_path}")
    print(f"[convert] output: {out_path}")
    print(f"[convert] converted={converted}, skipped={skipped}")

if __name__ == "__main__":
    main()