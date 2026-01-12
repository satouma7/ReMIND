# review.py: external LLM review runner for ReMIND v1.0
# - Runs independent external LLM reviewers on (prompt, rewakeout) pairs
# - Reads remind_sweep_*.jsonl
# - For each row, sends (prompt, rewakeout) to external LLM(s)
# - Writes remind_review_*.jsonl with appended review JSON
# Usage:
#   python review.py /path/to/remind_sweep_XXXX.jsonl
#   python review.py /path/to/remind_sweep_XXXX.jsonl --out logs/remind_review_XXXX.jsonl --only openai
#   python review.py /path/to/remind_sweep_XXXX.jsonl --max-runs 50 --sleep 0.5 --resume
# Output:
# - Creates remind_review_*.jsonl with same structure as input but with additional "review" field
# Env (recommended):
#   export OPENAI_API_KEY="..."
#   export GOOGLE_API_KEY="..."
from __future__ import annotations
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import requests

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def jsonl_append(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_existing_run_ids(out_path: Path) -> Set[int]:
    """For --resume: read existing output JSONL and collect run_id that are already written."""
    if not out_path.exists():
        return set()
    done: Set[int] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("run_id")
                if isinstance(rid, int):
                    done.add(rid)
            except Exception:
                continue
    return done

def is_skipped_rewake(rewakeout: Any) -> bool:
    if rewakeout is None:
        return True
    if not isinstance(rewakeout, str):
        return True
    s = rewakeout.strip()
    if not s:
        return True
    if s.lower() in {"(skipped)", "skipped"}:
        return True
    return False

def build_review_prompt(prompt: str, rewakeout: str) -> Tuple[str, str]:
    """Return (system_text, user_text)."""
    system_text = (
        "You are an independent expert reviewer evaluating an AI-generated response.\n\n"
        "Evaluate REWAKE as an answer to PROMPT.\n\n"
        "Score the response according to the following criteria.\n"
        "Scores must be integers from 1 (very poor) to 5 (excellent).\n\n"
        "1. Alignment:\n"
        "Does REWAKE properly address PROMPT without drifting off-topic?\n\n"
        "2. Coherence:\n"
        "Is REWAKE internally consistent, logically structured, and understandable?\n\n"
        "3. Novelty:\n"
        "Does REWAKE introduce genuinely new or creative ideas beyond a trivial restatement?\n\n"
        "Return your evaluation strictly in the following JSON format:\n\n"
        "{\n"
        '  "alignment": <int 1-5>,\n'
        '  "coherence": <int 1-5>,\n'
        '  "novelty": <int 1-5>,\n'
        '  "short_rationale": "2–4 sentences explaining your scores"\n'
        "}\n"
    )

    user_text = (
        "[PROMPT]\n"
        f"{prompt}\n\n"
        "[REWAKE]\n"
        f"{rewakeout}\n"
    )
    return system_text, user_text


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Parse the first {...} block found in the LLM output as JSON.
    If no JSON object is found, return an error structure that includes the raw text.
    """
    text = text.strip()
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {
            "_parse_error": "No JSON object found in model output",
            "_raw_text": text,
        }

    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        return {
            "_parse_error": f"JSON decode error: {e}",
            "_raw_text": blob,
        }


def clamp_score(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        return 1
    return max(1, min(5, v))

def normalize_review(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alignment": clamp_score(obj.get("alignment")),
        "coherence": clamp_score(obj.get("coherence")),
        "novelty": clamp_score(obj.get("novelty")),
        "short_rationale": str(obj.get("short_rationale", "")).strip(),
    }

# OpenAI Reviewer (HTTP)
@dataclass
class OpenAIReviewer:
    api_key: str
    model: str = "gpt-5.2"
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 120
    max_retries: int = 3

    def review(self, prompt: str, rewakeout: str) -> Dict[str, Any]:
        system_text, user_text = build_review_prompt(prompt, rewakeout)
        # We choose chat/completions for compatibility with OpenAI-compatible servers.
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 300,
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise RuntimeError(f"OpenAI HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                parsed = extract_json_object(content)
                return normalize_review(parsed)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
                else:
                    raise
        raise last_err or RuntimeError("OpenAI review failed unexpectedly")

# Gemini Reviewer (HTTP)
@dataclass
class GeminiReviewer:
    api_key: str
    model: str = "gemini-3-flash-preview"
    base_url: str = "https://generativelanguage.googleapis.com"
    timeout_s: int = 120
    max_retries: int = 3

    def review(self, prompt: str, rewakeout: str) -> Dict[str, Any]:
        system_text, user_text = build_review_prompt(prompt, rewakeout)

        # Gemini Generative Language API:
        # POST {base}/v1beta/models/{model}:generateContent?key=API_KEY
        url = f"{self.base_url.rstrip('/')}/v1beta/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": system_text + "\n\n" + user_text}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 300,
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, params=params, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()

                # Typical shape:
                # { "candidates":[{"content":{"parts":[{"text":"..."}]}}], ... }
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:500]}")
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if not parts or "text" not in parts[0]:
                    raise RuntimeError(f"Gemini unexpected response: {json.dumps(data)[:500]}")
                content = parts[0]["text"]

                parsed = extract_json_object(content)
                return normalize_review(parsed)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
                else:
                    raise
        raise last_err or RuntimeError("Gemini review failed unexpectedly")

# Main
def iter_review_items(records: List[Dict[str, Any]]) -> Iterable[Tuple[int, Dict[str, Any], str, str]]:
    """
    Yield (run_id, original_record, prompt, rewakeout) for valid items.
    Works with sweep JSONL format where data is under record["result"].
    """
    for r in records:
        run_id = r.get("run_id")
        if not isinstance(run_id, int):
            # Skip malformed rows
            continue

        res = r.get("result", r)
        prompt = res.get("prompt", "")
        rewakeout = res.get("rewakeout", None)

        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if is_skipped_rewake(rewakeout):
            continue

        yield run_id, r, prompt, str(rewakeout)


def default_out_path(in_path: Path) -> Path:
    # remind_sweep_XXXX.jsonl -> logs/remind_review_XXXX.jsonl (same folder as input by default)
    name = in_path.name
    suffix = name.replace("remind_sweep_", "").replace(".jsonl", "")
    out_name = f"remind_review_{suffix or utc_ts()}.jsonl"
    return in_path.parent / out_name


def main() -> None:
    ap = argparse.ArgumentParser(description="External LLM review for ReMIND (prompt vs rewakeout).")
    ap.add_argument("jsonl", type=str, help="path to remind_sweep_*.jsonl")
    ap.add_argument("--out", type=str, default="", help="output JSONL path (default: next to input)")
    ap.add_argument("--only", choices=["both", "openai", "gemini"], default="both", help="which reviewer(s) to run")
    ap.add_argument("--openai-model", type=str, default="gpt-5.2", help="OpenAI model name")
    ap.add_argument("--gemini-model", type=str, default="gemini-3-flash-preview", help="Gemini model name")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between API calls")
    ap.add_argument("--max-runs", type=int, default=0, help="limit number of reviewed items (0=all)")
    ap.add_argument("--resume", action="store_true", help="skip run_id already present in output JSONL")
    ap.add_argument("--verbose", action="store_true", help="print progress for each run")
    args = ap.parse_args()

    in_path = Path(args.jsonl).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    out_path = Path(args.out).expanduser() if args.out else default_out_path(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # API keys
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")).strip()

    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    gemini_base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").strip()

    openai_reviewer: Optional[OpenAIReviewer] = None
    gemini_reviewer: Optional[GeminiReviewer] = None

    if args.only in ("both", "openai"):
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set (required for --only openai/both).")
        openai_reviewer = OpenAIReviewer(api_key=openai_key, model=args.openai_model, base_url=openai_base)

    if args.only in ("both", "gemini"):
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set (required for --only gemini/both).")
        gemini_reviewer = GeminiReviewer(api_key=gemini_key, model=args.gemini_model, base_url=gemini_base)

    print(f"[review] input : {in_path}")
    print(f"[review] output: {out_path}")
    print(f"[review] mode  : {args.only}")
    if openai_reviewer:
        print(f"[review] openai : model={openai_reviewer.model} base={openai_reviewer.base_url}")
    if gemini_reviewer:
        print(f"[review] gemini : model={gemini_reviewer.model} base={gemini_reviewer.base_url}")

    records = load_jsonl(in_path)
    items = list(iter_review_items(records))
    print(f"[review] loaded records: {len(records)}")
    print(f"[review] valid (has prompt & rewakeout): {len(items)}")

    done_ids: Set[int] = set()
    if args.resume:
        done_ids = load_existing_run_ids(out_path)
        print(f"[review] resume enabled: already done run_id count = {len(done_ids)}")

    reviewed = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for idx, (run_id, orig, prompt, rewakeout) in enumerate(items, start=1):
        if args.max_runs and reviewed >= args.max_runs:
            break
        if args.resume and run_id in done_ids:
            skipped += 1
            continue

        sweep = orig.get("sweep", {})
        res = orig.get("result", orig)

        out_rec: Dict[str, Any] = {
            "run_id": run_id,
            "sweep": sweep,
            "pair": res.get("pair", sweep.get("pair")),
            "template_id": res.get("template_id", sweep.get("template_id")),
            "word_limit": res.get("word_limit", sweep.get("word_limit")),
            "temp_dream": sweep.get("temp_dream"),
            "seed_dream": sweep.get("seed_dream"),
            "prompt": prompt,
            "rewakeout": rewakeout,
            "meta": {
                "ts_utc": utc_ts(),
                "status": "init",
            },
            "reviews": {},
        }

        try:
            if openai_reviewer:
                if args.verbose:
                    print(f"[review] run_id={run_id} -> OpenAI ...")
                out_rec["reviews"]["openai"] = {
                    "model": openai_reviewer.model,
                    **openai_reviewer.review(prompt, rewakeout),
                }

            if gemini_reviewer:
                if args.verbose:
                    print(f"[review] run_id={run_id} -> Gemini ...")
                out_rec["reviews"]["gemini"] = {
                    "model": gemini_reviewer.model,
                    **gemini_reviewer.review(prompt, rewakeout),
                }

            out_rec["meta"]["status"] = "ok"
            reviewed += 1

        except Exception as e:
            out_rec["meta"]["status"] = "error"
            out_rec["meta"]["error_type"] = type(e).__name__
            out_rec["meta"]["error"] = str(e)
            failed += 1

        jsonl_append(out_path, out_rec)

        # pacing
        time.sleep(max(0.0, args.sleep))

        # progress
        if (reviewed + failed) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[review] processed={reviewed+failed} ok={reviewed} failed={failed} skipped={skipped} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[review] done. ok={reviewed} failed={failed} skipped={skipped} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()