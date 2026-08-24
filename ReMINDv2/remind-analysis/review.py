# review.py: external LLM review runner for ReMIND v1.0
# - Reviews ONE target per run: wake | dream | rewake (default: rewake)
# - Or reviews ALL targets with --all, but writes THREE separate files:
#     *_wake.jsonl, *_dream.jsonl, *_rewake.jsonl
# Usage:
#   python review.py logs/remind_sweep_XXXX.jsonl
#   python review.py logs/remind_sweep_XXXX.jsonl --wake
#   python review.py logs/remind_sweep_XXXX.jsonl --dream
#   python review.py logs/remind_sweep_XXXX.jsonl --rewake
#   python review.py logs/remind_sweep_XXXX.jsonl --all
# Options:
#   --model gpt-5-mini|gpt-5.2|gemini   (default: gpt-5-mini)
#   --resume: skip already reviewed (run_id, target) already present in the corresponding output
# - OpenAI: gpt-5.2 uses temperature=0; gpt-5-mini omits temperature
# - seed supported (default 0) for best-effort determinism
# - Enforce JSON output via response_format when possible
# - If JSON parse fails, mark meta.status=error (do NOT silently clamp to 1)

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

# -------------------------
# Utilities
# -------------------------
def _truncate(s: str, n: int = 2000) -> str:
    s = s if isinstance(s, str) else str(s)
    return s[:n]

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

def is_blank_text(x: Any) -> bool:
    if x is None or not isinstance(x, str):
        return True
    s = x.strip()
    if not s:
        return True
    if s.lower() in {"(skipped)", "skipped", "none", "null"}:
        return True
    return False

# -------------------------
# Prompt builder
# -------------------------
def build_review_prompt(prompt: str, target: str, text: str) -> Tuple[str, str]:
    system_text = (
        "You are an independent expert reviewer evaluating an AI-generated response.\n\n"
        "Evaluate TEXT as an answer to PROMPT.\n\n"
        "Score the response according to the following criteria.\n"
        "Scores must be integers from 1 (very poor) to 5 (excellent).\n\n"
        "1. Alignment:\n"
        "Does TEXT properly address PROMPT without drifting off-topic?\n\n"
        "2. Coherence:\n"
        "Is TEXT internally consistent, logically structured, and understandable?\n\n"
        "3. Novelty:\n"
        "Does TEXT introduce genuinely new or creative ideas beyond a trivial restatement?\n\n"
        "IMPORTANT:\n"
        "- Output JSON ONLY.\n"
        "- Do not wrap in Markdown.\n"
        "- Use exactly these keys: alignment, coherence, novelty, short_rationale.\n\n"
        "Return strictly:\n"
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
        f"[TARGET]\n{target}\n\n"
        "[TEXT]\n"
        f"{text}\n"
    )
    return system_text, user_text

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {"_parse_error": "No JSON object found", "_raw_text": text}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        return {"_parse_error": f"JSON decode error: {e}", "_raw_text": blob}

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

def has_parse_error(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and ("_parse_error" in obj)

# -------------------------
# Reviewers
# -------------------------
@dataclass
class OpenAIReviewer:
    api_key: str
    model: str = "gpt-5-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 120
    seed: int = 0

    def review(self, prompt: str, target: str, text: str) -> Dict[str, Any]:

        system_text, user_text = build_review_prompt(prompt, target, text)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_completion_tokens": 300,
            "seed": int(self.seed),
            "response_format": {"type": "json_object"},
        }

        if self.model == "gpt-5.2":
            payload["temperature"] = 0.0
            payload["reasoning_effort"] = "none"

        if self.model == "gpt-5-mini":
            payload["reasoning_effort"] = "minimal"

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)

        raw_http_status = r.status_code
        raw_http_text = r.text

        if raw_http_status >= 400:
            raise RuntimeError(f"OpenAI HTTP {raw_http_status}: {raw_http_text[:1000]}")

        data = r.json()
        raw_response_json = json.dumps(data, ensure_ascii=False)[:5000]

        # ----- content 抽出 -----
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content")

        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            content = "\n".join(parts).strip()

        if not isinstance(content, str):
            content = ""

        # tool_calls fallback
        if not content:
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                fn = (tool_calls[0].get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    content = args.strip()

        parsed = extract_json_object(content)

        if "_parse_error" in parsed:
            return {
                "_error": "json_parse_failed",
                "_raw_model_output": _truncate(content),
                "_raw_http_response": _truncate(raw_http_text),
                "_raw_response_json": raw_response_json,
            }

        return normalize_review(parsed)
    
@dataclass
class GeminiReviewer:
    api_key: str
    model: str = "gemini-3-flash-preview"
    base_url: str = "https://generativelanguage.googleapis.com"
    timeout_s: int = 120
    max_retries: int = 3

    def review(self, prompt: str, target: str, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        system_text, user_text = build_review_prompt(prompt, target, text)
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
                "thinkingConfig": {
                    "thinkingBudget": 0,
                },
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(url, params=params, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code >= 400:
                    raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:500]}")
                parts = (candidates[0].get("content") or {}).get("parts") or []
                if not parts or "text" not in parts[0]:
                    raise RuntimeError(f"Gemini unexpected response: {json.dumps(data)[:500]}")
                content = parts[0]["text"]
                parsed = extract_json_object(content)

                if has_parse_error(parsed):
                    return (
                        normalize_review({}),
                        {"parse_error": parsed.get("_parse_error"), "raw_text": parsed.get("_raw_text", "")},
                    )
                return normalize_review(parsed), {"raw_text": content}

            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
                else:
                    raise
        raise last_err or RuntimeError("Gemini review failed unexpectedly")

# -------------------------
# Target selection + iterator
# -------------------------
def choose_targets(args: argparse.Namespace) -> List[str]:
    if getattr(args, "all", False):
        return ["wake", "dream", "rewake"]
    flags = {"wake": bool(args.wake), "dream": bool(args.dream), "rewake": bool(args.rewake)}
    chosen = [k for k, v in flags.items() if v]
    if len(chosen) == 0:
        return ["rewake"]
    if len(chosen) > 1:
        raise ValueError(f"Choose only one target among --wake/--dream/--rewake (got: {chosen})")
    return [chosen[0]]

def get_text_for_target(res: Dict[str, Any], target: str) -> Optional[str]:
    if target == "wake":
        return res.get("wakeout")
    if target == "dream":
        return res.get("dreamout")
    if target == "rewake":
        return res.get("rewakeout")
    raise ValueError(target)

def iter_review_items(
    records: List[Dict[str, Any]],
    targets: List[str],
    min_temp_dream: Optional[float] = None,
) -> Iterable[Tuple[int, Dict[str, Any], str, str, str]]:
    for r in records:
        run_id = r.get("run_id")
        if not isinstance(run_id, int):
            continue
        res = r.get("result", r)
        if not isinstance(res, dict):
            continue
        if min_temp_dream is not None:
            td = r.get("sweep", {}).get("temp_dream")
            if td is None:
                td = res.get("temp_dream")
            if td is None or float(td) < min_temp_dream:
                continue
        prompt = res.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        for target in targets:
            text = get_text_for_target(res, target)
            if is_blank_text(text):
                continue
            yield run_id, r, prompt, target, str(text).strip()

def model_tag(model_choice: str, openai_model: str = "", gemini_model: str = "") -> str:
    """
    Convert model choice/name to a stable filename tag.
    Desired:
      gpt-5.2   -> _gpt5.2
      gpt-5-mini -> _gpt5mini
      gemini    -> _gemini
    """
    mc = (model_choice or "").strip().lower()

    if mc in {"gpt-5.2", "gpt5.2"}:
        return "_gpt5.2"
    if mc in {"gpt-5-mini", "gpt5-mini"}:
        return "_gpt5mini"
    if mc == "gemini":
        return "_gemini"

    # Fallback: try infer from provided override names
    name = (openai_model or gemini_model or mc).lower()
    if "gpt-5.2" in name or "gpt5.2" in name:
        return "_gpt5.2"
    if "gpt-5-mini" in name or "gpt5-mini" in name:
        return "_gpt5mini"
    if "gemini" in name:
        return "_gemini"

    # last resort: sanitize
    safe = re.sub(r"[^a-z0-9._-]+", "", name)[:24]
    return f"_{safe}" if safe else ""

# -------------------------
# Output naming + resume keys
# -------------------------
def suffix_from_input_name(in_path: Path) -> str:
    name = in_path.name
    if name.startswith("remind_sweep_") and name.endswith(".jsonl"):
        return name.replace("remind_sweep_", "").replace(".jsonl", "")
    return in_path.stem

def default_out_path(in_path: Path, target: str, *, tag: str = "") -> Path:
    suffix = suffix_from_input_name(in_path)
    out_name = f"remind_review_{suffix}_{target}{tag}.jsonl"
    return in_path.parent / out_name

def load_existing_keys(out_path: Path) -> Set[Tuple[int, str]]:
    if not out_path.exists():
        return set()
    done: Set[Tuple[int, str]] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("run_id")
                tgt = obj.get("target")
                if isinstance(rid, int) and isinstance(tgt, str):
                    done.add((rid, tgt))
            except Exception:
                continue
    return done

# -------------------------
# Core runner (per target)
# -------------------------
def run_review_for_target(
    *,
    records: List[Dict[str, Any]],
    target: str,
    out_path: Path,
    model_choice: str,
    openai_reviewer: Optional[OpenAIReviewer],
    gemini_reviewer: Optional[GeminiReviewer],
    sleep_s: float,
    max_runs: int,
    resume: bool,
    verbose: bool,
    min_temp_dream: Optional[float] = None,
) -> Tuple[int, int, int, float]:
    items = list(iter_review_items(records, [target], min_temp_dream=min_temp_dream))
    print(f"[review:{target}] output: {out_path}")
    print(f"[review:{target}] valid items: {len(items)}")

    done_keys: Set[Tuple[int, str]] = set()
    if resume:
        done_keys = load_existing_keys(out_path)
        print(f"[review:{target}] resume enabled: already done count = {len(done_keys)}")

    reviewed = skipped = failed = 0
    t0 = time.time()

    for idx, (run_id, orig, prompt, tgt, text) in enumerate(items, start=1):
        if max_runs and reviewed >= max_runs:
            break
        if resume and (run_id, tgt) in done_keys:
            skipped += 1
            continue

        sweep = orig.get("sweep", {})
        res = orig.get("result", orig)

        out_rec: Dict[str, Any] = {
            "run_id": run_id,
            "target": tgt,
            "sweep": sweep,
            "pair": res.get("pair", sweep.get("pair")),
            "template_id": res.get("template_id", sweep.get("template_id")),
            "word_limit": res.get("word_limit", sweep.get("word_limit")),
            "temp_dream": sweep.get("temp_dream"),
            "seed_dream": sweep.get("seed_dream"),
            "prompt": prompt,
            "text": text,
            "meta": {"ts_utc": utc_ts(), "status": "init"},
            "reviews": {},
        }

        try:
            if model_choice in ("gpt-5.2", "gpt-5-mini"):
                if not openai_reviewer:
                    raise RuntimeError("OPENAI_API_KEY missing or OpenAIReviewer not initialized.")
                if verbose:
                    print(f"[review:{tgt}] run_id={run_id} -> OpenAI({openai_reviewer.model}) ...")

                review_obj = openai_reviewer.review(prompt, tgt, text)

                # JSON parse失敗の場合
                if isinstance(review_obj, dict) and review_obj.get("_error") == "json_parse_failed":
                    out_rec["meta"]["status"] = "error"
                    out_rec["meta"]["error_type"] = "OpenAIJSONParseError"
                    out_rec["meta"]["error"] = "OpenAI JSON parse failed"
                    out_rec["meta"]["debug"] = {
                        "raw_model_output": review_obj.get("_raw_model_output", ""),
                        "raw_http_response": review_obj.get("_raw_http_response", ""),
                        "raw_response_json": review_obj.get("_raw_response_json", ""),
                    }
                    failed += 1
                    jsonl_append(out_path, out_rec)
                    time.sleep(max(0.0, sleep_s))
                    continue

                out_rec["reviews"]["openai"] = {
                    "model": openai_reviewer.model,
                    **review_obj,
                }

            elif model_choice == "gemini":
                if not gemini_reviewer:
                    raise RuntimeError("GEMINI_API_KEY missing or GeminiReviewer not initialized.")
                if verbose:
                    print(f"[review:{tgt}] run_id={run_id} -> Gemini({gemini_reviewer.model}) ...")

                review_obj, debug_meta = gemini_reviewer.review(prompt, tgt, text)
                if "parse_error" in debug_meta:
                    raise RuntimeError(f"Gemini JSON parse failed: {debug_meta.get('parse_error')}")

                out_rec["reviews"]["gemini"] = {"model": gemini_reviewer.model, **review_obj}

            else:
                raise ValueError(f"Unknown --model: {model_choice}")

            out_rec["meta"]["status"] = "ok"
            reviewed += 1

        except Exception as e:
            out_rec["meta"]["status"] = "error"
            out_rec["meta"]["error_type"] = type(e).__name__
            out_rec["meta"]["error"] = str(e)
            failed += 1

        jsonl_append(out_path, out_rec)
        time.sleep(max(0.0, sleep_s))

        if (reviewed + failed) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[review:{tgt}] processed={reviewed+failed} ok={reviewed} failed={failed} skipped={skipped} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[review:{target}] done. ok={reviewed} failed={failed} skipped={skipped} elapsed={elapsed:.1f}s")
    return reviewed, failed, skipped, elapsed

# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="External LLM review for ReMIND (single target or --all=three outputs).")
    ap.add_argument("jsonl", type=str, help="path to remind_sweep_*.jsonl")
    ap.add_argument("--out", type=str, default="", help="(single target only) output JSONL path; ignored with --all")

    ap.add_argument(
        "--model",
        choices=["gpt-5-mini", "gpt-5.2", "gemini"],
        default="gpt-5-mini",
        help="review model (default: gpt-5-mini)",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for OpenAI best-effort determinism (default: 0)")

    ap.add_argument("--openai-model", type=str, default="", help="override OpenAI model (advanced)")
    ap.add_argument("--gemini-model", type=str, default="gemini-3-flash-preview", help="Gemini model name")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep seconds between API calls")
    ap.add_argument("--max-runs", type=int, default=0, help="limit number of reviewed items per target (0=all)")
    ap.add_argument("--resume", action="store_true", help="skip (run_id,target) already present in output JSONL")
    ap.add_argument("--min-temp-dream", type=float, default=None, metavar="T",
                    help="skip records with temp_dream < T (e.g. 1.0 to skip 0/0.3/0.6)")
    ap.add_argument("--verbose", action="store_true", help="print progress for each run")

    ap.add_argument("--wake", action="store_true", help="review wakeout only")
    ap.add_argument("--dream", action="store_true", help="review dreamout only")
    ap.add_argument("--rewake", action="store_true", help="review rewakeout only (default)")
    ap.add_argument("--all", action="store_true", help="review wake+dream+rewake (writes three JSONLs)")

    args = ap.parse_args()

    in_path = Path(args.jsonl).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    targets = choose_targets(args)

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")).strip()

    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    gemini_base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").strip()

    openai_reviewer: Optional[OpenAIReviewer] = None
    gemini_reviewer: Optional[GeminiReviewer] = None

    if args.model in ("gpt-5-mini", "gpt-5.2"):
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        model_name = args.openai_model.strip() or args.model
        openai_reviewer = OpenAIReviewer(
            api_key=openai_key,
            model=model_name,
            base_url=openai_base,
            seed=int(args.seed),
        )

    if args.model == "gemini":
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        gemini_reviewer = GeminiReviewer(api_key=gemini_key, model=args.gemini_model, base_url=gemini_base)

    print(f"[review] input  : {in_path}")
    print(f"[review] targets: {targets}")
    print(f"[review] model  : {args.model}")
    if openai_reviewer:
        print(f"[review] openai : model={openai_reviewer.model} base={openai_reviewer.base_url} seed={openai_reviewer.seed}")
    if gemini_reviewer:
        print(f"[review] gemini : model={gemini_reviewer.model} base={gemini_reviewer.base_url}")

    records = load_jsonl(in_path)
    print(f"[review] loaded records: {len(records)}")

    # --- filename model tag ---
    chosen_openai_model = (args.openai_model.strip() or args.model) if args.model in ("gpt-5-mini", "gpt-5.2") else ""
    chosen_gemini_model = args.gemini_model.strip() if args.model == "gemini" else ""
    tag = model_tag(args.model, openai_model=chosen_openai_model, gemini_model=chosen_gemini_model)

    if args.all:
        out_paths = {t: default_out_path(in_path, t, tag=tag) for t in targets}
    else:
        if len(targets) != 1:
            raise RuntimeError("Internal error: non-all mode must have exactly one target.")
        target = targets[0]
        out_paths = {target: (Path(args.out).expanduser() if args.out else default_out_path(in_path, target, tag=tag))}

    for p in out_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    totals_ok = totals_failed = totals_skipped = 0
    t0 = time.time()

    for target in targets:
        out_path = out_paths[target]
        ok, failed, skipped, _elapsed = run_review_for_target(
            records=records,
            target=target,
            out_path=out_path,
            model_choice=args.model,
            openai_reviewer=openai_reviewer,
            gemini_reviewer=gemini_reviewer,
            sleep_s=float(args.sleep),
            max_runs=int(args.max_runs),
            resume=bool(args.resume),
            verbose=bool(args.verbose),
            min_temp_dream=args.min_temp_dream,
        )
        totals_ok += ok
        totals_failed += failed
        totals_skipped += skipped

    elapsed = time.time() - t0
    print(f"[review] done(all_targets={args.all}). ok={totals_ok} failed={totals_failed} skipped={totals_skipped} elapsed={elapsed:.1f}s")

if __name__ == "__main__":
    main()