# judge.py: Judge module for ReMIND v1.0
# Performs structured evaluation of LLM outputs to quantify semantic stability
# and detect the presence of novel conceptual content.
# This module operationalizes the "judge" phase of ReMIND by:
#  - scoring internal consistency on a 1–5 scale
#  - extracting a single sentence that represents the most novel idea (if present)
# The judge enforces strict two-line output format to enable reliable downstream parsing.
# - Uses an OpenAI-compatible /v1/completions endpoint (base URL in config.py)
# - Builds model-specific prompts (gemma vs gpt-oss) via prompting.py
# - For gpt-oss family (oss120b/oss20b), post-processes the raw text
#   to extract the final answer span before parsing
from __future__ import annotations
import requests
import re
from config import LLM
from prompting import build_gemma_prompt, build_gemma4_prompt, build_qwen_prompt, build_nemo_prompt, build_gpt_oss_prompt, extract_final_answer, postprocess_text

_SCORE_RE = re.compile(r"^\s*SCORE:\s*([1-5])\s*$", re.IGNORECASE)
_IDEA_RE  = re.compile(r"^\s*IDEA:\s*(.*)\s*$", re.IGNORECASE)

developer_prompt = """
Do NOT show your reasoning.
Do NOT describe the task.
Respond ONLY with the final answer.
Do not use markdown, bullets, headers, or code fences.

You are evaluating the consistency and novelty of a LLM response.

Consistency score (1–5):
1 = inconsistent
2 = mostly inconsistent
3 = partially consistent
4 = mostly consistent
5 = fully consistent

Novel idea:
If there is a novel or creative idea, extract ONE sentence.
If none exists, write EMPTY.

Return EXACTLY two lines and nothing else:
Line 1 must be exactly in this format: SCORE: <integer 1-5>
Line 2 must be exactly in this format: IDEA: <ONE sentence or EMPTY>
Do not add any extra text before or after these two lines.

Evaluate the following response:
""".strip()

def _is_gpt_oss_family(llm: str) -> bool:
    """Treat gpt-oss style local models as Harmony-formatted."""
    return llm in {"oss120b", "oss20b"}

def _is_gemma4_family(llm: str) -> bool:
    return llm in {"gemma4_31b", "gemma4_26b"}

def _is_chatml_family(llm: str) -> bool:
    return llm in {"qwen35_27b", "qwen35_35b"}

def _is_nemo_family(llm: str) -> bool:
    return llm in {"nemo30b"}

def _strip_begin_end_markers(text: str) -> str:
    """Remove ===BEGIN===/===END=== markers (redundant in ChatML user turn)."""
    text = re.sub(r"^===BEGIN===\n?", "", text)
    text = re.sub(r"\n?===END===$", "", text)
    return text.strip()

def parse_two_line_format(txt: str) -> dict:
    score = None
    idea = None
    lines = (txt or "").splitlines()

    for i, line in enumerate(lines):
        m = _SCORE_RE.match(line)
        if m:
            score = int(m.group(1))
            continue

        m = _IDEA_RE.match(line)
        if m:
            idea = m.group(1).strip()
            # IDEA: の後が空なら次の非空行を拾う（保険）
            if idea == "":
                for j in range(i+1, len(lines)):
                    cand = lines[j].strip()
                    if cand:
                        idea = cand
                        break
            continue

    if score is None or idea is None:
        raise RuntimeError(f"Judge format error (missing SCORE/IDEA)\nRAW:\n{txt}")

    if idea.upper() == "EMPTY":
        idea = ""

    return {"score": score, "idea": idea}

def judge_raw(text: str, *, llm: str, max_tokens: int, temperature: float, seed: int) -> str:
    """
    Same as judge(), but returns the raw LLM output text (two lines expected).
    """
    if llm not in LLM:
        raise KeyError(f"Unknown llm key: {llm} (available: {list(LLM.keys())})")

    llm_cfg = LLM[llm]
    base = llm_cfg["url"].rstrip("/")
    url = f"{base}/v1/completions"

    if _is_gpt_oss_family(llm):
        full_prompt = build_gpt_oss_prompt(developer_prompt, text)
    elif _is_gemma4_family(llm):
        full_prompt = build_gemma4_prompt(developer_prompt, text)
    elif _is_nemo_family(llm):
        full_prompt = build_nemo_prompt(developer_prompt, _strip_begin_end_markers(text))
    elif _is_chatml_family(llm):
        full_prompt = build_qwen_prompt(developer_prompt, _strip_begin_end_markers(text))
    else:
        full_prompt = build_gemma_prompt(developer_prompt, text)

    judge_request = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if _is_chatml_family(llm) or _is_nemo_family(llm):
        judge_request["stop"] = ["<|im_end|>"]

    res = requests.post(url, json=judge_request, timeout=120)
    res.raise_for_status()
    data = res.json()

    raw = (data.get("choices") or [{}])[0].get("text", "")
    return "" if raw is None else str(raw)

def judge(
    prompt: str,
    *,
    llm: str = "oss120b",
    max_tokens: int = 320,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict:
    """Run a single completion for the judge module (scoring + idea extraction)."""
    if llm not in LLM:
        raise KeyError(f"Unknown llm key: {llm} (available: {list(LLM.keys())})")

    llm_cfg = LLM[llm]
    base = llm_cfg["url"].rstrip("/")
    url = f"{base}/v1/completions"

    if _is_gpt_oss_family(llm):
        full_prompt = build_gpt_oss_prompt(developer_prompt, prompt)
    elif _is_gemma4_family(llm):
        full_prompt = build_gemma4_prompt(developer_prompt, prompt)
    elif _is_nemo_family(llm):
        full_prompt = build_nemo_prompt(developer_prompt, _strip_begin_end_markers(prompt))
    elif _is_chatml_family(llm):
        full_prompt = build_qwen_prompt(developer_prompt, _strip_begin_end_markers(prompt))
    else:
        full_prompt = build_gemma_prompt(developer_prompt, prompt)

    judge_request = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if _is_chatml_family(llm) or _is_nemo_family(llm):
        judge_request["stop"] = ["<|im_end|>"]

    res = requests.post(url, json=judge_request, timeout=120)
    res.raise_for_status()
    data = res.json()

    raw = (data.get("choices") or [{}])[0].get("text", "")
    raw = "" if raw is None else str(raw)
    txt = postprocess_text(llm, raw)

    if _is_gpt_oss_family(llm):
        txt = extract_final_answer(txt)

    return parse_two_line_format(txt)