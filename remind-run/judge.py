# judge.py: Evaluation module for ReMIND v1.0
# This module implements the "judge" phase of ReMIND.
# It evaluates LLM-generated text for:
#  - internal consistency (1–5 score)
#  - presence of a novel or creative idea (single quoted sentence)
from __future__ import annotations
import json
import requests
from config import LLM
from prompting import build_gemma_prompt, build_gpt_oss_prompt, extract_final_answer, extract_json

developer_prompt = """
Do NOT show your reasoning. 
Do NOT describe the task. 
Respond ONLY with the final answer.

You are an expert judge evaluating the consistency and novelty of LLM responses in JSON format.
Return only valid JSON. No prose, no code fences.

1. Rate the response’s consistency on a 5-point scale (score):
   1 = Incoherent, contradictory, or nonsensical
   2 = Mostly inconsistent, confusing, or unstable meaning
   3 = Somewhat consistent but with noticeable gaps or shifts
   4 = Mostly consistent, clear, and reasonable with minor flaws
   5 = Fully consistent, coherent, and self-aligned throughout

2. Identify novel ideas:
   - If the response contains any sentence that introduces a novel, original, or creative idea, quote that sentence.
   - If multiple such ideas exist, select the most significant one.
   - If no such idea is found, leave this field empty.

Output only in JSON without extra text:
{
  "score": <integer 1–5>,
  "idea": "<quote the most novel idea, or empty>"
}

Evaluate the following LLM response:
"""

def judge(
    prompt: str,
    *,
    llm: str = "gemma27b",
    max_tokens: int = 200,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict:
    """Run a single completion for the judge module (JSON scoring + idea extraction)."""
    llm_cfg = LLM[llm]
    url = llm_cfg["url"]

    if llm== "oss120b":
        full_prompt = build_gpt_oss_prompt(developer_prompt , prompt)
    else:
        full_prompt = build_gemma_prompt(developer_prompt, prompt)

    judge_request = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed, 
    }

    res = requests.post(url, json=judge_request, timeout=120)
    res.raise_for_status()
    data = res.json()
    txt = data["choices"][0]["text"].strip()

    # oss120b may include extra wrappers; extract the final answer span for consistency.
    if llm == "oss120b":
        txt = extract_final_answer(txt)

    json_text = extract_json(txt)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Judge JSON parse failed: {e}\nRAW:\n{txt}") from e