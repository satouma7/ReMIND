# wake.py: Wake module for ReMIND v1.0
# Runs a low-temperature, single-shot completion as a stable semantic baseline.
# - Uses an OpenAI-compatible /v1/completions endpoint specified in config.py
# - Builds model-specific prompts (gemma vs gpt-oss) via prompting.py
# - For oss120b, post-processes the raw text to extract the final answer span
from __future__ import annotations
import requests
from config import LLM
from prompting import build_gemma_prompt, build_gpt_oss_prompt, extract_final_answer

developer_prompt = (
    "Do NOT show your reasoning.\n"
    "Do NOT describe the task.\n"
    "Respond ONLY with the final answer."
)

def wake(
    prompt: str,
    *,
    llm: str = "oss120b",
    max_tokens: int = 150,
    temperature: float = 0.6,
    seed: int = 1,
) -> str:
    """Run a single completion for the wake module (low-temperature baseline)."""
    llm_cfg = LLM[llm]
    url = llm_cfg["url"]

    if llm == "oss120b":
        full_prompt = build_gpt_oss_prompt(developer_prompt, prompt)
    else:
        full_prompt = build_gemma_prompt(developer_prompt, prompt)

    wake_request = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed, 
    }

    res = requests.post(url, json=wake_request, timeout=120)
    res.raise_for_status()
    data = res.json()
    raw = data["choices"][0]["text"]

    # oss120b may include extra wrappers; extract the final answer span for consistency.
    if llm == "oss120b":
        return extract_final_answer(raw)
    else:
        return raw.strip()