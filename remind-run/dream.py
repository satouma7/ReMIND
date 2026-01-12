# dream.py: High-temperature exploratory generation for ReMIND v1.0
# The dream module induces semantic exploration by stochastic generation,
# intentionally prioritizing novelty and deviation over coherence.
from __future__ import annotations
import requests
from config import LLM
from prompting import build_gemma_prompt, build_gpt_oss_prompt, extract_final_answer

developer_prompt = (
    "Do NOT show your reasoning.\n"
    "Do NOT describe the task.\n"
    "Respond ONLY with the final answer."
)

def dream(
    prompt: str,
    *,
    llm: str = "gemma27b",
    max_tokens: int = 150,
    temperature: float = 3,
    seed: int = 0,
) -> str:
    """Run a single completion for the dream module  (high-temperature exploration)."""
    llm_cfg = LLM[llm]
    url = llm_cfg["url"]

    if llm == "oss120b":
        full_prompt = build_gpt_oss_prompt(developer_prompt , prompt)
    else:
        full_prompt = build_gemma_prompt(developer_prompt, prompt)

    dream_request = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed, 
    }

    res = requests.post(url, json=dream_request, timeout=120)
    res.raise_for_status()
    data = res.json()
    raw = data["choices"][0]["text"]

    # oss120b may include extra wrappers; extract the final answer span for consistency.
    if llm == "oss120b":
        return extract_final_answer(raw)
    else:
        return raw.strip()
