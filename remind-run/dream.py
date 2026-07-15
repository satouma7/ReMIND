# dream.py: Dream module for ReMIND v1.0
# Performs high-temperature stochastic generation to induce semantic drift.
# Unlike wake, this module intentionally explores unstable, novel, and
# non-canonical associations to probe the model’s creative/associative regime.
# - Uses an OpenAI-compatible /v1/completions endpoint (base URL is in config.py)
# - Builds model-specific prompts (gemma vs gpt-oss) via prompting.py
# - For gpt-oss family (oss120b/oss20b), post-processes the raw text
#   to extract the final answer span for consistency
from __future__ import annotations
import requests
from config import LLM
from prompting import build_gemma_prompt, build_gemma4_prompt, build_qwen_prompt, build_nemo_prompt, build_gpt_oss_prompt, build_base_prompt, extract_final_answer, postprocess_text

developer_prompt = (
    "Do NOT show your reasoning.\n"
    "Do NOT describe the task.\n"
    "Respond ONLY with the final answer."
)

def _is_gpt_oss_family(llm: str) -> bool:
    """
    Treat any 'gpt-oss' style local models as Harmony-formatted.
    Current keys: oss120b, oss20b
    """
    return llm in {"oss120b", "oss20b"}

def _is_gemma4_family(llm: str) -> bool:
    return llm in {"gemma4_31b", "gemma4_26b"}

def _is_chatml_family(llm: str) -> bool:
    return llm in {"qwen35_27b", "qwen35_35b"}

def _is_nemo_family(llm: str) -> bool:
    return llm in {"nemo30b"}

def _is_base_family(llm: str) -> bool:
    return "_base" in llm

def dream(
    prompt: str,
    *,
    llm: str = "gemma4_31b",
    max_tokens: int = 150,
    temperature: float = 3.0,
    seed: int = 0,
) -> str:
    """Run a single completion for the dream module (high-temperature exploration)."""
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
        full_prompt = build_nemo_prompt(developer_prompt, prompt)
    elif _is_chatml_family(llm):
        full_prompt = build_qwen_prompt(developer_prompt, prompt)
    elif _is_base_family(llm):
        full_prompt = build_base_prompt(prompt)
    else:
        full_prompt = build_gemma_prompt(developer_prompt, prompt)

    dream_request = {
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if _is_chatml_family(llm) or _is_nemo_family(llm):
        dream_request["stop"] = ["<|im_end|>"]

    res = requests.post(url, json=dream_request, timeout=120)
    res.raise_for_status()
    data = res.json()

    raw = (data.get("choices") or [{}])[0].get("text", "")
    raw = "" if raw is None else str(raw)
    raw = postprocess_text(llm, raw)

    if _is_gpt_oss_family(llm):
        raw = extract_final_answer(raw)
    return raw.strip()