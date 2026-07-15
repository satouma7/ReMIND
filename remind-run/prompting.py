# prompting.py: Prompt pre-processing and post-processing utilities for ReMIND v1.0
# - gpt-oss style: wraps system/developer/user turns with special tokens
# - gemma style: uses <start_of_turn> ... <end_of_turn> format
# - post-processing helpers to extract final answers or JSON spans

import re

# Tokens that indicate the model started emitting chat-template / turn markers.
# If these appear in OUTPUT, it usually means "next turn generation" leakage.
_CUT_MARKERS = [
    "<start_of_turn>",
    "<end_of-turn>",          # gemma/llama chat template
    "<end_of_turn>",            # gemma/llama chat template
    "<|end_of_turn|>",          # some chat templates
    "<|eot_id|>",               # some llama templates
    "<turn|>",                  # gemma4 chat template
    "<|im_end|>",               # qwen/chatml chat template
]

def build_gpt_oss_prompt(developer_prompt: str, user_prompt: str) -> str:
    system_msg = (
        "<|start|>system<|message|>"
        "You are a concise and helpful assistant."
        "<|end|>"
    )
    developer_msg = (
        "<|start|>developer<|message|>"
        + developer_prompt.strip()
        + "<|end|>"
    )
    user_msg = (
        "<|start|>user<|message|>"
        + user_prompt.strip()
        + "<|end|>"
    )
    assistant_prefix = "<|start|>assistant<|channel|>final<|message|>"
    return system_msg + developer_msg + user_msg + assistant_prefix

def build_gemma_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<start_of_turn>system\n" + system_prompt.strip() + "\n<end_of_turn>\n"
        "<start_of_turn>user\n" + user_prompt.strip() + "\n<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def build_qwen_prompt(system_prompt: str, user_prompt: str) -> str:
    # Pre-fill an empty <think> block to suppress chain-of-thought output
    # (Qwen3.5 is a thinking model; without this, reasoning fills max_tokens)
    return (
        "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )

def build_nemo_prompt(system_prompt: str, user_prompt: str) -> str:
    # Nemotron requires <think></think> without newlines (unlike Qwen3.5)
    return (
        "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )

def build_gemma4_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<|turn>system\n" + system_prompt.strip() + "<turn|>\n"
        "<|turn>user\n" + user_prompt.strip() + "<turn|>\n"
        "<|turn>model\n"
        "<|channel>thought\n<channel|>"
    )

def extract_final_answer(raw_text: str) -> str:
    key = "<|channel|>final<|message|>"
    text = raw_text.split(key)[-1] if key in raw_text else raw_text

    for stop_tok in ("<|return|>", "<|end|>", "</s>", "<|call|>"):
        if stop_tok in text:
            text = text.split(stop_tok)[0]
    return text.strip()

def extract_json(raw: str) -> str:
    raw = raw.strip()
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        return raw[i:j+1].strip()
    return raw

def _truncate_on_markers(s: str) -> str:
    """Cut output at the earliest occurrence of any marker."""
    earliest = None
    for m in _CUT_MARKERS:
        idx = s.find(m)
        if idx != -1:
            earliest = idx if earliest is None else min(earliest, idx)
    return s if earliest is None else s[:earliest]

def postprocess_text(llm: str, text: str) -> str:
    """
    Post-process raw completion text.
    Key idea: if chat-template markers appear in the OUTPUT, truncate there
    (do not merely delete markers), because the tail is usually garbage / duplicated.
    """
    if text is None:
        return ""
    s = str(text)

    # 1) Truncate on leaked chat-template markers (fixes llama70b duplication)
    s = _truncate_on_markers(s)

    # 2) Normalize newlines/spaces a bit
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()

def build_base_prompt(user_prompt: str) -> str:
    """Plain text continuation for base (non-instruction-tuned) models."""
    return user_prompt.strip() + "\n\n"
