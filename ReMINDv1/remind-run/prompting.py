# prompting.py: Prompt pre-processing and post-processing utilities for ReMIND v1.0
# - gpt-oss style: wraps system/developer/user turns with special tokens
# - gemma style: uses <start_of_turn> ... <end_of_turn> format
# - post-processing helpers to extract final answers or JSON spans
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

def extract_final_answer(raw_text: str) -> str:
    key = "<|channel|>final<|message|>"
    text = raw_text.split(key)[-1] if key in raw_text else raw_text

    for stop_tok in ("<|return|>", "<|end|>", "</s>", "<|call|>"):
        if stop_tok in text:
            text = text.split(stop_tok)[0]
    return text.strip()

def extract_json(raw: str) -> str:
    """
    Robust JSON extraction for judge outputs.
    """
    raw = raw.strip()
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        return raw[i:j+1].strip()
    return raw