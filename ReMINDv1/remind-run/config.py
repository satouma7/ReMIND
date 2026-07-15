# Configuration file used by ensure_tmux and the wake/dream/judge modules
LLM = {
    "oss120b": {
        "url": "http://localhost:8000/v1/completions",
        "model": "/home/satouma/llm/ReMIND-run/models/gpt-oss-120b-MXFP4-00001-of-00002.gguf",
        "tmux": "oss120b",
        "port": 8000,
    },
    "gemma27b": {
        "url": "http://localhost:8001/v1/completions",
        "model": "/home/satouma/llm/ReMIND-run/models/gemma-3-27b-it-Q4_K_M.gguf",
        "tmux": "gemma27b",
        "port": 8001,
    },
}