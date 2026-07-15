# config.py: Configuration file used by ensure_tmux and the wake/dream/judge modules

MODEL_DIR = "/home/satouma/llm/models"

LLM = {
    # --- Instruction-tuned models (uppercase in LLM_CODE_MAP) ---
    "oss120b": {
        "url": "http://localhost:8000",
        "model": f"{MODEL_DIR}/gpt-oss-120b-MXFP4-00001-of-00002.gguf",
        "tmux": "oss120b",
        "port": 8000,
    },
    "gemma4_31b": {
        "url": "http://localhost:8001",
        "model": f"{MODEL_DIR}/gemma-4-31B-it-Q4_K_M.gguf",
        "tmux": "gemma4_31b",
        "port": 8001,
    },
    "oss20b": {
        "url": "http://localhost:8002",
        "model": f"{MODEL_DIR}/gpt-oss-20b-Q4_K_M.gguf",
        "tmux": "oss20b",
        "port": 8002,
    },
    "llama3_70b": {
        "url": "http://localhost:8003",
        "model": f"{MODEL_DIR}/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "tmux": "llama3_70b",
        "port": 8003,
    },
    "llama4_17b": {
        "url": "http://localhost:8004",
        "model": f"{MODEL_DIR}/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
        "tmux": "llama4_17b",
        "port": 8004,
    },
    "qwen35_27b": {
        "url": "http://localhost:8005",
        "model": f"{MODEL_DIR}/Qwen3.5-27B-Q4_K_M.gguf",
        "tmux": "qwen35_27b",
        "port": 8005,
    },
    "qwen35_35b": {
        "url": "http://localhost:8006",
        "model": f"{MODEL_DIR}/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "tmux": "qwen35_35b",
        "port": 8006,
    },
    "deep8b": {
        "url": "http://localhost:8007",
        "model": f"{MODEL_DIR}/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
        "tmux": "deep8b",
        "port": 8007,
    },
    "deep32b": {
        "url": "http://localhost:8008",
        "model": f"{MODEL_DIR}/deepseek-r1-distill-qwen-32b-q4_k_m.gguf",
        "tmux": "deep32b",
        "port": 8008,
    },
    "nemo30b": {
        "url": "http://localhost:8009",
        "model": f"{MODEL_DIR}/Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
        "tmux": "nemo30b",
        "port": 8009,
    },
    "gemma4_26b": {
        "url": "http://localhost:8010",
        "model": f"{MODEL_DIR}/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "tmux": "gemma4_26b",
        "port": 8010,
    },
    # --- Base models (lowercase in LLM_CODE_MAP) ---
    "qwen8b_base": {
        "url": "http://localhost:8011",
        "model": f"{MODEL_DIR}/Qwen3-8B-Base.Q4_K_M.gguf",
        "tmux": "qwen8b_base",
        "port": 8011,
    },
    "oss20b_base": {
        "url": "http://localhost:8012",
        "model": f"{MODEL_DIR}/gpt-oss-20b-base.Q4_K_M.gguf",
        "tmux": "oss20b_base",
        "port": 8012,
    },
}

# Single-letter codes for CLI --llm arguments
#   Uppercase = large instruction-tuned models
#   Lowercase = base (non-instruction-tuned) models
LLM_CODE_MAP = {
    "O": "oss120b",
    "G": "gemma4_31b",
    "Q": "qwen35_27b",
    "N": "nemo30b",
    "o": "oss20b_base",
    "q": "qwen8b_base",
}
