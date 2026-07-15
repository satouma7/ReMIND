# Ensure that oss120b and gemma27b servers are running on the LLM server
import subprocess
import time
import requests
from config import LLM

URL1= LLM["oss120b"]["url"]
URL2= LLM["gemma27b"]["url"]

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a shell command and return the result without raising exceptions."""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def tmux_exists(session: str) -> bool:
    """Check whether a tmux session with the given name exists."""
    cp = _run(["tmux", "has-session", "-t", session])
    return cp.returncode == 0

def tmux_start(session: str, model_path: str, port: int, n_gpu_layers: int = -1) -> None:
    """Start llama_cpp.server in a new tmux session."""
    cmd = (
        f"python -m llama_cpp.server "
        f"--model {model_path} "
        f"--n_gpu_layers {n_gpu_layers} "
        f"--host 0.0.0.0 "
        f"--port {port}"
    )
    _run(["tmux", "new-session", "-d", "-s", session, cmd])

def wait_server(url: str, timeout_sec: int = 120, interval_sec: float = 1.0) -> bool:
    """Wait until the server responds to HTTP requests."""
    base = url.split("/v1/")[0]
    models_url = f"{base}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get(models_url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval_sec)
    return False

def ensure_tmux() -> None:
    """
    Ensure that oss120b (port 8000) and gemma27b (port 8001) are running in tmux.
    - Start the tmux session if it does not exist
    - Wait until the HTTP API becomes responsive
    """
    oss_session = "oss120b"
    gemma_session = "gemma27b"

    oss_model = "/home/satouma/llm/ReMIND-run/models/gpt-oss-120b-MXFP4-00001-of-00002.gguf"
    gemma_model = "/home/satouma/llm/ReMIND-run/models/gemma-3-27b-it-Q4_K_M.gguf"

    if not tmux_exists(oss_session):
        print("Starting up gpt-oss-120b")
        tmux_start(session=oss_session, model_path=oss_model, port=8000)
    else:
        print("oss120b tmux session already exists")

    if not tmux_exists(gemma_session):
        print("Starting up gemma27b")
        tmux_start(session=gemma_session, model_path=gemma_model, port=8001)
    else:
        print("gemma27b tmux session already exists")

    if not wait_server(URL1, timeout_sec=180):
        raise RuntimeError("oss120b server is not responding on :8000")
    if not wait_server(URL2, timeout_sec=180):
        raise RuntimeError("gemma27b server is not responding on :8001")