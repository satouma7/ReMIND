# ensure_tmux.py: Ensure that selected LLM servers are running in tmux
# Usage: 
#   Stopping all unused llms:python ensure_tmux.py --stop-unused --llms
from __future__ import annotations
import subprocess
import time
from typing import Iterable, Optional
import requests
from config import LLM

MAX_LLM_COUNT = 3
HEAVY_PAIRS_LIMIT = [
    ({"oss120b", "llama70b"}, 2),  # このペアが含まれるなら最大2つまで
]
DEFAULT_LLMS = ["oss120b", "gemma4_31b"]

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def tmux_exists(session: str) -> bool:
    cp = _run(["tmux", "has-session", "-t", session])
    return cp.returncode == 0

def tmux_kill(session: str) -> None:
    _run(["tmux", "kill-session", "-t", session])

_OLLAMA_CUDA12_LIB = "/usr/local/lib/ollama/cuda_v12"

def tmux_start(session: str, model_path: str, port: int, n_gpu_layers: int = -1, chat_format: str | None = None, n_ctx: int | None = None) -> None:
    import os
    cmd = (
        f"/home/satouma/miniconda3/envs/py311/bin/python -m llama_cpp.server "
        f"--model {model_path} "
        f"--n_gpu_layers {n_gpu_layers} "
        f"--host 0.0.0.0 "
        f"--port {port} "
    )
    if chat_format:
        cmd += f"--chat_format {chat_format} "
    if n_ctx is not None:
        cmd += f"--n_ctx {n_ctx} "
    if os.path.isdir(_OLLAMA_CUDA12_LIB):
        cmd = f"export LD_LIBRARY_PATH={_OLLAMA_CUDA12_LIB}:$LD_LIBRARY_PATH && " + cmd
    _run(["tmux", "new-session", "-d", "-s", session, cmd])

def wait_server(url: str, timeout_sec: int = 180, interval_sec: float = 1.0) -> bool:
    base = url.rstrip("/")
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

def _guard_required(required: list[str]) -> None:
    req = set(required)
    # 通常の最大数制限
    if len(req) > MAX_LLM_COUNT:
        raise RuntimeError(
            f"Too many LLMs requested: {sorted(req)} (max={MAX_LLM_COUNT})"
        )

    # 条件付き制限
    for pair, limit in HEAVY_PAIRS_LIMIT:
        if pair.issubset(req) and len(req) > limit:
            raise RuntimeError(
                f"Combination {sorted(pair)} is heavy. "
                f"Allowed up to {limit} LLMs, got {len(req)}: {sorted(req)}"
            )

def tmux_list_sessions() -> set[str]:
    cp = _run(["tmux", "list-sessions", "-F", "#{session_name}"])
    if cp.returncode != 0:
        return set()
    return {line.strip() for line in cp.stdout.splitlines() if line.strip()}

def running_llms() -> set[str]:
    sess_names = tmux_list_sessions()
    out = set()
    for name, cfg in LLM.items():
        if cfg["tmux"] in sess_names:
            out.add(name)
    return out

def ensure_tmux(
    required: Optional[Iterable[str]] = None,
    *,
    n_gpu_layers: int = -1,
    stop_unused: bool = False,
    n_ctx: int | None = None,
) -> None:
    if required is None:
        required = DEFAULT_LLMS
    required = list(required)

    for name in required:
        if name not in LLM:
            raise KeyError(f"Unknown LLM key: {name}. Available: {list(LLM.keys())}")

    _guard_required(required)

    req = set(required)
    run = running_llms()

    to_stop  = sorted(run - req)
    to_start = sorted(req - run)

    print(f"[ensure_tmux] running={sorted(run)} required={sorted(req)}")
    print(f"[ensure_tmux] start={to_start} stop={to_stop} stop_unused={stop_unused}")

    # stop phase
    if stop_unused:
        for name in to_stop:
            sess = LLM[name]["tmux"]
            print(f"[ensure_tmux] stopping: {name} (session={sess})")
            tmux_kill(sess)

    # start phase (only missing)
    for name in to_start:
        cfg = LLM[name]
        sess = cfg["tmux"]
        port = int(cfg["port"])
        model = cfg["model"]
        chat_format = cfg.get("chat_format")

        print(f"[ensure_tmux] starting: {name} (session={sess}, port={port})")
        tmux_start(sess, model, port, n_gpu_layers=n_gpu_layers, chat_format=chat_format, n_ctx=n_ctx)

    # wait phase (ensure all required are responsive)
    for name in required:
        cfg = LLM[name]
        port = int(cfg["port"])
        url = cfg["url"]
        if not wait_server(url, timeout_sec=180):
            raise RuntimeError(f"{name} server is not responding on :{port} ({url})")
        
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ensure LLM tmux servers are running.")
    ap.add_argument(
        "--llms",
        nargs="*",
        default=None,
        help=f"LLM keys to start (default: DEFAULT_LLMS (oss120b, gemma4_31b)). Available: {list(LLM.keys())}",
    )
    ap.add_argument(
        "--stop-unused",
        action="store_true",
        help="Kill tmux sessions not in --llms (DANGEROUS)",
    )
    ap.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="n_gpu_layers passed to llama_cpp.server (default: -1)",
    )
    ap.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help="Context length passed to llama_cpp.server (default: model default)",
    )
    args = ap.parse_args()

    ensure_tmux(
        required=args.llms,
        stop_unused=args.stop_unused,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
    )