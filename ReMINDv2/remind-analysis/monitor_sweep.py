#!/usr/bin/env python3
"""
monitor_sweep.py
Monitors spark1/2 sweep jobs and runs the full pipeline for each completed cond.
Pipeline: sweep done -> rename (remove timestamp) -> rsync to Mac
          -> review --all -> select_top -> evaluate -> violin_hitemp
"""
from __future__ import annotations
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path.home() / "Python/LLM/ReMIND-analysis"
LOGS_DIR   = SCRIPT_DIR / "logs"
REPORTS_DIR = SCRIPT_DIR / "Reports"

# spark1: qoq -> qqo -> qnq  (session: sweep_q1)
# spark2: qgq -> qqg -> qqn  (session: sweep_q2)
PLAN = [
    ("spark1", "qoo", None, "sweep_qoo"),
]

THEMES = ["time_space", "aperiodic_craft", "periodic_tarot"]

processed: set[str] = set()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def ssh(host: str, cmd: str) -> tuple[str, int]:
    r = subprocess.run(["ssh", host, cmd], capture_output=True, text=True)
    return r.stdout.strip(), r.returncode


def get_sweep_file(host: str, cond: str) -> str | None:
    out, _ = ssh(host, f"ls /home/satouma/llm/ReMIND-run/logs/remind_sweep_{cond}_*.jsonl 2>/dev/null | head -1")
    return out or None


def file_size_on_spark(host: str, filepath: str) -> int:
    out, _ = ssh(host, f"wc -c < '{filepath}' 2>/dev/null || echo 0")
    try:
        return int(out.strip())
    except ValueError:
        return 0


def session_alive(host: str, session: str) -> bool:
    _, rc = ssh(host, f"tmux has-session -t {session} 2>/dev/null")
    return rc == 0


def run(cmd: str) -> int:
    log(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR)
    return r.returncode


def is_complete(host: str, cond: str, next_cond: str | None, session: str) -> bool:
    sweep_file = get_sweep_file(host, cond)
    if not sweep_file:
        return False
    if next_cond:
        nf = get_sweep_file(host, next_cond)
        return bool(nf) and file_size_on_spark(host, nf) > 200
    else:
        return not session_alive(host, session)


def process(host: str, cond: str) -> None:
    log(f"=== Processing {cond} from {host} ===")

    # 1. Remove timestamp and rsync
    sweep_file = get_sweep_file(host, cond)
    new_name = f"remind_sweep_{cond}_core.jsonl"
    if sweep_file and Path(sweep_file).name != new_name:
        ssh(host, f"mv /home/satouma/llm/ReMIND-run/logs/'{Path(sweep_file).name}' /home/satouma/llm/ReMIND-run/logs/{new_name}")
    run(f"rsync -av {host}:/home/satouma/llm/ReMIND-run/logs/{new_name} {LOGS_DIR}/")

    sweep_local = LOGS_DIR / new_name
    if not sweep_local.exists():
        log(f"  [warn] {sweep_local} not found after rsync, skipping")
        return

    # 2. review --all (wake + dream + rewake)
    run(f"python review.py {sweep_local} --all --resume")

    # 3. Setup directory structure
    cond_dir = REPORTS_DIR / cond
    for sub in ["1_similarity_output", "2_review_output", "3_select_output", "4_evaluate_output"]:
        (cond_dir / sub).mkdir(parents=True, exist_ok=True)

    # 4. select_top -> move to 3_select_output
    rewake_review = LOGS_DIR / f"remind_review_{cond}_core_rewake_gpt5mini.jsonl"
    cond_core = f"{cond}_core"  # select_top infer_cond returns e.g. "qoq_core"
    if rewake_review.exists():
        run(f"python select_top.py {rewake_review} --reports /tmp/remind_select_{cond}")
        src_dir = Path(f"/tmp/remind_select_{cond}") / cond_core
        dst_dir = cond_dir / "3_select_output"
        if src_dir.exists():
            for f in src_dir.glob("top_rewakeout_*.csv"):
                shutil.move(str(f), str(dst_dir / f.name))
                log(f"  moved {f.name} -> 3_select_output/")

    # 5. evaluate -> move to 4_evaluate_output
    select_dir = cond_dir / "3_select_output"
    eval_dir   = cond_dir / "4_evaluate_output"
    for theme in THEMES:
        top_csv = select_dir / f"top_rewakeout_{cond_core}_{theme}.csv"
        if top_csv.exists():
            run(f"python evaluate.py {top_csv}")
            # move top10_*.csv from Reports/ root to 4_evaluate_output/
            for f in REPORTS_DIR.glob(f"top10_{cond_core}_{theme}*.csv"):
                shutil.move(str(f), str(eval_dir / f.name))
                log(f"  moved {f.name} -> 4_evaluate_output/")
            for f in (SCRIPT_DIR / "reports").glob(f"top10_{cond}_{theme}*.json"):
                shutil.move(str(f), str(eval_dir / f.name))

    # 6. violin_hitemp
    if rewake_review.exists():
        hitemp_dir = cond_dir / "2_review_output" / "violin_hitemp"
        run(f"python review_violin.py {rewake_review} --separate --reports {hitemp_dir}")

    processed.add(cond)
    log(f"=== Done: {cond} ===\n")


def send_mail(subject: str, body: str) -> None:
    script = f'''tell application "Mail"
    set msg to make new outgoing message
    set subject of msg to "{subject}"
    set content of msg to "{body}"
    set message signature of msg to missing value
    tell msg
        make new to recipient at end of to recipients with properties {{name:"satouma", address:"satouma@mac.com"}}
    end tell
    send msg
end tell'''
    subprocess.run(["osascript", "-e", script])


def main() -> None:
    log("Sweep monitor started.")
    log(f"Watching: {[p[1] for p in PLAN]}")

    while True:
        for host, cond, next_cond, session in PLAN:
            if cond in processed:
                continue
            if is_complete(host, cond, next_cond, session):
                try:
                    process(host, cond)
                except Exception as e:
                    log(f"[ERROR] {cond}: {e}")

        if len(processed) == len(PLAN):
            log("All 6 conditions complete!")
            subprocess.run(["pkill", "caffeinate"])
            log("caffeinate terminated.")
            send_mail(
                "✅ ReMIND qoo pipeline 完了",
                "qoo の sweep → review → select_top → evaluate → violin_hitemp が完了しました。caffeinate も終了しました。"
            )
            break

        time.sleep(300)  # 5分ごとにチェック


if __name__ == "__main__":
    main()
