# review_violin_rewake_wake.py
# Compare review scores (alignment/coherence/novelty/sum_score) between
# rewake (dream-path final proposal) and rewake_wake (wake-path final
# proposal, via `python review.py ... --rewake-wake`).
# Answers: does the Dream (high-temp) step add novelty over a Wake-only
# baseline, as judged by the external reviewer?
# Kept separate from review_violin.py: that script's TARGET_ORDER/pair
# comparisons are hardcoded to wake/dream/rewake and back existing paper
# figures, so it is not touched here. Only its pure helper functions are
# reused.
# Input:
#   logs/remind_review_XXXX_rewake<tag>.jsonl
#   -> auto-detect sibling: logs/remind_review_XXXX_rewake_wake<tag>.jsonl
# Output:
#   reports/violin_rewake_wake_{metric}_{XXXX}_All.png
# Usage:
#   python review_violin_rewake_wake.py remind_review_qoq_core_rewake_gpt5.2.jsonl
#   python review_violin_rewake_wake.py remind_review_qoq_core_rewake_gpt5.2.jsonl --separate
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from review_violin import (
    load_jsonl,
    theme_from_pair,
    pick_review_blob,
    cliffs_delta,
    sibling_path,
    resolve_input_path,
    infer_xxxx_from_filename,
)

METRICS = ["alignment", "coherence", "novelty", "sum_score"]
TARGET_ORDER = ["rewake", "rewake_wake"]


def build_dataframe(paths: Dict[str, Path], reviewer: str) -> pd.DataFrame:
    records = []
    for tgt, p in paths.items():
        if not p.exists():
            print(f"[warn] missing file for target={tgt}: {p}")
            continue
        for obj in load_jsonl(p):
            run_id = obj.get("run_id")
            if not isinstance(run_id, int):
                continue
            blob = pick_review_blob(obj.get("reviews") or {}, reviewer)
            if blob is None:
                continue
            a, c, n = blob.get("alignment"), blob.get("coherence"), blob.get("novelty")
            if any(x is None for x in [a, c, n]):
                continue
            try:
                a, c, n = int(a), int(c), int(n)
            except Exception:
                continue
            temp_dream = obj.get("temp_dream")
            if temp_dream is None:
                temp_dream = (obj.get("sweep") or {}).get("temp_dream")
            try:
                temp_dream = float(temp_dream)
            except (TypeError, ValueError):
                temp_dream = float("nan")
            records.append({
                "run_id": run_id,
                "theme": theme_from_pair(obj),
                "target": tgt,
                "temp_dream": temp_dream,
                "alignment": a,
                "coherence": c,
                "novelty": n,
                "sum_score": a + c + n,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df["target"] = pd.Categorical(df["target"], categories=TARGET_ORDER, ordered=True)
    return df


def mann_whitney_report(df: pd.DataFrame, scope_name: str) -> None:
    print(f"\n[stats] Mann-Whitney U (two-sided) | scope={scope_name}")
    for metric in METRICS:
        xa = df.loc[df["target"] == "rewake", metric].dropna().astype(float).to_numpy()
        xb = df.loc[df["target"] == "rewake_wake", metric].dropna().astype(float).to_numpy()
        if len(xa) < 2 or len(xb) < 2:
            print(f"  metric={metric}: nA={len(xa)} nB={len(xb)} -> skip (too few)")
            continue
        res = mannwhitneyu(xa, xb, alternative="two-sided", method="auto")
        d = cliffs_delta(xa, xb)
        print(
            f"  metric={metric}: U={res.statistic:.1f}, p={res.pvalue:.3e}, "
            f"Cliff's delta={d:+.3f}, "
            f"rewake(n={len(xa)}, med={np.median(xa):.2f}) | "
            f"rewake_wake(n={len(xb)}, med={np.median(xb):.2f})"
        )


def violin_plot(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data, labels = [], []
    for tgt in TARGET_ORDER:
        vals = df.loc[df["target"] == tgt, metric].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(tgt)

    if not data:
        print(f"[violin] skip (no data): {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)
    parts = ax.violinplot(
        dataset=data,
        positions=np.arange(1, len(data) + 1),
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2.4)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(loc=i, scale=0.06, size=len(vals))
        ax.scatter(x, vals, s=18, alpha=0.6)

    means = [float(np.mean(v)) for v in data]
    for i, m in enumerate(means, start=1):
        if np.isfinite(m):
            ax.scatter([i], [m], marker="x", s=110, linewidths=2.2, color="black", zorder=5)

    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_ylim((0.5, 15.5) if metric == "sum_score" else (0.5, 5.5))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Violin plots comparing rewake (dream-path) vs rewake_wake (wake-path) review scores."
    )
    ap.add_argument("rewake_jsonl", help="input file: remind_review_XXXX_rewake<tag>.jsonl (default folder: logs/)")
    ap.add_argument("--separate", action="store_true", help="output per-theme violins instead of All-only")
    ap.add_argument("--reviewer", choices=["openai", "gemini", "auto"], default="openai",
                    help="which reviewer scores to use (default: openai)")
    ap.add_argument("--reports", default="reports", help="output folder (default: reports/)")
    ap.add_argument("--all-temps", action="store_true",
                    help="include all temp_dream values (default: only temp_dream >= 1.0)")
    args = ap.parse_args()

    rewake_path = resolve_input_path(args.rewake_jsonl)
    xxxx = infer_xxxx_from_filename(rewake_path)

    paths = {
        "rewake": rewake_path,
        "rewake_wake": sibling_path(rewake_path, "rewake_wake"),
    }
    print(f"[violin] input(rewake)    : {rewake_path}")
    print(f"[violin] auto rewake_wake : {paths['rewake_wake']}")
    print(f"[violin] reviewer         : {args.reviewer}")

    df = build_dataframe(paths, reviewer=args.reviewer)
    if df.empty:
        raise RuntimeError("No valid review rows found. Check files and reviewer key (openai/gemini).")

    if not args.all_temps:
        df = df[df["temp_dream"] >= 1.0].copy()
        print(f"[violin] temp filter      : temp_dream >= 1.0 (n={len(df)})")
    else:
        print(f"[violin] temp filter      : none (all temps, n={len(df)})")

    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scope_df = df.copy()
    mann_whitney_report(scope_df, scope_name="All")
    for metric in METRICS:
        out = reports_dir / f"violin_rewake_wake_{metric}_{xxxx}_All.png"
        violin_plot(scope_df, metric=metric, title=f"{metric}: rewake vs rewake_wake ({xxxx})", out_path=out)
        print(f"[violin] saved: {out}")

    if args.separate:
        themes = sorted([t for t in df["theme"].dropna().unique().tolist() if isinstance(t, str)])
        for theme in themes:
            sub = df[df["theme"] == theme].copy()
            if sub.empty:
                continue
            mann_whitney_report(sub, scope_name=theme)
            for metric in METRICS:
                out = reports_dir / f"violin_rewake_wake_{metric}_{xxxx}_{theme}.png"
                violin_plot(
                    sub, metric=metric,
                    title=f"{metric}: rewake vs rewake_wake ({xxxx} | {theme})",
                    out_path=out,
                )
                print(f"[violin] saved: {out}")

    print("[violin] done.")


if __name__ == "__main__":
    main()
