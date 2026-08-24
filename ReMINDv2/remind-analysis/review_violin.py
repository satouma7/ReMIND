# review_violin.py
# Compare review scores (alignment/coherence/novelty/sum_score) across targets (wake/dream/rewake)
# from ReMIND review JSONL files, and output violin plots.
# Input:
#   logs/remind_review_XXXX_rewake.jsonl
#   -> auto-detect sibling files:
#      logs/remind_review_XXXX_wake.jsonl
#      logs/remind_review_XXXX_dream.jsonl
# Output:
#   reports/violin_{metric}_{XXXX}_All.png              (default)
#   reports/violin_{metric}_{XXXX}_{theme}.png          (--separate)
# Also prints Mann–Whitney U tests + Cliff's delta:
#   wake–dream / wake–rewake / dream–rewake
# Usage:
#   python review_violin.py remind_review_1ooo_rewake.jsonl
#   python review_violin.py remind_review_1ooo_rewake.jsonl --separate
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

METRICS = ["alignment", "coherence", "novelty", "sum_score"]
TARGET_ORDER = ["wake", "dream", "rewake"]

def infer_xxxx_from_filename(p: Path) -> str:
    """
    remind_review_XXXX_rewake.jsonl -> XXXX
    """
    name = p.name
    if name.startswith("remind_review_") and name.endswith(".jsonl"):
        core = name[len("remind_review_") : -len(".jsonl")]
        # strip trailing _rewake/_wake/_dream if present
        for suf in ["_rewake", "_wake", "_dream"]:
            if core.endswith(suf):
                core = core[: -len(suf)]
        return core
    # fallback to stem
    stem = p.stem
    for suf in ["_rewake", "_wake", "_dream"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def theme_from_pair(obj: Dict[str, Any]) -> str:
    """
    Derive theme string from obj["pair"] or obj["sweep"]["pair"].
    Examples:
      ["time","space"] -> "time_space"
      ["aperiodic","craft"] -> "aperiodic_craft"
      ["periodic","tarot"] -> "periodic_tarot"
    """
    pair = obj.get("pair")
    if pair is None:
        pair = (obj.get("sweep") or {}).get("pair")

    if isinstance(pair, list) and all(isinstance(x, str) for x in pair) and len(pair) > 0:
        parts = [x.strip().lower().replace(" ", "_") for x in pair if x.strip()]
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return parts[0] if parts else "unknown"

    if isinstance(pair, str) and pair.strip():
        return pair.strip().lower().replace(" ", "_")

    return "unknown"


def pick_review_blob(reviews: Dict[str, Any], preferred: str) -> Optional[Dict[str, Any]]:
    """
    reviews: {"openai": {...}, "gemini": {...}}
    preferred: "openai" | "gemini" | "auto"
    """
    if not isinstance(reviews, dict) or not reviews:
        return None

    if preferred != "auto":
        blob = reviews.get(preferred)
        return blob if isinstance(blob, dict) else None

    # auto: prioritize openai then gemini then first dict
    for k in ["openai", "gemini"]:
        if isinstance(reviews.get(k), dict):
            return reviews[k]
    for v in reviews.values():
        if isinstance(v, dict):
            return v
    return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def resolve_input_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p

    # default folder: logs/
    alt = Path("logs") / p
    if alt.exists():
        return alt

    raise FileNotFoundError(p)


def sibling_path(rewake_path: Path, target: str) -> Path:
    """
    logs/remind_review_XXXX_rewake.jsonl         -> logs/remind_review_XXXX_{target}.jsonl
    logs/remind_review_XXXX_rewake_gpt5mini.jsonl -> logs/remind_review_XXXX_{target}_gpt5mini.jsonl
    """
    import re
    name = rewake_path.name
    # Handle optional model tag: _rewake{tag}.jsonl -> _{target}{tag}.jsonl
    m = re.match(r'^(.+)_rewake(_.+)?\.jsonl$', name)
    if m:
        prefix, tag = m.group(1), m.group(2) or ""
        return rewake_path.with_name(f"{prefix}_{target}{tag}.jsonl")
    # if user gives without _rewake, try add
    if name.endswith(".jsonl") and "_rewake" not in name:
        stem = rewake_path.stem
        return rewake_path.with_name(stem + f"_{target}.jsonl")
    # fallback: replace last token
    return rewake_path.with_name(rewake_path.stem + f"_{target}.jsonl")


def build_dataframe(paths: Dict[str, Path], reviewer: str) -> pd.DataFrame:
    """
    Returns long-form dataframe:
      columns: run_id, theme, target, metric, score
    """
    records = []
    for tgt, p in paths.items():
        if not p.exists():
            print(f"[warn] missing file for target={tgt}: {p}")
            continue

        rows = load_jsonl(p)
        for obj in rows:
            run_id = obj.get("run_id")
            if not isinstance(run_id, int):
                continue

            theme = theme_from_pair(obj)

            reviews = obj.get("reviews") or {}
            blob = pick_review_blob(reviews, reviewer)
            if blob is None:
                continue

            a = blob.get("alignment")
            c = blob.get("coherence")
            n = blob.get("novelty")
            if any(x is None for x in [a, c, n]):
                continue

            try:
                a = int(a); c = int(c); n = int(n)
            except Exception:
                continue

            ss = a + c + n

            temp_dream = obj.get("temp_dream")
            if temp_dream is None:
                temp_dream = (obj.get("sweep") or {}).get("temp_dream")
            try:
                temp_dream = float(temp_dream)
            except (TypeError, ValueError):
                temp_dream = float("nan")

            records.append({
                "run_id": run_id,
                "theme": theme,
                "target": tgt,
                "temp_dream": temp_dream,
                "alignment": a,
                "coherence": c,
                "novelty": n,
                "sum_score": ss,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    df["target"] = pd.Categorical(df["target"], categories=TARGET_ORDER, ordered=True)
    return df


def cliffs_delta(xa: np.ndarray, xb: np.ndarray) -> float:
    """
    δ = cliffs_delta(from, to): positive = to > from (improvement), i.e.
    (P(X<Y) - P(X>Y)) over all pairs. Matches cliff_ranking.py's convention
    and the manuscript's stated Cliff's delta convention.
    Works well for ordinal/discrete scores with ties.
    """
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    # Broadcasting comparisons: O(n*m). Here n,m ~ 100-500 so it's fine.
    diff = xa[:, None] - xb[None, :]
    n_gt = float(np.sum(diff > 0))
    n_lt = float(np.sum(diff < 0))
    denom = float(xa.size * xb.size)
    return (n_lt - n_gt) / denom


def mann_whitney_report(df: pd.DataFrame, scope_name: str) -> None:
    """
    Print Mann–Whitney U tests + Cliff's delta for each metric:
      wake–dream / wake–rewake / dream–rewake
    """
    pairs = [("wake", "dream"), ("wake", "rewake"), ("dream", "rewake")]
    print(f"\n[stats] Mann–Whitney U (two-sided) | scope={scope_name}")
    for metric in METRICS:
        print(f"  metric={metric}")
        for a, b in pairs:
            xa = df.loc[df["target"] == a, metric].dropna().astype(float).to_numpy()
            xb = df.loc[df["target"] == b, metric].dropna().astype(float).to_numpy()
            if len(xa) < 2 or len(xb) < 2:
                print(f"    {a} vs {b}: nA={len(xa)} nB={len(xb)} -> skip (too few)")
                continue
            res = mannwhitneyu(xa, xb, alternative="two-sided", method="auto")
            med_a = float(np.median(xa)); med_b = float(np.median(xb))
            d = cliffs_delta(xa, xb)
            print(
                f"    {a} vs {b}: U={res.statistic:.1f}, p={res.pvalue:.3e}, "
                f"Cliff's δ={d:+.3f}, "
                f"nA={len(xa)} medA={med_a:.2f} | nB={len(xb)} medB={med_b:.2f}"
            )


def violin_plot(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    """
    Matplotlib-only violin + jitter scatter + median (black) + mean (black 'x').
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    labels = []
    for tgt in TARGET_ORDER:
        vals = df.loc[df["target"] == tgt, metric].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(tgt)

    if not data:
        print(f"[violin] skip (no data): {out_path.name}")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    parts = ax.violinplot(
        dataset=data,
        positions=np.arange(1, len(data) + 1),
        showmeans=False,
        showmedians=True,   # we'll restyle to black
        showextrema=False,
    )

    # Make median bars clearly visible: black and thicker
    if "cmedians" in parts and parts["cmedians"] is not None:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2.4)

    # jittered points (deterministic)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        x = rng.normal(loc=i, scale=0.06, size=len(vals))
        ax.scatter(x, vals, s=18, alpha=0.6)

    # Mean markers: black 'x', slightly larger so it stands out even if overlapping median
    means = [float(np.mean(v)) for v in data]
    for i, m in enumerate(means, start=1):
        if np.isfinite(m):
            ax.scatter([i], [m], marker="x", s=110, linewidths=2.2, color="black", zorder=5)

    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=16)

    if metric == "sum_score":
        ax.set_ylim(0.5, 15.5)
    else:
        ax.set_ylim(0.5, 5.5)

    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Violin plots for ReMIND reviews across targets.")
    ap.add_argument("rewake_jsonl", help="input file: remind_review_XXXX_rewake.jsonl (default folder: logs/)")
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
        "wake": sibling_path(rewake_path, "wake"),
        "dream": sibling_path(rewake_path, "dream"),
    }

    print(f"[violin] input(rewake): {rewake_path}")
    print(f"[violin] auto wake     : {paths['wake']}")
    print(f"[violin] auto dream    : {paths['dream']}")
    print(f"[violin] reviewer      : {args.reviewer}")

    df = build_dataframe(paths, reviewer=args.reviewer)
    if df.empty:
        raise RuntimeError("No valid review rows found. Check files and reviewer key (openai/gemini).")

    if not args.all_temps:
        df = df[df["temp_dream"] >= 1.0].copy()
        print(f"[violin] temp filter   : temp_dream >= 1.0 (n={len(df)})")
    else:
        print(f"[violin] temp filter   : none (all temps, n={len(df)})")

    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # All (always)
    scope_df = df.copy()
    scope_name = "All"
    mann_whitney_report(scope_df, scope_name=scope_name)

    for metric in METRICS:
        out = reports_dir / f"violin_{metric}_{xxxx}_All.png"
        title = f"{metric} by target ({xxxx})"
        violin_plot(scope_df, metric=metric, title=title, out_path=out)
        print(f"[violin] saved: {out}")

    if args.separate:
        themes = sorted([t for t in df["theme"].dropna().unique().tolist() if isinstance(t, str)])
        for theme in themes:
            sub = df[df["theme"] == theme].copy()
            if sub.empty:
                continue

            mann_whitney_report(sub, scope_name=theme)

            for metric in METRICS:
                out = reports_dir / f"violin_{metric}_{xxxx}_{theme}.png"
                title = f"{metric} by target ({xxxx} | {theme})"
                violin_plot(sub, metric=metric, title=title, out_path=out)
                print(f"[violin] saved: {out}")

    print("[violin] done.")


if __name__ == "__main__":
    main()