# similarity_violin.py: Violin-plot analysis for ReMIND cosine similarities
# Input : reports/similarity_XXXX.csv  (from similarity.py)
# Output: reports/similarity_violin_XXXX.png
# Stats : pairwise Mann–Whitney U tests with Holm correction
# Usage:
#   python similarity_violin.py reports/similarity_1ogg.csv
#
# Plot layout:
#   Left panel  (narrow) : dot plot for temp=0 (deterministic; KDE undefined)
#   Right panel (wide)   : violin + strip for temp>0
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def extract_suffix(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.startswith("similarity_"):
        return stem.replace("similarity_", "", 1)
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Violin plot + MWU stats for similarity CSV.")
    ap.add_argument("csv", type=str, help="path to similarity_XXXX.csv")
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="output PNG path (default: reports/similarity_violin_<suffix>.png)",
    )
    ap.add_argument("--dpi", type=int, default=300, help="figure dpi")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    suffix = extract_suffix(csv_path)
    out_fig = Path(args.out).expanduser() if args.out else Path("reports") / f"similarity_violin_{suffix}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    df = pd.read_csv(csv_path)

    required = {"temp_dream", "cosine_similarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    temp_order = sorted(df["temp_dream"].dropna().unique())
    has_zero = 0.0 in temp_order
    temps_rest = [t for t in temp_order if t != 0.0]

    # Detect homo condition: temp=0 has near-zero variance (deterministic output)
    homo = False
    if has_zero:
        std0 = df[df["temp_dream"] == 0.0]["cosine_similarity"].std()
        homo = std0 < 0.02

    y_lim = (0, 1.02)

    if homo and temps_rest:
        # ---- split layout: left dot plot (temp=0) + right violin (temp>0) ----
        n_rest = len(temps_rest)
        fig, (ax0, ax1) = plt.subplots(
            1, 2,
            figsize=(1.2 + n_rest * 0.9, 4),
            gridspec_kw={"width_ratios": [1, n_rest], "wspace": 0.08},
            sharey=True,
        )

        df0 = df[df["temp_dream"] == 0.0]["cosine_similarity"].dropna()
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.15, 0.15, size=len(df0))
        ax0.scatter(jitter, df0.values, color="steelblue", alpha=0.4, s=8, zorder=2)
        mean0 = df0.mean()
        ax0.hlines(mean0, -0.25, 0.25, colors="navy", linewidths=1.5, zorder=3)
        ax0.set_xlim(-0.5, 0.5)
        ax0.set_xticks([0])
        ax0.set_xticklabels(["0"])
        ax0.set_xlabel("temp=0")
        ax0.set_ylabel("Cosine similarity (wake–dream)")
        ax0.set_ylim(*y_lim)
        ax0.spines["right"].set_visible(False)

        df_rest = df[df["temp_dream"].isin(temps_rest)].copy()
        df_rest["temp_dream"] = pd.Categorical(df_rest["temp_dream"], categories=temps_rest, ordered=True)
        sns.violinplot(data=df_rest, x="temp_dream", y="cosine_similarity",
                       inner="quartile", cut=0, linewidth=1, color="lightgray", ax=ax1)
        sns.stripplot(data=df_rest, x="temp_dream", y="cosine_similarity",
                      color="black", alpha=0.25, size=2, jitter=True, ax=ax1)
        ax1.set_xlabel("Dream temperature")
        ax1.set_ylabel("")
        ax1.set_ylim(*y_lim)
        ax1.spines["left"].set_visible(False)
        ax1.tick_params(left=False)

    else:
        # ---- original single violin layout (hetero or no temp=0) ----
        fig, ax = plt.subplots(figsize=(5, 4))
        df["temp_dream"] = pd.Categorical(df["temp_dream"], categories=temp_order, ordered=True)
        sns.violinplot(data=df, x="temp_dream", y="cosine_similarity",
                       inner="quartile", cut=0, linewidth=1, color="lightgray", ax=ax)
        sns.stripplot(data=df, x="temp_dream", y="cosine_similarity",
                      color="black", alpha=0.25, size=2, jitter=True, ax=ax)
        ax.set_xlabel("Dream temperature")
        ax.set_ylabel("Cosine similarity (wake–dream)")
        ax.set_ylim(*y_lim)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=args.dpi)
    plt.close()

    print(f"[saved] {out_fig}")

    # ---- stats: pairwise MWU across temps present in data ----
    temps = [t for t in temp_order if pd.notna(t)]
    if len(temps) < 2:
        print("\n[stats] Not enough temp_dream groups for pairwise tests.")
        return

    groups = {t: df[df["temp_dream"] == t]["cosine_similarity"].dropna() for t in temps}

    comparisons = list(combinations(temps, 2))
    pvals = []
    labels = []

    for t1, t2 in comparisons:
        g1, g2 = groups[t1], groups[t2]
        if len(g1) == 0 or len(g2) == 0:
            continue
        p = mannwhitneyu(g1, g2, alternative="two-sided").pvalue
        pvals.append(p)
        labels.append(f"{t1} vs {t2}")

    if not pvals:
        print("\n[stats] No valid comparisons (some groups empty).")
        return

    # Holm correction
    reject, pvals_adj, _, _ = multipletests(pvals, method="holm")

    print("\n=== Mann–Whitney U test (Holm corrected) ===")
    for lab, p_raw, p_adj, r in zip(labels, pvals, pvals_adj, reject):
        print(f"{lab}: p_raw={p_raw:.3e}, p_adj={p_adj:.3e}, significant={bool(r)}")

if __name__ == "__main__":
    main()