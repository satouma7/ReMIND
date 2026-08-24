# plot_scatter.py: Scatter plots for ReMIND analyses (pair-wise)
# Input:
#   - reports/<NAME>.csv (e.g., 5ooo_space_time_rank.csv)
#     Must include: cosine_similarity, alignment, coherence, novelty, sum_score
#     Optional: rank, pair
# Output (in reports/):
#   - scatter_alignment_<suffix>.png
#   - scatter_coherence_<suffix>.png
#   - scatter_novelty_<suffix>.png
#   - scatter_sum_score_<suffix>.png
#   - (optional) scatter_rank_<suffix>.png
# Usage:
#  python plot_scatter.py 5ooo_space_time_rank.csv
# Notes:
#   - Spearman correlation is reported in plot title.
#   - If "pair" has multiple values, plots are generated per pair.

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

REPORTS_DIR = Path("reports")

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'["“”\'`]', "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "pair"

def infer_suffix(path: Path) -> str:
    # Example: 5ooo_space_time_rank.csv -> 5ooo_space_time
    stem = path.stem
    if stem.endswith("_rank"):
        stem = stem[:-5]
    # If someone passes review_similarity_5ooo.csv
    stem = stem.replace("review_similarity_", "")
    return stem

def spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    sub = pd.concat([x, y], axis=1).dropna()
    if len(sub) < 3:
        return float("nan"), float("nan"), len(sub)
    rho, p = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
    return float(rho), float(p), len(sub)

def scatter_plot(df: pd.DataFrame, x: str, y: str, ylabel: str, out_path: Path) -> None:
    # compute correlation on non-missing
    rho, p, n = spearman(df[x], df[y])

    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.7)

    # regression line (visual guide only)
    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter=False,
        color="black",
        line_kws={"linewidth": 1, "linestyle": "--"},
    )

    plt.xlabel("Cosine similarity (wake–dream)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs cosine similarity\nSpearman ρ = {rho:.2f} (p = {p:.2e}, n = {n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="",
                    help="input CSV (default: find in reports/)")
    ap.add_argument("--out-dir", default="reports", help="output directory (default: reports/)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve input path ----
    in_path = Path(args.csv) if args.csv else None
    if in_path and in_path.exists():
        pass
    else:
        # if user gave only a filename, try reports/
        if in_path and not in_path.exists():
            cand = REPORTS_DIR / in_path.name
            if cand.exists():
                in_path = cand
        # if nothing given, fail loudly with hint
        if not in_path or not in_path.exists():
            raise FileNotFoundError(
                f"Input CSV not found. Pass a path like: python plot_scatter.py reports/5ooo_space_time_rank.csv"
            )

    suffix = infer_suffix(in_path)

    df = pd.read_csv(in_path)

    required = {"cosine_similarity", "alignment", "coherence", "novelty", "sum_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Keep only rows with cosine similarity
    df = df.dropna(subset=["cosine_similarity"]).copy()

    has_rank = "rank" in df.columns
    has_pair = "pair" in df.columns and df["pair"].notna().any()
    pair_values = []
    if has_pair:
        pair_values = [p for p in df["pair"].dropna().unique().tolist() if str(p).strip()]

    # Decide grouping: per-pair if multiple pairs exist
    if has_pair and len(pair_values) > 1:
        groups = []
        for p in pair_values:
            sub = df[df["pair"] == p].copy()
            groups.append((str(p), sub))
    else:
        groups = [("all", df)]

    print(f"[plot_scatter] input : {in_path}")
    print(f"[plot_scatter] rows  : {len(df)} (after dropna cosine_similarity)")
    print(f"[plot_scatter] rank  : {'yes' if has_rank else 'no'}")
    print(f"[plot_scatter] pairs : {len(groups)}")

    for pair_name, gdf in groups:
        pair_tag = "" if pair_name == "all" else f"_{slugify(pair_name)}"
        base = f"{suffix}{pair_tag}"

        scatter_plot(
            gdf, "cosine_similarity", "alignment", "Alignment score",
            out_dir / f"scatter_alignment_{base}.png"
        )
        scatter_plot(
            gdf, "cosine_similarity", "coherence", "Coherence score",
            out_dir / f"scatter_coherence_{base}.png"
        )
        scatter_plot(
            gdf, "cosine_similarity", "novelty", "Novelty score",
            out_dir / f"scatter_novelty_{base}.png"
        )
        scatter_plot(
            gdf, "cosine_similarity", "sum_score", "External evaluation (sum score)",
            out_dir / f"scatter_sum_score_{base}.png"
        )

        # Optional: rank correlation output + plot
        if has_rank and gdf["rank"].notna().any():
            # rank vs novelty (relative vs absolute novelty)
            rho_rn, p_rn, n_rn = spearman(gdf["rank"], gdf["novelty"])
            print(f"[spearman] ({base}) rank vs novelty: rho={rho_rn:.3f}, p={p_rn:.3e}, n={n_rn}")

            # often useful too
            rho_rs, p_rs, n_rs = spearman(gdf["rank"], gdf["sum_score"])
            print(f"[spearman] ({base}) rank vs sum_score: rho={rho_rs:.3f}, p={p_rs:.3e}, n={n_rs}")

            # Plot rank (x) vs novelty? ここは好み。必要ならON。
            plt.figure(figsize=(5, 4))
            sub = gdf.dropna(subset=["rank", "novelty"])
            if len(sub) >= 3:
                sns.scatterplot(data=sub, x="rank", y="novelty", alpha=0.7)
                sns.regplot(
                    data=sub, x="rank", y="novelty",
                    scatter=False, color="black",
                    line_kws={"linewidth": 1, "linestyle": "--"},
                )
            plt.xlabel("Relative rank (1=most novel)")
            plt.ylabel("Novelty score (absolute)")
            plt.title(f"Novelty score vs relative rank\nSpearman ρ = {rho_rn:.2f} (p = {p_rn:.2e}, n = {n_rn})")
            plt.tight_layout()
            plt.savefig(out_dir / f"scatter_rank_vs_novelty_{base}.png", dpi=300)
            plt.close()

    print(f"[done] scatter plots saved to {out_dir}")

if __name__ == "__main__":
    main()