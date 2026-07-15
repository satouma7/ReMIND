# similarity_violin.py: Violin-plot analysis for ReMIND cosine similarities
# - Input: run-level CSV produced by similarity.py (columns include temp_dream, cosine_similarity)
# - Output: violin plot grouped by dream temperature
# - Stats: pairwise Mann–Whitney U tests with Holm correction
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

CSV_PATH = "reports/idea_similarity1.csv"
OUT_DIR = Path("reports/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "violin_wake_dream_similarity_by_temp.png"

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

df = pd.read_csv(CSV_PATH)

# Ensure temperature is categorical and ordered
temp_order = sorted(df["temp_dream"].unique())
df["temp_dream"] = pd.Categorical(df["temp_dream"], categories=temp_order, ordered=True)

plt.figure(figsize=(5, 4))

sns.violinplot(
    data=df,
    x="temp_dream",
    y="cosine_similarity",
    inner="quartile",
    cut=0,
    linewidth=1,
    color="lightgray"
)

sns.stripplot(
    data=df,
    x="temp_dream",
    y="cosine_similarity",
    color="black",
    alpha=0.25,
    size=2,
    jitter=True
)

plt.xlabel("Dream temperature")
plt.ylabel("Cosine similarity (wake–dream)")
plt.ylim(0, 1.02)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(f"[saved] {OUT_FIG}")

temps = [1.0, 3.0, 10.0]
groups = {}
for t in temps:
    subset = df[df["temp_dream"] == t]
    groups[t] = subset["cosine_similarity"]

comparisons = [
    (1.0, 3.0),
    (1.0, 10.0),
    (3.0, 10.0),
]

pvals = []
labels = []

for t1, t2 in comparisons:
    p = mannwhitneyu(
        groups[t1],
        groups[t2],
        alternative="two-sided"
    ).pvalue
    pvals.append(p)
    labels.append(f"{t1} vs {t2}")

# Holm correction
reject, pvals_adj, _, _ = multipletests(pvals, method="holm")

print("\n=== Mann–Whitney U test (Holm corrected) ===")
for lab, p_raw, p_adj, r in zip(labels, pvals, pvals_adj, reject):
    print(f"{lab}: p_raw={p_raw:.3e}, p_adj={p_adj:.3e}, significant={r}")