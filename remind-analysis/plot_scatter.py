# plot_scatter.py: Scatter plots of wake–dream cosine similarity vs review scores
# Input:
#   - reports/idea_similarity_with_review.csv
# Output:
#   - Scatter plots comparing cosine similarity with:
#       * total review score
#       * novelty, coherence, and alignment scores
# Notes:
#   - Spearman correlation (ρ) is reported for each plot.
#   - Regression lines are shown for visualization only.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

CSV_PATH = "reports/idea_similarity_with_review.csv"
OUT_DIR = "reports"
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["cosine_similarity"])

def scatter(x, y, ylabel, fname):
    plt.figure(figsize=(5,4))

    # 欠損を除いたデータで相関を計算
    sub = df[[x, y]].dropna()
    rho, pval = spearmanr(sub[x], sub[y])

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="pair",
        alpha=0.7
    )
    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter=False,
        color="black",
        line_kws={"linewidth":1, "linestyle":"--"}
    )

    plt.xlabel("Cosine similarity (wake–dream)")
    plt.ylabel(ylabel)

    plt.title(
        f"{ylabel} vs cosine similarity\n"
        f"Spearman ρ = {rho:.2f} (p = {pval:.2e})"
    )

    plt.legend([], [], frameon=False)  # delete legend
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=300)
    plt.close()

scatter(
    x="cosine_similarity",
    y="sum_score",
    ylabel="External evaluation (sum score)",
    fname="scatter_cosine_vs_sum_score.png"
)

scatter(
    x="cosine_similarity",
    y="novelty",
    ylabel="Novelty score",
    fname="scatter_cosine_vs_novelty.png"
)

scatter(
    x="cosine_similarity",
    y="coherence",
    ylabel="Coherence score",
    fname="scatter_cosine_vs_coherence.png"
)

scatter(
    x="cosine_similarity",
    y="alignment",
    ylabel="Alignment score",
    fname="scatter_cosine_vs_alignment.png"
)

print("[done] scatter plots saved to", OUT_DIR)