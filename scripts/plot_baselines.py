"""Generate baseline comparison chart."""
import sys, os, json
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("results/figures", exist_ok=True)

with open("results/baseline_results.json") as f:
    results = json.load(f)

names = list(results.keys())
mrrs = [results[n]["mrr"] for n in names]
hits = [results[n]["hits@10"] for n in names]
colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars1 = ax1.bar(names, mrrs, color=colors, edgecolor="white", linewidth=1.5)
ax1.axhline(y=0.169, color="red", linestyle="--", alpha=0.5, label="Random expected")
ax1.set_ylabel("MRR", fontsize=12)
ax1.set_title("Mean Reciprocal Rank (higher = better)", fontsize=13)
ax1.set_ylim(0, 1)
for bar, v in zip(bars1, mrrs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)

bars2 = ax2.bar(names, hits, color=colors, edgecolor="white", linewidth=1.5)
ax2.set_ylabel("Hits@10", fontsize=12)
ax2.set_title("Hits@10 (higher = better)", fontsize=13)
ax2.set_ylim(0, 1)
for bar, v in zip(bars2, hits):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/10_baseline_comparison.png", dpi=150)
plt.close()
print("Saved: results/figures/10_baseline_comparison.png")
