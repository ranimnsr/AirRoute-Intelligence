"""Generate TGN training curves and baseline comparison chart."""
import sys, os, json
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("results/figures", exist_ok=True)

# --- Figure 11: Training loss curve ---
with open("results/tgn_training_log.json") as f:
    log = json.load(f)

epochs = [e["epoch"] for e in log]
losses = [e["train_loss"] for e in log]
mrrs = [e["val_mrr"] for e in log]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, losses, "o-", color="#E91E63", linewidth=2, markersize=8)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("BPR Loss", fontsize=12)
ax1.set_title("Training Loss (3 months, memory-only TGN)", fontsize=13)
ax1.set_xticks(epochs)

ax2.plot(epochs, mrrs, "s-", color="#2196F3", linewidth=2, markersize=8)
ax2.axhline(y=0.744, color="red", linestyle="--", alpha=0.6, label="HistFreq (0.744)")
ax2.axhline(y=0.682, color="orange", linestyle="--", alpha=0.4, label="Recency (0.682)")
ax2.axhline(y=0.169, color="gray", linestyle="--", alpha=0.3, label="Random (0.169)")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Validation MRR", fontsize=12)
ax2.set_title("Validation MRR vs Baselines", fontsize=13)
ax2.set_xticks(epochs)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig("results/figures/11_tgn_training_curves.png", dpi=150)
plt.close()
print("Saved: results/figures/11_tgn_training_curves.png")

# --- Figure 12: Full model comparison ---
with open("results/baseline_results.json") as f:
    baselines = json.load(f)

names = list(baselines.keys()) + ["TGN"]
mrr_vals = [baselines[n]["mrr"] for n in baselines] + [log[-1]["val_mrr"]]
hits_vals = [baselines[n]["hits@10"] for n in baselines] + [log[-1]["val_hits10"]]
colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars1 = ax1.bar(names, mrr_vals, color=colors, edgecolor="white", linewidth=1.5)
ax1.axhline(y=0.169, color="red", linestyle="--", alpha=0.4, label="Random expected")
ax1.set_ylabel("MRR", fontsize=12)
ax1.set_title("Mean Reciprocal Rank — All Models", fontsize=13)
ax1.set_ylim(0, 1)
for bar, v in zip(bars1, mrr_vals):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)

bars2 = ax2.bar(names, hits_vals, color=colors, edgecolor="white", linewidth=1.5)
ax2.set_ylabel("Hits@10", fontsize=12)
ax2.set_title("Hits@10 — All Models", fontsize=13)
ax2.set_ylim(0, 1.05)
for bar, v in zip(bars2, hits_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/12_full_model_comparison.png", dpi=150)
plt.close()
print("Saved: results/figures/12_full_model_comparison.png")
