"""Generate Phase 5 comparison charts: val vs test for all models."""
import sys, os, json
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("results/figures", exist_ok=True)

with open("results/baseline_results.json") as f:
    val_results = json.load(f)
with open("results/tgn_training_log.json") as f:
    tgn_log = json.load(f)
val_results["TGN"] = {
    "mrr": tgn_log[-1]["val_mrr"],
    "hits@10": tgn_log[-1]["val_hits10"],
}

with open("results/test_results.json") as f:
    test_results = json.load(f)

models = ["Random", "HistFreq", "Recency", "EdgeBank", "TGN"]
colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

# --- Figure 13: Test set results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

mrr_test = [test_results[m]["mrr"] for m in models]
hits_test = [test_results[m]["hits@10"] for m in models]

bars1 = ax1.bar(models, mrr_test, color=colors, edgecolor="white", linewidth=1.5)
ax1.axhline(y=0.169, color="red", linestyle="--", alpha=0.4, label="Random expected")
ax1.set_ylabel("MRR", fontsize=12)
ax1.set_title("Test Set — Mean Reciprocal Rank", fontsize=13)
ax1.set_ylim(0, 1)
for bar, v in zip(bars1, mrr_test):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)

bars2 = ax2.bar(models, hits_test, color=colors, edgecolor="white", linewidth=1.5)
ax2.set_ylabel("Hits@10", fontsize=12)
ax2.set_title("Test Set — Hits@10", fontsize=13)
ax2.set_ylim(0, 1.05)
for bar, v in zip(bars2, hits_test):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("results/figures/13_test_results.png", dpi=150)
plt.close()
print("Saved: results/figures/13_test_results.png")

# --- Figure 14: Val vs Test comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(models))
w = 0.35

mrr_val = [val_results[m]["mrr"] for m in models]
mrr_test = [test_results[m]["mrr"] for m in models]

bars_v1 = ax1.bar(x - w/2, mrr_val, w, label="Validation", color="#64B5F6", edgecolor="white")
bars_t1 = ax1.bar(x + w/2, mrr_test, w, label="Test", color="#E91E63", edgecolor="white")
ax1.set_ylabel("MRR", fontsize=12)
ax1.set_title("MRR: Validation vs Test", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, 1)
ax1.legend(fontsize=10)
for bar, v in zip(bars_v1, mrr_val):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=9, color="#1565C0")
for bar, v in zip(bars_t1, mrr_test):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=9, color="#C62828")

hits_val = [val_results[m]["hits@10"] for m in models]
hits_test = [test_results[m]["hits@10"] for m in models]

bars_v2 = ax2.bar(x - w/2, hits_val, w, label="Validation", color="#64B5F6", edgecolor="white")
bars_t2 = ax2.bar(x + w/2, hits_test, w, label="Test", color="#E91E63", edgecolor="white")
ax2.set_ylabel("Hits@10", fontsize=12)
ax2.set_title("Hits@10: Validation vs Test", fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=10)
for bar, v in zip(bars_v2, hits_val):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=9, color="#1565C0")
for bar, v in zip(bars_t2, hits_test):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{v:.3f}", ha="center", fontsize=9, color="#C62828")

plt.tight_layout()
plt.savefig("results/figures/14_val_vs_test.png", dpi=150)
plt.close()
print("Saved: results/figures/14_val_vs_test.png")
