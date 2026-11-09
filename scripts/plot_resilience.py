"""Generate resilience analysis charts."""
import sys, os, json
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("results/figures", exist_ok=True)

with open("results/resilience_results.json") as f:
    res = json.load(f)

# --- Figure 15: Disruption curves ---
targeted = res["disruption"]["targeted"]
random_ = res["disruption"]["random"]

x_t = [r["removed"] for r in targeted]
y_t = [r["connectivity"] * 100 for r in targeted]
x_r = [r["removed"] for r in random_]
y_r = [r["connectivity"] * 100 for r in random_]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x_t, y_t, "o-", color="#E91E63", linewidth=2, markersize=3, label="Targeted (betweenness)")
ax1.plot(x_r, y_r, "s-", color="#2196F3", linewidth=2, markersize=3, label="Random")
ax1.set_xlabel("Airports Removed", fontsize=12)
ax1.set_ylabel("Largest SCC (% of total)", fontsize=12)
ax1.set_title("Network Connectivity Under Disruption", fontsize=13)
ax1.legend(fontsize=11)
ax1.set_ylim(80, 90)

# Number of SCCs
scc_t = [r["num_scc"] for r in targeted]
scc_r = [r["num_scc"] for r in random_]
ax2.plot(x_t, scc_t, "o-", color="#E91E63", linewidth=2, markersize=3, label="Targeted")
ax2.plot(x_r, scc_r, "s-", color="#2196F3", linewidth=2, markersize=3, label="Random")
ax2.set_xlabel("Airports Removed", fontsize=12)
ax2.set_ylabel("Number of SCCs", fontsize=12)
ax2.set_title("Network Fragmentation", fontsize=13)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig("results/figures/15_disruption_curves.png", dpi=150)
plt.close()
print("Saved: results/figures/15_disruption_curves.png")

# --- Figure 16: Top airports by centrality ---
top_deg = res["top_airports"]["by_degree"][:10]
top_bc = res["top_airports"]["by_betweenness"][:10]
top_pr = res["top_airports"]["by_pagerank"][:10]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

colors_deg = ["#E91E63"] + ["#64B5F6"] * 9
colors_bc = ["#E91E63"] + ["#64B5F6"] * 9
colors_pr = ["#E91E63"] + ["#64B5F6"] * 9

y_pos = np.arange(10)

axes[0].barh(y_pos, range(10, 0, -1), color=colors_deg, edgecolor="white")
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(top_deg, fontsize=10)
axes[0].set_title("By Weighted Degree", fontsize=12)
axes[0].invert_yaxis()
axes[0].set_xlabel("Rank score", fontsize=10)

axes[1].barh(y_pos, range(10, 0, -1), color=colors_bc, edgecolor="white")
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(top_bc, fontsize=10)
axes[1].set_title("By Betweenness", fontsize=12)
axes[1].invert_yaxis()
axes[1].set_xlabel("Rank score", fontsize=10)

axes[2].barh(y_pos, range(10, 0, -1), color=colors_pr, edgecolor="white")
axes[2].set_yticks(y_pos)
axes[2].set_yticklabels(top_pr, fontsize=10)
axes[2].set_title("By PageRank", fontsize=12)
axes[2].invert_yaxis()
axes[2].set_xlabel("Rank score", fontsize=10)

plt.suptitle("Top 10 Critical Airports by Centrality Metric", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("results/figures/16_top_airports.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/16_top_airports.png")

# --- Figure 17: Community size distribution ---
comm_sizes = sorted(res["communities"]["sizes"], reverse=True)

fig, ax = plt.subplots(figsize=(10, 5))
x_comm = range(1, len(comm_sizes) + 1)
ax.bar(x_comm, comm_sizes, color="#4CAF50", edgecolor="white", alpha=0.8)
ax.set_xlabel("Community Rank", fontsize=12)
ax.set_ylabel("Number of Airports", fontsize=12)
ax.set_title(f"Community Size Distribution ({res['communities']['num_communities']} communities)",
             fontsize=13)
ax.set_xlim(0, min(len(comm_sizes) + 1, 25))
plt.tight_layout()
plt.savefig("results/figures/17_community_sizes.png", dpi=150)
plt.close()
print("Saved: results/figures/17_community_sizes.png")
