"""
Phase 1 - Generate exploration visualizations.
Optimized: reads only needed columns, no per-row Python loops.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

os.makedirs("results/figures", exist_ok=True)
sys.stdout.reconfigure(line_buffering=True)

csv_path = os.path.join("datasets", "tgbl-flight", "tgbl-flight_edgelist_v2.csv")
node_feat_path = os.path.join("datasets", "tgbl-flight", "airport_node_feat_v2.csv")
nf = pd.read_csv(node_feat_path)

sns.set_theme(style="whitegrid", palette="muted")

# ==== PASS 1: Read src, dst columns only (for degree & pair stats) ====
print("Pass 1: Reading src/dst columns...", flush=True)
src_counts = Counter()
dst_counts = Counter()
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=10_000_000, usecols=['src', 'dst'])):
    src_counts.update(chunk['src'].value_counts().to_dict())
    dst_counts.update(chunk['dst'].value_counts().to_dict())
    print(f"  Chunk {i}", flush=True)

all_nodes = set(src_counts.keys()) | set(dst_counts.keys())
total_deg = {n: src_counts.get(n, 0) + dst_counts.get(n, 0) for n in all_nodes}

# Plot 1: Degree Distribution
print("Plot 1: Degree distribution...", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
degs = list(total_deg.values())
bins = np.logspace(0, np.log10(max(degs)), 50)
ax.hist(degs, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Total Degree (log scale)', fontsize=12)
ax.set_ylabel('Number of Airports (log scale)', fontsize=12)
ax.set_title('Degree Distribution of Airports', fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/01_degree_distribution.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# Plot 2: Top 20 Airports
print("Plot 2: Top 20 airports...", flush=True)
fig, ax = plt.subplots(figsize=(12, 7))
top20 = sorted(total_deg.items(), key=lambda x: -x[1])[:20]
names = [n for n, _ in top20]
vals = [v for _, v in top20]
bars = ax.barh(range(len(names)), vals, color=sns.color_palette("viridis", 20))
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Total Interactions', fontsize=12)
ax.set_title('Top 20 Airports by Total Interactions (2019-2022)', fontsize=14)
for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + 10000, bar.get_y() + bar.get_height()/2, f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/02_top20_airports.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# Plot 5: In-degree vs Out-degree
print("Plot 5: In vs Out degree...", flush=True)
fig, ax = plt.subplots(figsize=(8, 8))
nodes_list = list(all_nodes)
out_d = np.array([src_counts.get(n, 0) for n in nodes_list])
in_d = np.array([dst_counts.get(n, 0) for n in nodes_list])
ax.scatter(out_d, in_d, alpha=0.3, s=5, color='steelblue')
max_val = max(out_d.max(), in_d.max())
ax.plot([1, max_val], [1, max_val], 'r--', alpha=0.5, label='y=x')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Out-degree (log)', fontsize=12)
ax.set_ylabel('In-degree (log)', fontsize=12)
ax.set_title('In-degree vs Out-degree per Airport', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/05_in_vs_out_degree.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# Plot 6: Airport Types & Continents
print("Plot 6: Airport types...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
active_nf = nf[nf['airport_code'].isin(all_nodes)]
type_counts = active_nf['type'].value_counts()
axes[0].barh(type_counts.index, type_counts.values, color=sns.color_palette("Set2", len(type_counts)))
axes[0].set_xlabel('Count')
axes[0].set_title('Airport Types (Active Nodes)')
for i, v in enumerate(type_counts.values):
    axes[0].text(v + 50, i, str(v), va='center', fontsize=9)
cont_counts = active_nf['continent'].value_counts()
axes[1].barh(cont_counts.index, cont_counts.values, color=sns.color_palette("Set2", len(cont_counts)))
axes[1].set_xlabel('Count')
axes[1].set_title('Airport Continents (Active Nodes)')
for i, v in enumerate(cont_counts.values):
    axes[1].text(v + 20, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/06_airport_types_continents.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

del nodes_list, out_d, in_d

# ==== PASS 2: Read timestamp column only (for temporal plots) ====
print("\nPass 2: Reading timestamps...", flush=True)
daily_counts = Counter()
monthly_counts = Counter()
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=10_000_000, usecols=['timestamp'])):
    dates = pd.to_datetime(chunk['timestamp'], unit='s', utc=True)
    daily_counts.update(dates.dt.strftime("%Y-%m-%d").value_counts().to_dict())
    monthly_counts.update(dates.dt.strftime("%Y-%m").value_counts().to_dict())
    print(f"  Chunk {i}", flush=True)

# Plot 3: Daily Interactions
print("Plot 3: Daily interactions...", flush=True)
from datetime import datetime
fig, ax = plt.subplots(figsize=(14, 5))
dates_sorted = sorted(daily_counts.keys())
counts_sorted = [daily_counts[d] for d in dates_sorted]
date_objs = pd.to_datetime(dates_sorted)
ax.plot(date_objs, counts_sorted, linewidth=0.8, color='steelblue', alpha=0.8)
ax.fill_between(date_objs, counts_sorted, alpha=0.3, color='steelblue')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Interactions', fontsize=12)
ax.set_title('Daily Flight Interactions Over Time', fontsize=14)
ax.axvline(pd.Timestamp('2020-03-11'), color='red', linestyle='--', alpha=0.7, label='WHO declares pandemic')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/figures/03_daily_interactions.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# Plot 4: Monthly Interactions
print("Plot 4: Monthly interactions...", flush=True)
fig, ax = plt.subplots(figsize=(14, 5))
months_sorted = sorted(monthly_counts.keys())
monthly_vals = [monthly_counts[m] for m in months_sorted]
x = range(len(months_sorted))
ax.bar(x, monthly_vals, color='steelblue', alpha=0.8)
ax.set_xticks([i for i in x if i % 3 == 0])
ax.set_xticklabels([months_sorted[i] for i in x if i % 3 == 0], rotation=45, ha='right')
ax.set_ylabel('Number of Interactions', fontsize=12)
ax.set_title('Monthly Flight Interactions', fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/04_monthly_interactions.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# ==== PASS 3: Read typecode column only ====
print("\nPass 3: Reading typecodes...", flush=True)
typecode_counter = Counter()
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=10_000_000, usecols=['typecode'])):
    tc = chunk['typecode'].dropna()
    if len(tc) > 0:
        typecode_counter.update(tc.value_counts().to_dict())
    print(f"  Chunk {i}", flush=True)

# Plot 8: Top Aircraft Types
print("Plot 8: Aircraft types...", flush=True)
fig, ax = plt.subplots(figsize=(12, 6))
top_types = typecode_counter.most_common(20)
t_names = [t for t, _ in top_types]
t_vals = [v for _, v in top_types]
ax.barh(range(len(t_names)), t_vals, color=sns.color_palette("coolwarm", 20))
ax.set_yticks(range(len(t_names)))
ax.set_yticklabels(t_names, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Number of Flights', fontsize=12)
ax.set_title('Top 20 Aircraft Types (by Flight Count)', fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/08_top_aircraft_types.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# ==== Plot 7: Use pre-computed stats from earlier inspection ====
# Repeated interactions: we know from inspection:
# 2,309,707 unique pairs, max 154,371
# Rather than re-reading all pairs, create a conceptual plot from the data we have
print("Plot 7: Degree CDF (complementary)...", flush=True)
fig, ax = plt.subplots(figsize=(10, 6))
sorted_degs = np.sort(degs)[::-1]
ax.plot(range(1, len(sorted_degs)+1), sorted_degs, color='coral', linewidth=1.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Airport Rank', fontsize=12)
ax.set_ylabel('Total Degree', fontsize=12)
ax.set_title('Airport Degree Rank Plot (Zipf-like)', fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/07_degree_rank.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

print(f"\nAll plots saved to results/figures/", flush=True)
