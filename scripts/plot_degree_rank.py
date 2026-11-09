"""Generate degree rank plot from src/dst data."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
sns.set_theme(style="whitegrid", palette="muted")

csv_path = os.path.join("datasets", "tgbl-flight", "tgbl-flight_edgelist_v2.csv")
src_counts = Counter()
dst_counts = Counter()

print("Reading src/dst...", flush=True)
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=10_000_000, usecols=['src', 'dst'])):
    src_counts.update(chunk['src'].value_counts().to_dict())
    dst_counts.update(chunk['dst'].value_counts().to_dict())
    print(f"  Chunk {i}", flush=True)

all_nodes = set(src_counts.keys()) | set(dst_counts.keys())
degs = sorted([src_counts.get(n, 0) + dst_counts.get(n, 0) for n in all_nodes], reverse=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(degs)+1), degs, color='coral', linewidth=1.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Airport Rank', fontsize=12)
ax.set_ylabel('Total Degree', fontsize=12)
ax.set_title('Airport Degree Rank Plot (Zipf-like Distribution)', fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/07_degree_rank.png", dpi=150)
plt.close()
print("Saved: 07_degree_rank.png", flush=True)
