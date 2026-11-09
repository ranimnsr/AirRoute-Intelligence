"""
Phase 1 - Load tgbl-flight directly from CSV (memory-efficient).
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import Counter

csv_path = os.path.join("datasets", "tgbl-flight", "tgbl-flight_edgelist_v2.csv")
node_feat_path = os.path.join("datasets", "tgbl-flight", "airport_node_feat_v2.csv")

# ---- 1. Node Features ----
print("=" * 60)
print("1. NODE FEATURES")
print("=" * 60)
nf = pd.read_csv(node_feat_path)
print(f"Shape: {nf.shape}")
print(f"Columns: {list(nf.columns)}")
print(f"\nFirst 10 rows:")
print(nf.head(10))
print(f"\nColumn analysis:")
for col in nf.columns:
    n_unique = nf[col].nunique()
    n_null = nf[col].isnull().sum()
    print(f"  {col}: {n_unique} unique, {n_null} nulls, dtype={nf[col].dtype}")
    if n_unique <= 20:
        print(f"    Values: {nf[col].value_counts().to_dict()}")
    if nf[col].dtype in ['float64', 'int64']:
        print(f"    Range: [{nf[col].min():.4f}, {nf[col].max():.4f}]")

# ---- 2. Edge List Header ----
print("\n" + "=" * 60)
print("2. EDGE LIST STRUCTURE")
print("=" * 60)
sample = pd.read_csv(csv_path, nrows=20)
print(f"Columns: {list(sample.columns)}")
print(f"Dtypes:\n{sample.dtypes}\n")
print(f"First 10 rows:")
print(sample.head(10))

# Check typecode (edge feature)
print(f"\nCallsign sample: {sample['callsign'].tolist()[:5]}")
print(f"Typecode sample: {sample['typecode'].tolist()[:5]}")

# ---- 3. Full dataset stats (chunked) ----
print("\n" + "=" * 60)
print("3. FULL DATASET STATISTICS (chunked reading)")
print("=" * 60)
chunk_size = 5_000_000
n_total = 0
all_sources = Counter()
all_dests = Counter()
pair_counter = Counter()
ts_min = float('inf')
ts_max = float('-inf')
ts_unique = set()
typecode_counter = Counter()
callsign_counter = Counter()
typecode_nulls = 0

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
    n_total += len(chunk)
    all_sources.update(chunk['src'].tolist())
    all_dests.update(chunk['dst'].tolist())
    for s, d in zip(chunk['src'], chunk['dst']):
        pair_counter[(s, d)] += 1
    chunk_ts = chunk['timestamp'].values
    ts_min = min(ts_min, chunk_ts.min())
    ts_max = max(ts_max, chunk_ts.max())
    ts_unique.update(chunk_ts.tolist())
    typecode_counter.update(chunk['typecode'].dropna().tolist())
    typecode_nulls += chunk['typecode'].isnull().sum()
    print(f"  Chunk {i}: {n_total:,} rows processed")

all_nodes = set(all_sources.keys()) | set(all_dests.keys())
print(f"\n--- Core Statistics ---")
print(f"Total interactions: {n_total:,}")
print(f"Unique sources: {len(all_sources):,}")
print(f"Unique destinations: {len(all_dests):,}")
print(f"Unique nodes in edges: {len(all_nodes):,}")
print(f"Node features CSV rows: {len(nf):,}")
print(f"Nodes in edges but NOT in node feat CSV: {len(all_nodes - set(nf['airport_code'].tolist())):,}")
print(f"Nodes in feat CSV but NOT in edges: {len(set(nf['airport_code'].tolist()) - all_nodes):,}")

print(f"\n--- Timestamps ---")
print(f"Min: {ts_min} = {datetime.fromtimestamp(ts_min, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
print(f"Max: {ts_max} = {datetime.fromtimestamp(ts_max, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
print(f"Unique timestamps: {len(ts_unique):,}")
ts_sorted = sorted(ts_unique)
diffs = np.diff(ts_sorted)
print(f"Timestamp step: min={min(diffs):.0f}s ({min(diffs)/3600:.1f}h), max={max(diffs):.0f}s ({max(diffs)/86400:.1f}d), median={np.median(diffs):.0f}s ({np.median(diffs)/86400:.1f}d)")

print(f"\n--- Edge Features ---")
print(f"Typecode: {len(typecode_counter)} unique values, {typecode_nulls:,} nulls ({100*typecode_nulls/n_total:.1f}%)")
print(f"Top 20 typecodes: {typecode_counter.most_common(20)}")

print(f"\n--- Degree Distribution ---")
out_vals = list(all_sources.values())
in_vals = list(all_dests.values())
total_deg = {n: all_sources.get(n, 0) + all_dests.get(n, 0) for n in all_nodes}
total_vals = list(total_deg.values())
print(f"Out-degree: min={min(out_vals)}, max={max(out_vals):,}, mean={np.mean(out_vals):.0f}, median={np.median(out_vals):.0f}")
print(f"In-degree:  min={min(in_vals)}, max={max(in_vals):,}, mean={np.mean(in_vals):.0f}, median={np.median(in_vals):.0f}")
print(f"Total-deg:  min={min(total_vals)}, max={max(total_vals):,}, mean={np.mean(total_vals):.0f}, median={np.median(total_vals):.0f}")

print(f"\nTop 20 airports by total interactions:")
top20 = sorted(total_deg.items(), key=lambda x: -x[1])[:20]
for node, deg in top20:
    out_d = all_sources.get(node, 0)
    in_d = all_dests.get(node, 0)
    # Get airport info from node features
    info = nf[nf['airport_code'] == node]
    atype = info['type'].values[0] if len(info) > 0 else "?"
    continent = info['continent'].values[0] if len(info) > 0 and pd.notna(info['continent'].values[0]) else "?"
    region = info['iso_region'].values[0] if len(info) > 0 else "?"
    print(f"  {node:6s} ({atype:15s}, {continent:2s}, {region:6s}): total={deg:>9,} (out={out_d:>9,}, in={in_d:>9,})")

print(f"\n--- Repeated Interactions ---")
print(f"Unique (src, dst) pairs: {len(pair_counter):,}")
print(f"Avg interactions per pair: {n_total/len(pair_counter):.1f}")
counts = list(pair_counter.values())
print(f"Pairs with 1 interaction: {sum(1 for c in counts if c == 1):,}")
print(f"Pairs with >10: {sum(1 for c in counts if c > 10):,}")
print(f"Pairs with >100: {sum(1 for c in counts if c > 100):,}")
print(f"Pairs with >1000: {sum(1 for c in counts if c > 1000):,}")
print(f"Max interactions for a pair: {max(counts):,}")
top5_pairs = pair_counter.most_common(5)
for (s, d), c in top5_pairs:
    print(f"  {s} -> {d}: {c:,}")

print(f"\n--- Network Density ---")
n_n = len(all_nodes)
n_p = len(pair_counter)
max_possible = n_n * (n_n - 1)
print(f"Nodes: {n_n:,}, Unique pairs: {n_p:,}")
print(f"Directed density: {n_p/max_possible:.6f} ({n_p/max_possible*100:.4f}%)")

# Airport type distribution among active nodes
print(f"\n--- Airport Types (active nodes only) ---")
active_nf = nf[nf['airport_code'].isin(all_nodes)]
print(f"Active airports with features: {len(active_nf):,}")
print(f"Type distribution:")
print(active_nf['type'].value_counts().to_string())
print(f"\nContinent distribution:")
print(active_nf['continent'].value_counts().to_string())

# Estimated temporal split
print(f"\n--- Estimated Temporal Split (70/15/15) ---")
n70 = int(0.70 * len(ts_sorted))
n85 = int(0.85 * len(ts_sorted))
print(f"~Train period ends: {datetime.fromtimestamp(ts_sorted[n70], tz=timezone.utc).strftime('%Y-%m-%d')}")
print(f"~Val period ends:   {datetime.fromtimestamp(ts_sorted[n85], tz=timezone.utc).strftime('%Y-%m-%d')}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
