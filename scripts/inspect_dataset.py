"""
Phase 1 - Dataset inspection script.
Loads tgbl-flight via TGB and prints all key statistics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timezone
from collections import Counter

print("=" * 60)
print("LOADING tgbl-flight dataset from TGB...")
print("=" * 60)

from tgb.linkproppred.dataset import LinkPropPredDataset

dataset = LinkPropPredDataset(name="tgbl-flight", root="datasets", preprocess=True)
data = dataset.full_data

print("\n=== DATASET LOADED SUCCESSFULLY ===\n")

# Basic info
print("--- Keys in full_data ---")
for k, v in data.items():
    if v is not None:
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v).__name__}")
    else:
        print(f"  {k}: None")

src = data["sources"]
dst = data["destinations"]
ts = data["timestamps"]
edge_feat = data.get("edge_feat")
edge_label = data.get("edge_label")

print(f"\n--- Core Statistics ---")
print(f"Number of interactions (edges): {len(src):,}")
print(f"Number of unique sources: {len(np.unique(src)):,}")
print(f"Number of unique destinations: {len(np.unique(dst)):,}")
all_nodes = np.unique(np.concatenate([src, dst]))
print(f"Number of unique nodes (airports): {len(all_nodes):,}")
print(f"Node ID range: [{all_nodes.min()}, {all_nodes.max()}]")

print(f"\n--- Timestamps ---")
print(f"Min timestamp: {ts.min()}")
print(f"Max timestamp: {ts.max()}")
ts_min_dt = datetime.fromtimestamp(ts.min(), tz=timezone.utc)
ts_max_dt = datetime.fromtimestamp(ts.max(), tz=timezone.utc)
print(f"Date range: {ts_min_dt.strftime('%Y-%m-%d')} to {ts_max_dt.strftime('%Y-%m-%d')}")
unique_ts = np.unique(ts)
print(f"Number of unique timestamps: {len(unique_ts):,}")
print(f"Timestamp dtype: {ts.dtype}")
print(f"First 10 unique timestamps: {unique_ts[:10]}")

# Check granularity
if len(unique_ts) > 1:
    diffs = np.diff(unique_ts)
    print(f"Min time diff: {diffs.min():.0f}s = {diffs.min()/3600:.1f}h")
    print(f"Max time diff: {diffs.max():.0f}s = {diffs.max()/86400:.1f}d")
    print(f"Median time diff: {np.median(diffs):.0f}s = {np.median(diffs)/86400:.1f}d")

print(f"\n--- Edge Features ---")
if edge_feat is not None:
    print(f"Shape: {edge_feat.shape}")
    print(f"Dtype: {edge_feat.dtype}")
    n_unique = len(np.unique(edge_feat, axis=0)) if edge_feat.ndim > 1 else len(np.unique(edge_feat))
    print(f"Number of unique edge features: {n_unique:,}")
    print(f"Sample (first 5): {edge_feat[:5]}")
    print(f"Min: {edge_feat.min()}, Max: {edge_feat.max()}, Mean: {edge_feat.mean():.4f}")
else:
    print("No edge features")

print(f"\n--- Edge Labels ---")
if edge_label is not None:
    print(f"Shape: {edge_label.shape}")
    unique_labels = np.unique(edge_label)
    print(f"Unique values ({len(unique_labels)}): {unique_labels[:20]}")
else:
    print("No edge labels")

print(f"\n--- Node Features ---")
if hasattr(dataset, "node_feat") and dataset.node_feat is not None:
    nf = dataset.node_feat
    print(f"Shape: {nf.shape}")
    print(f"Dtype: {nf.dtype}")
    print(f"Sample (first 3 rows):")
    for i in range(min(3, len(nf))):
        print(f"  Node {i}: {nf[i]}")
    print(f"Min per col: {nf.min(axis=0)}")
    print(f"Max per col: {nf.max(axis=0)}")
else:
    print("No node features in dataset object")
    # Check CSV directly
    csv_path = os.path.join("datasets", "tgbl-flight", "airport_node_feat_v2.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"\nFound node feature CSV: {csv_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nValue counts for categorical columns:")
        for col in df.columns:
            nunique = df[col].nunique()
            if nunique < 30:
                print(f"  {col}: {nunique} unique values")
                print(f"    {df[col].value_counts().head(10).to_dict()}")

print(f"\n--- Train/Val/Test Split ---")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
n_total = len(src)
n_train = int(np.sum(train_mask))
n_val = int(np.sum(val_mask))
n_test = int(np.sum(test_mask))
print(f"Total interactions: {n_total:,}")
print(f"Train: {n_train:,} ({100*n_train/n_total:.1f}%)")
print(f"Val:   {n_val:,} ({100*n_val/n_total:.1f}%)")
print(f"Test:  {n_test:,} ({100*n_test/n_total:.1f}%)")

train_ts = ts[train_mask]
val_ts = ts[val_mask]
test_ts = ts[test_mask]
print(f"\nTrain period: {datetime.fromtimestamp(train_ts.min(), tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(train_ts.max(), tz=timezone.utc).strftime('%Y-%m-%d')}")
print(f"Val period:   {datetime.fromtimestamp(val_ts.min(), tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(val_ts.max(), tz=timezone.utc).strftime('%Y-%m-%d')}")
print(f"Test period:  {datetime.fromtimestamp(test_ts.min(), tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(test_ts.max(), tz=timezone.utc).strftime('%Y-%m-%d')}")

# Verify chronological ordering
print(f"\nChronological check:")
print(f"  Train max <= Val min? {train_ts.max() <= val_ts.min()} (train_max={train_ts.max():.0f}, val_min={val_ts.min():.0f})")
print(f"  Val max <= Test min?  {val_ts.max() <= test_ts.min()} (val_max={val_ts.max():.0f}, test_min={test_ts.min():.0f})")

print(f"\n--- Evaluation Metric ---")
print(f"Official metric: {dataset.eval_metric}")

print(f"\n--- Repeated Interactions ---")
pair_counts = Counter(zip(src.tolist(), dst.tolist()))
print(f"Total interactions: {len(src):,}")
print(f"Unique (src, dst) pairs: {len(pair_counts):,}")
print(f"Average interactions per unique pair: {len(src)/len(pair_counts):.1f}")
counts = list(pair_counts.values())
print(f"Pairs with 1 interaction: {sum(1 for c in counts if c == 1):,}")
print(f"Pairs with >10 interactions: {sum(1 for c in counts if c > 10):,}")
print(f"Pairs with >100 interactions: {sum(1 for c in counts if c > 100):,}")
print(f"Pairs with >1000 interactions: {sum(1 for c in counts if c > 1000):,}")
print(f"Max interactions for a single pair: {max(counts):,}")
top_pairs = pair_counts.most_common(5)
for (s, d), c in top_pairs:
    print(f"  ({s}, {d}): {c:,} interactions")

print(f"\n--- Degree Distribution ---")
out_deg = Counter(src.tolist())
in_deg = Counter(dst.tolist())
total_deg = Counter()
for n in all_nodes:
    total_deg[n] = out_deg.get(n, 0) + in_deg.get(n, 0)

out_vals = list(out_deg.values())
in_vals = list(in_deg.values())
total_vals = list(total_deg.values())
print(f"Out-degree: min={min(out_vals)}, max={max(out_vals):,}, mean={np.mean(out_vals):.0f}, median={np.median(out_vals):.0f}")
print(f"In-degree:  min={min(in_vals)}, max={max(in_vals):,}, mean={np.mean(in_vals):.0f}, median={np.median(in_vals):.0f}")
print(f"Total-deg:  min={min(total_vals)}, max={max(total_vals):,}, mean={np.mean(total_vals):.0f}, median={np.median(total_vals):.0f}")

print(f"\nTop 20 airports by total degree:")
for node, deg in total_deg.most_common(20):
    out = out_deg.get(node, 0)
    inp = in_deg.get(node, 0)
    print(f"  Node {node:5d}: total={deg:>8,}  (out={out:>8,}, in={inp:>8,})")

print(f"\n--- Network Density ---")
n_nodes = len(all_nodes)
n_pairs = len(pair_counts)
max_possible_pairs = n_nodes * (n_nodes - 1)
density = n_pairs / max_possible_pairs if max_possible_pairs > 0 else 0
print(f"Nodes: {n_nodes:,}")
print(f"Unique directed pairs: {n_pairs:,}")
print(f"Max possible directed pairs: {max_possible_pairs:,}")
print(f"Density: {density:.6f} ({density*100:.4f}%)")

print(f"\n--- Memory Usage ---")
total_bytes = 0
for k, v in data.items():
    if isinstance(v, np.ndarray):
        total_bytes += v.nbytes
        print(f"  {k}: {v.nbytes / 1e6:.1f} MB")
if hasattr(dataset, "node_feat") and dataset.node_feat is not None:
    total_bytes += dataset.node_feat.nbytes
    print(f"  node_feat: {dataset.node_feat.nbytes / 1e6:.1f} MB")
print(f"  TOTAL in-memory: {total_bytes / 1e6:.1f} MB ({total_bytes / 1e9:.2f} GB)")

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETE")
print("=" * 60)
