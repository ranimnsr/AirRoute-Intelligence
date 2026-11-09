"""Generate the temporal train/val/test split visualization."""
import sys, os
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime, timezone, timedelta

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("results/figures", exist_ok=True)

print("Loading preprocessed data...", flush=True)
timestamps = np.load("data/processed/timestamps.npy")
train_mask = np.load("data/processed/train_mask.npy")
val_mask = np.load("data/processed/val_mask.npy")
test_mask = np.load("data/processed/test_mask.npy")

epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

fig, ax = plt.subplots(figsize=(14, 5))
for label, mask, color in [
    ("Train (70%)", train_mask, "#2196F3"),
    ("Validation (15%)", val_mask, "#FF9800"),
    ("Test (15%)", test_mask, "#F44336"),
]:
    ts_masked = timestamps[mask]
    days = (ts_masked // 86400).astype(np.int32)
    unique_days, counts = np.unique(days, return_counts=True)
    dates = [epoch + timedelta(days=int(d)) for d in unique_days]
    ax.fill_between(dates, counts, alpha=0.5, color=color, label=label)
    ax.plot(dates, counts, color=color, linewidth=0.5, alpha=0.7)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Daily Interactions", fontsize=12)
ax.set_title("Temporal Train/Val/Test Split", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("results/figures/09_temporal_split.png", dpi=150)
plt.close()
print("Saved: results/figures/09_temporal_split.png", flush=True)
