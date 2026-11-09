"""Generate temporal plots using fast integer arithmetic on UNIX timestamps."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sys.stdout.reconfigure(line_buffering=True)
os.makedirs("results/figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

csv_path = os.path.join("datasets", "tgbl-flight", "tgbl-flight_edgelist_v2.csv")

print("Reading timestamps (integer arithmetic)...", flush=True)
daily_counts = Counter()
typecode_counter = Counter()

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=10_000_000, usecols=['timestamp', 'typecode'])):
    # Convert UNIX timestamps to day numbers (integer division, no string ops)
    days = (chunk['timestamp'].values // 86400).astype(np.int32)
    unique_days, counts = np.unique(days, return_counts=True)
    for d, c in zip(unique_days, counts):
        daily_counts[int(d)] += int(c)

    tc = chunk['typecode'].dropna()
    if len(tc) > 0:
        typecode_counter.update(tc.value_counts().to_dict())
    print(f"  Chunk {i}", flush=True)

# Convert day numbers back to dates for plotting
from datetime import datetime, timezone, timedelta
epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Plot 3: Daily Interactions
print("Plot 3: Daily interactions...", flush=True)
days_sorted = sorted(daily_counts.keys())
counts_sorted = [daily_counts[d] for d in days_sorted]
date_objs = [epoch + timedelta(days=d) for d in days_sorted]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(date_objs, counts_sorted, linewidth=0.8, color='steelblue', alpha=0.8)
ax.fill_between(date_objs, counts_sorted, alpha=0.3, color='steelblue')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Number of Interactions', fontsize=12)
ax.set_title('Daily Flight Interactions Over Time', fontsize=14)
ax.axvline(datetime(2020, 3, 11, tzinfo=timezone.utc), color='red', linestyle='--', alpha=0.7, label='WHO declares pandemic')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/figures/03_daily_interactions.png", dpi=150)
plt.close()
print("  Saved.", flush=True)

# Plot 4: Monthly Interactions
print("Plot 4: Monthly interactions...", flush=True)
monthly = Counter()
for d, c in daily_counts.items():
    dt = epoch + timedelta(days=d)
    monthly[dt.strftime("%Y-%m")] += c

months_sorted = sorted(monthly.keys())
monthly_vals = [monthly[m] for m in months_sorted]
fig, ax = plt.subplots(figsize=(14, 5))
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

print("\nDone! Temporal plots saved.", flush=True)
