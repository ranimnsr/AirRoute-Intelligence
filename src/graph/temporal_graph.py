"""
Temporal graph representation for tgbl-flight.

This module provides TemporalFlightGraph, which wraps the preprocessed
numpy arrays into PyTorch Geometric's TemporalData format and offers
utilities for temporal sub-sampling, split access, and basic statistics.

Key design decisions:
  - Event-based representation (not snapshots): each edge is a timestamped
    interaction (src, dst, t, feat), matching TGN's expected input format.
  - Directed graph: flight A→B ≠ B→A (confirmed by Phase 1 analysis).
  - Repeated interactions: the same (src, dst) pair can appear many times
    at different timestamps — these are interactions, not persistent routes.
  - Chronological ordering is preserved and enforced.
"""

import os
import numpy as np
import torch
from torch_geometric.data import TemporalData


class TemporalFlightGraph:
    """Wrapper around preprocessed tgbl-flight data.

    Loads the numpy arrays produced by preprocessing.py and provides:
      - PyG TemporalData objects for train/val/test splits
      - Sub-sampling by date range for development on limited hardware
      - Graph statistics (nodes, edges, time range, density)
      - Node feature access
    """

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

        # Load arrays (memory-mapped for low footprint until accessed)
        self.sources = np.load(os.path.join(processed_dir, "sources.npy"), mmap_mode="r")
        self.destinations = np.load(os.path.join(processed_dir, "destinations.npy"), mmap_mode="r")
        self.timestamps = np.load(os.path.join(processed_dir, "timestamps.npy"), mmap_mode="r")
        self.edge_feat = np.load(os.path.join(processed_dir, "edge_feat.npy"), mmap_mode="r")
        self.node_features = np.load(os.path.join(processed_dir, "node_features.npy"))

        self.train_mask = np.load(os.path.join(processed_dir, "train_mask.npy"), mmap_mode="r")
        self.val_mask = np.load(os.path.join(processed_dir, "val_mask.npy"), mmap_mode="r")
        self.test_mask = np.load(os.path.join(processed_dir, "test_mask.npy"), mmap_mode="r")

        # Load node mapping for interpretability
        import pandas as pd
        self.node_mapping = pd.read_csv(os.path.join(processed_dir, "node_mapping.csv"))
        self._code_to_id = dict(zip(self.node_mapping["airport_code"], self.node_mapping["node_id"]))
        self._id_to_code = dict(zip(self.node_mapping["node_id"], self.node_mapping["airport_code"]))

    @property
    def num_nodes(self) -> int:
        return len(self.node_features)

    @property
    def num_edges(self) -> int:
        return len(self.sources)

    @property
    def num_features(self) -> int:
        return self.node_features.shape[1]

    @property
    def time_range(self) -> tuple:
        return int(self.timestamps[0]), int(self.timestamps[-1])

    def airport_code(self, node_id: int) -> str:
        return self._id_to_code[node_id]

    def node_id(self, airport_code: str) -> int:
        return self._code_to_id[airport_code]

    def get_split(self, split: str) -> TemporalData:
        """Get a TemporalData object for a split ('train', 'val', 'test').

        Returns a PyG TemporalData with:
          - src: source node IDs
          - dst: destination node IDs
          - t: timestamps
          - msg: edge features (reshaped to 2D as TGN expects)
        """
        mask = {"train": self.train_mask, "val": self.val_mask, "test": self.test_mask}[split]

        # Convert to contiguous arrays then tensors
        src = torch.from_numpy(np.array(self.sources[mask], dtype=np.int64))
        dst = torch.from_numpy(np.array(self.destinations[mask], dtype=np.int64))
        t = torch.from_numpy(np.array(self.timestamps[mask], dtype=np.int64))
        msg = torch.from_numpy(np.array(self.edge_feat[mask], dtype=np.float32)).unsqueeze(1)

        return TemporalData(src=src, dst=dst, t=t, msg=msg)

    def get_subsample(self, start_ts: int, end_ts: int) -> TemporalData:
        """Get a TemporalData for a time window [start_ts, end_ts].

        Useful for development: work on a small temporal slice before
        scaling to the full dataset.
        """
        mask = (self.timestamps >= start_ts) & (self.timestamps <= end_ts)

        src = torch.from_numpy(np.array(self.sources[mask], dtype=np.int64))
        dst = torch.from_numpy(np.array(self.destinations[mask], dtype=np.int64))
        t = torch.from_numpy(np.array(self.timestamps[mask], dtype=np.int64))
        msg = torch.from_numpy(np.array(self.edge_feat[mask], dtype=np.float32)).unsqueeze(1)

        return TemporalData(src=src, dst=dst, t=t, msg=msg)

    def get_monthly_subsample(self, year: int, month: int) -> TemporalData:
        """Get interactions for a single month."""
        from datetime import datetime, timezone
        start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
        if month == 12:
            end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp()) - 1
        else:
            end = int(datetime(year, month + 1, 1, tzinfo=timezone.utc).timestamp()) - 1
        return self.get_subsample(start, end)

    def statistics(self) -> dict:
        """Compute summary statistics."""
        from datetime import datetime, timezone

        t_min, t_max = self.time_range
        unique_pairs = len(set(zip(
            self.sources[:100_000].tolist(),
            self.destinations[:100_000].tolist()
        )))

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_node_features": self.num_features,
            "time_start": datetime.fromtimestamp(t_min, tz=timezone.utc).isoformat(),
            "time_end": datetime.fromtimestamp(t_max, tz=timezone.utc).isoformat(),
            "train_edges": int(self.train_mask.sum()),
            "val_edges": int(self.val_mask.sum()),
            "test_edges": int(self.test_mask.sum()),
        }

    def __repr__(self) -> str:
        stats = self.statistics()
        return (
            f"TemporalFlightGraph(\n"
            f"  nodes={stats['num_nodes']:,}, edges={stats['num_edges']:,}\n"
            f"  node_features={stats['num_node_features']}\n"
            f"  time: {stats['time_start'][:10]} to {stats['time_end'][:10]}\n"
            f"  train={stats['train_edges']:,}, val={stats['val_edges']:,}, test={stats['test_edges']:,}\n"
            f")"
        )
