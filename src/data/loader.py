"""
Data loader for the tgbl-flight dataset from TGB (Temporal Graph Benchmark).

This module wraps the TGB loading utilities to provide a consistent interface
for accessing the flight interaction dataset.
"""

import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset


def load_tgbl_flight(root: str = "datasets") -> dict:
    """Load the tgbl-flight dataset using TGB's NumPy interface.

    Returns a dict with keys: sources, destinations, timestamps, edge_feat,
    node_feat, train_mask, val_mask, test_mask, eval_metric, num_nodes.
    """
    dataset = LinkPropPredDataset(name="tgbl-flight", root=root, preprocess=True)
    data = dataset.full_data

    result = {
        "sources": data["sources"],
        "destinations": data["destinations"],
        "timestamps": data["timestamps"],
        "edge_feat": data.get("edge_feat"),
        "edge_label": data.get("edge_label"),
        "num_nodes": dataset.num_nodes if hasattr(dataset, "num_nodes") else None,
        "eval_metric": dataset.eval_metric,
        "train_mask": dataset.train_mask,
        "val_mask": dataset.val_mask,
        "test_mask": dataset.test_mask,
    }

    if hasattr(dataset, "node_feat") and dataset.node_feat is not None:
        result["node_feat"] = dataset.node_feat
    else:
        result["node_feat"] = None

    return result
