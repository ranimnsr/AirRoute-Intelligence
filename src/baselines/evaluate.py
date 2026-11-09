"""
Evaluation harness for temporal link prediction baselines.

Implements the One-vs-Many MRR evaluation protocol from TGB:
  For each validation/test edge (src, dst, t):
    - The baseline scores the true destination (positive)
    - It also scores 20 negative destinations (from TGB's pre-generated set)
    - MRR = mean of 1/rank(positive among all 21 candidates)

This module handles loading negative samples, calling each baseline's
scoring function, and computing MRR via TGB's Evaluator.
"""

import os
import pickle
import numpy as np
from tgb.linkproppred.evaluate import Evaluator


def load_negative_samples(path: str) -> dict:
    """Load TGB pre-generated negative samples."""
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_baseline(
    baseline,
    sources: np.ndarray,
    destinations: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
    neg_samples: dict,
    dataset_name: str = "tgbl-flight",
    batch_size: int = 50000,
) -> dict:
    """Evaluate a baseline on a split using TGB's MRR protocol.

    Args:
        baseline: object with a score_batch(src, dst_candidates, t) method
        sources, destinations, timestamps: full preprocessed arrays
        mask: boolean mask for the split (val or test)
        neg_samples: dict from load_negative_samples()
        dataset_name: TGB dataset name for the Evaluator
        batch_size: number of edges to process at once

    Returns:
        dict with 'mrr' and 'hits@10' metrics
    """
    evaluator = Evaluator(name=dataset_name)

    val_src = sources[mask]
    val_dst = destinations[mask]
    val_ts = timestamps[mask]

    all_pos_scores = []
    all_neg_scores = []
    skipped = 0

    n = len(val_src)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_src = val_src[start:end]
        batch_dst = val_dst[start:end]
        batch_ts = val_ts[start:end]

        for j in range(end - start):
            s = int(batch_src[j])
            d = int(batch_dst[j])
            t = int(batch_ts[j])

            key = (np.int64(s), np.int64(d), np.int64(t))
            if key not in neg_samples:
                skipped += 1
                continue

            neg_dsts = neg_samples[key]

            pos_score = baseline.score(s, d, t)
            neg_scores = np.array([baseline.score(s, int(nd), t) for nd in neg_dsts])

            all_pos_scores.append(pos_score)
            all_neg_scores.append(neg_scores)

        if (start // batch_size) % 20 == 0:
            print(f"  Evaluated {end:,}/{n:,} edges...", flush=True)

    if skipped > 0:
        print(f"  Skipped {skipped:,} edges (not in neg samples)", flush=True)

    y_pred_pos = np.array(all_pos_scores, dtype=np.float32)
    y_pred_neg = np.stack(all_neg_scores).astype(np.float32)

    result = evaluator.eval({
        "y_pred_pos": y_pred_pos,
        "y_pred_neg": y_pred_neg,
        "eval_metric": ["mrr"],
    })

    return {
        "mrr": float(result["mrr"]),
        "hits@10": float(result["hits@10"]),
        "num_evaluated": len(all_pos_scores),
        "num_skipped": skipped,
    }
