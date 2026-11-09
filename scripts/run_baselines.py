"""
Run and evaluate all baselines on the validation set.

Evaluates each baseline separately to stay within RAM limits (~2.8 GB free).
Uses TGB's One-vs-Many MRR protocol with pre-generated negative samples.
"""

import sys, os, time, json, gc, pickle
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from collections import defaultdict
from tgb.linkproppred.evaluate import Evaluator


def load_data():
    """Load preprocessed arrays (memory-mapped) and compute train boundary."""
    d = "data/processed"
    sources = np.load(os.path.join(d, "sources.npy"), mmap_mode="r")
    destinations = np.load(os.path.join(d, "destinations.npy"), mmap_mode="r")
    timestamps = np.load(os.path.join(d, "timestamps.npy"), mmap_mode="r")
    train_mask = np.load(os.path.join(d, "train_mask.npy"), mmap_mode="r")
    train_end = int(np.sum(train_mask))
    return {
        "sources": sources,
        "destinations": destinations,
        "timestamps": timestamps,
        "train_end": train_end,
    }


def evaluate_mrr(pos_scores, neg_scores):
    """Compute MRR using TGB's evaluator."""
    evaluator = Evaluator(name="tgbl-flight")
    return evaluator.eval({
        "y_pred_pos": np.array(pos_scores, dtype=np.float32),
        "y_pred_neg": np.array(neg_scores, dtype=np.float32),
        "eval_metric": ["mrr"],
    })


# ---- Random Baseline ----

def eval_random(data, val_ns):
    print("\n=== Random Baseline ===", flush=True)
    rng = np.random.RandomState(42)
    n = len(val_ns)
    pos_scores = rng.random(n).astype(np.float32)
    neg_scores = rng.random((n, 20)).astype(np.float32)
    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f}", flush=True)
    return result


# ---- Historical Frequency ----

def eval_hist_freq(data, val_ns):
    print("\n=== Historical Frequency ===", flush=True)
    t0 = time.time()

    # Build pair counts in chunks to limit RAM usage
    te = data["train_end"]
    print(f"  Building pair counts from {te:,} training edges (chunked)...", flush=True)
    pair_counts = defaultdict(int)
    chunk = 5_000_000
    for start in range(0, te, chunk):
        end = min(start + chunk, te)
        src_c = np.array(data["sources"][start:end], dtype=np.int64)
        dst_c = np.array(data["destinations"][start:end], dtype=np.int64)
        keys = src_c * 20000 + dst_c
        unique_keys, counts = np.unique(keys, return_counts=True)
        for k, c in zip(unique_keys, counts):
            pair_counts[int(k)] += int(c)
    pair_counts = dict(pair_counts)
    print(f"  {len(pair_counts):,} unique pairs in {time.time()-t0:.1f}s", flush=True)

    # Evaluate
    print("  Scoring val edges...", flush=True)
    pos_scores = []
    neg_scores = []
    for i, ((s, d, t), negs) in enumerate(val_ns.items()):
        s, d = int(s), int(d)
        pos_scores.append(pair_counts.get(s * 20000 + d, 0))
        neg_row = [pair_counts.get(s * 20000 + int(nd), 0) for nd in negs]
        neg_scores.append(neg_row)
        if i % 1_000_000 == 0 and i > 0:
            print(f"    {i:,}/{len(val_ns):,}", flush=True)

    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f} ({time.time()-t0:.1f}s total)", flush=True)

    del pair_counts
    gc.collect()
    return result


# ---- Recency ----

def eval_recency(data, val_ns):
    print("\n=== Recency Baseline ===", flush=True)
    t0 = time.time()

    # Build last-seen map in chunks (chronological, last write wins)
    te = data["train_end"]
    print(f"  Building recency map from {te:,} training edges (chunked)...", flush=True)
    last_seen = {}
    chunk = 5_000_000
    for start in range(0, te, chunk):
        end = min(start + chunk, te)
        src_c = np.array(data["sources"][start:end], dtype=np.int64)
        dst_c = np.array(data["destinations"][start:end], dtype=np.int64)
        ts_c = np.array(data["timestamps"][start:end], dtype=np.int64)
        for s, d, t in zip(src_c.tolist(), dst_c.tolist(), ts_c.tolist()):
            last_seen[s * 20000 + d] = t
    print(f"  {len(last_seen):,} pairs in {time.time()-t0:.1f}s", flush=True)

    # Evaluate
    print("  Scoring val edges...", flush=True)
    pos_scores = []
    neg_scores = []

    def recency_score(src, dst, t):
        key = src * 20000 + dst
        lt = last_seen.get(key)
        if lt is None:
            return 0.0
        dt = t - lt
        if dt <= 0:
            return 1.0
        return 1.0 / dt

    for i, ((s, d, t), negs) in enumerate(val_ns.items()):
        s, d, t = int(s), int(d), int(t)
        pos_scores.append(recency_score(s, d, t))
        neg_row = [recency_score(s, int(nd), t) for nd in negs]
        neg_scores.append(neg_row)
        if i % 1_000_000 == 0 and i > 0:
            print(f"    {i:,}/{len(val_ns):,}", flush=True)

    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f} ({time.time()-t0:.1f}s total)", flush=True)

    del last_seen
    gc.collect()
    return result


# ---- EdgeBank (unlimited memory) ----

def eval_edgebank(data, val_ns):
    print("\n=== EdgeBank (unlimited) ===", flush=True)
    t0 = time.time()

    # Build edge memory in chunks
    te = data["train_end"]
    print(f"  Building edge memory from {te:,} training edges (chunked)...", flush=True)
    memory = set()
    chunk = 5_000_000
    for start in range(0, te, chunk):
        end = min(start + chunk, te)
        src_c = np.array(data["sources"][start:end], dtype=np.int64)
        dst_c = np.array(data["destinations"][start:end], dtype=np.int64)
        keys = src_c * 20000 + dst_c
        memory.update(np.unique(keys).tolist())
    print(f"  {len(memory):,} unique pairs in memory in {time.time()-t0:.1f}s", flush=True)

    # Evaluate
    print("  Scoring val edges...", flush=True)
    pos_scores = []
    neg_scores = []
    for i, ((s, d, t), negs) in enumerate(val_ns.items()):
        s, d = int(s), int(d)
        pk = s * 20000 + d
        pos_scores.append(1.0 if pk in memory else 0.0)
        neg_row = [1.0 if (s * 20000 + int(nd)) in memory else 0.0 for nd in negs]
        neg_scores.append(neg_row)
        if i % 1_000_000 == 0 and i > 0:
            print(f"    {i:,}/{len(val_ns):,}", flush=True)

    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f} ({time.time()-t0:.1f}s total)", flush=True)

    del memory
    gc.collect()
    return result


def main():
    print("Loading preprocessed data...", flush=True)
    data = load_data()

    print("Loading validation negative samples (~1.2 GB)...", flush=True)
    t0 = time.time()
    val_ns = pickle.load(open("datasets/tgbl-flight/tgbl-flight_val_ns_v2.pkl", "rb"))
    print(f"  Loaded {len(val_ns):,} entries in {time.time()-t0:.1f}s", flush=True)

    results = {}

    results["Random"] = eval_random(data, val_ns)
    results["HistFreq"] = eval_hist_freq(data, val_ns)
    results["Recency"] = eval_recency(data, val_ns)
    results["EdgeBank"] = eval_edgebank(data, val_ns)

    # Summary
    print("\n" + "=" * 50, flush=True)
    print("BASELINE RESULTS (Validation Set)", flush=True)
    print("=" * 50, flush=True)
    print(f"{'Baseline':<20} {'MRR':>8} {'Hits@10':>8}", flush=True)
    print("-" * 38, flush=True)
    for name, res in results.items():
        print(f"{name:<20} {res['mrr']:>8.4f} {res['hits@10']:>8.4f}", flush=True)

    # Save results (convert numpy floats to Python floats)
    os.makedirs("results", exist_ok=True)
    serializable = {
        k: {mk: float(mv) for mk, mv in v.items()} for k, v in results.items()
    }
    with open("results/baseline_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print("\nResults saved to results/baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
