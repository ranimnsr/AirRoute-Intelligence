"""
Phase 5: Evaluate all models on the TEST set.

Protocol:
  - Baselines: build statistics from train+val data, evaluate on test
  - TGN: load trained parameters, warm up memory on train+val, evaluate on test
  - All use TGB's One-vs-Many MRR with pre-generated test negative samples

Run baselines first (less RAM), then TGN separately.
"""

import sys, os, time, json, gc, pickle
sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from collections import defaultdict
from tgb.linkproppred.evaluate import Evaluator


def load_data():
    d = "data/processed"
    sources = np.load(os.path.join(d, "sources.npy"), mmap_mode="r")
    destinations = np.load(os.path.join(d, "destinations.npy"), mmap_mode="r")
    timestamps = np.load(os.path.join(d, "timestamps.npy"), mmap_mode="r")
    train_mask = np.load(os.path.join(d, "train_mask.npy"), mmap_mode="r")
    val_mask = np.load(os.path.join(d, "val_mask.npy"), mmap_mode="r")
    train_end = int(np.sum(train_mask))
    val_end = train_end + int(np.sum(val_mask))
    return {
        "sources": sources, "destinations": destinations,
        "timestamps": timestamps,
        "train_end": train_end, "val_end": val_end,
    }


def evaluate_mrr(pos_scores, neg_scores):
    evaluator = Evaluator(name="tgbl-flight")
    return evaluator.eval({
        "y_pred_pos": np.array(pos_scores, dtype=np.float32),
        "y_pred_neg": np.array(neg_scores, dtype=np.float32),
        "eval_metric": ["mrr"],
    })


# ============================================================
# BASELINES ON TEST SET
# ============================================================

def eval_random_test(test_ns):
    print("\n=== Random (Test) ===", flush=True)
    rng = np.random.RandomState(42)
    n = len(test_ns)
    result = evaluate_mrr(rng.random(n).astype(np.float32),
                          rng.random((n, 20)).astype(np.float32))
    print(f"  MRR: {result['mrr']:.4f}, Hits@10: {result['hits@10']:.4f}", flush=True)
    return result


def eval_hist_freq_test(data, test_ns):
    """HistFreq using train+val edge counts."""
    print("\n=== HistFreq (Test) ===", flush=True)
    t0 = time.time()
    ve = data["val_end"]
    print(f"  Building pair counts from {ve:,} train+val edges...", flush=True)
    pair_counts = defaultdict(int)
    chunk = 5_000_000
    for start in range(0, ve, chunk):
        end = min(start + chunk, ve)
        src_c = np.array(data["sources"][start:end], dtype=np.int64)
        dst_c = np.array(data["destinations"][start:end], dtype=np.int64)
        keys = src_c * 20000 + dst_c
        unique_keys, counts = np.unique(keys, return_counts=True)
        for k, c in zip(unique_keys, counts):
            pair_counts[int(k)] += int(c)
    pair_counts = dict(pair_counts)
    print(f"  {len(pair_counts):,} unique pairs in {time.time()-t0:.1f}s", flush=True)

    print("  Scoring test edges...", flush=True)
    pos_scores, neg_scores = [], []
    for i, ((s, d, t), negs) in enumerate(test_ns.items()):
        s, d = int(s), int(d)
        pos_scores.append(pair_counts.get(s * 20000 + d, 0))
        neg_scores.append([pair_counts.get(s * 20000 + int(nd), 0) for nd in negs])
        if i % 1_000_000 == 0 and i > 0:
            print(f"    {i:,}/{len(test_ns):,}", flush=True)

    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f}, Hits@10: {result['hits@10']:.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    del pair_counts; gc.collect()
    return result


def eval_recency_test(data, test_ns):
    """Recency using last-seen from train+val."""
    print("\n=== Recency (Test) ===", flush=True)
    t0 = time.time()
    ve = data["val_end"]
    print(f"  Building recency map from {ve:,} train+val edges...", flush=True)
    last_seen = {}
    chunk = 5_000_000
    for start in range(0, ve, chunk):
        end = min(start + chunk, ve)
        src_c = np.array(data["sources"][start:end], dtype=np.int64)
        dst_c = np.array(data["destinations"][start:end], dtype=np.int64)
        ts_c = np.array(data["timestamps"][start:end], dtype=np.int64)
        for s, d, t in zip(src_c.tolist(), dst_c.tolist(), ts_c.tolist()):
            last_seen[s * 20000 + d] = t
    print(f"  {len(last_seen):,} pairs in {time.time()-t0:.1f}s", flush=True)

    print("  Scoring test edges...", flush=True)
    pos_scores, neg_scores = [], []
    for i, ((s, d, t), negs) in enumerate(test_ns.items()):
        s, d, t = int(s), int(d), int(t)
        def rscore(src, dst, ts):
            lt = last_seen.get(src * 20000 + dst)
            if lt is None: return 0.0
            dt = ts - lt
            return 1.0 / dt if dt > 0 else 1.0
        pos_scores.append(rscore(s, d, t))
        neg_scores.append([rscore(s, int(nd), t) for nd in negs])
        if i % 1_000_000 == 0 and i > 0:
            print(f"    {i:,}/{len(test_ns):,}", flush=True)

    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f}, Hits@10: {result['hits@10']:.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    del last_seen; gc.collect()
    return result


def eval_edgebank_test(data, test_ns):
    """EdgeBank with train+val edges."""
    print("\n=== EdgeBank (Test) ===", flush=True)
    t0 = time.time()
    ve = data["val_end"]
    print(f"  Building edge memory from {ve:,} train+val edges...", flush=True)
    memory = set()
    chunk = 5_000_000
    for start in range(0, ve, chunk):
        end = min(start + chunk, ve)
        src_c = np.array(data["sources"][start:end], dtype=np.int64)
        dst_c = np.array(data["destinations"][start:end], dtype=np.int64)
        memory.update(np.unique(src_c * 20000 + dst_c).tolist())
    print(f"  {len(memory):,} unique pairs in {time.time()-t0:.1f}s", flush=True)

    print("  Scoring test edges...", flush=True)
    pos_scores, neg_scores = [], []
    for i, ((s, d, t), negs) in enumerate(test_ns.items()):
        s, d = int(s), int(d)
        pos_scores.append(1.0 if (s * 20000 + d) in memory else 0.0)
        neg_scores.append([1.0 if (s * 20000 + int(nd)) in memory else 0.0 for nd in negs])
        if i % 1_000_000 == 0 and i > 0:
            print(f"    {i:,}/{len(test_ns):,}", flush=True)

    result = evaluate_mrr(pos_scores, neg_scores)
    print(f"  MRR: {result['mrr']:.4f}, Hits@10: {result['hits@10']:.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    del memory; gc.collect()
    return result


# ============================================================
# TGN ON TEST SET
# ============================================================

def eval_tgn_test(max_eval_edges=100000):
    """Evaluate TGN on test set.

    Steps:
      1. Build model, load trained parameters
      2. Process train subsample (3 months) through memory
      3. Process ALL val edges through memory
      4. Score test edges using test negative samples
    """
    import torch
    from torch_geometric.loader import TemporalDataLoader
    from src.models.tgn import build_tgn
    from src.graph.temporal_graph import TemporalFlightGraph

    print("\n=== TGN (Test) ===", flush=True)
    print("Loading graph and model...", flush=True)
    t0 = time.time()

    g = TemporalFlightGraph()

    components = build_tgn(
        num_nodes=g.num_nodes, raw_msg_dim=1,
        memory_dim=100, time_dim=100, use_gnn=False,
    )
    memory = components["memory"]
    link_pred = components["link_pred"]
    neighbor_loader = components["neighbor_loader"]

    # Load trained parameters
    ckpt = torch.load("results/tgn_best.pt", map_location="cpu", weights_only=False)
    memory.load_state_dict(ckpt["memory_state"])
    link_pred.load_state_dict(ckpt["link_pred_state"])
    print(f"  Model loaded (epoch {ckpt['epoch']}, val MRR {ckpt['mrr']:.4f})", flush=True)

    memory.eval()
    link_pred.eval()

    # --- Step 1: Warm up memory on training subsample ---
    print("\nWarming up memory on train subsample (3 months)...", flush=True)
    from datetime import datetime, timezone, timedelta
    timestamps = np.array(g.timestamps)
    val_time = np.quantile(timestamps, 0.70)
    val_dt = datetime.fromtimestamp(val_time, tz=timezone.utc)
    train_start_dt = val_dt - timedelta(days=90)
    train_data = g.get_subsample(int(train_start_dt.timestamp()), int(val_time))
    print(f"  Train subsample: {train_data.src.size(0):,} edges", flush=True)

    train_loader = TemporalDataLoader(train_data, batch_size=200)
    memory.reset_state()
    neighbor_loader.reset_state()

    n_batch = 0
    with torch.no_grad():
        for batch in train_loader:
            n_id = torch.cat([batch.src, batch.dst]).unique()
            memory(n_id)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
            n_batch += 1
            if n_batch % 5000 == 0:
                print(f"    train batch {n_batch:,}", flush=True)
    print(f"  Train warmup done: {n_batch:,} batches ({time.time()-t0:.0f}s)", flush=True)

    del train_data, train_loader
    gc.collect()

    # --- Step 2: Warm up memory on ALL val edges ---
    print("\nWarming up memory on validation set...", flush=True)
    t1 = time.time()
    val_data = g.get_split("val")
    print(f"  Val set: {val_data.src.size(0):,} edges", flush=True)
    val_loader = TemporalDataLoader(val_data, batch_size=200)

    n_batch = 0
    with torch.no_grad():
        for batch in val_loader:
            n_id = torch.cat([batch.src, batch.dst]).unique()
            memory(n_id)
            memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
            neighbor_loader.insert(batch.src, batch.dst)
            n_batch += 1
            if n_batch % 10000 == 0:
                print(f"    val batch {n_batch:,}", flush=True)
    print(f"  Val warmup done: {n_batch:,} batches ({time.time()-t1:.0f}s)", flush=True)

    del val_data, val_loader
    gc.collect()

    # --- Step 3: Evaluate on test ---
    print("\nLoading test negative samples...", flush=True)
    t2 = time.time()
    with open("datasets/tgbl-flight/tgbl-flight_test_ns_v2.pkl", "rb") as f:
        test_ns = pickle.load(f)
    print(f"  Loaded {len(test_ns):,} entries in {time.time()-t2:.1f}s", flush=True)

    print(f"Evaluating on test set (max {max_eval_edges:,} edges)...", flush=True)
    test_data = g.get_split("test")
    test_loader = TemporalDataLoader(test_data, batch_size=200)

    evaluator = Evaluator(name="tgbl-flight")
    all_pos, all_neg = [], []
    evaluated = 0
    skipped = 0
    n_batch = 0
    done_scoring = False

    with torch.no_grad():
        for batch in test_loader:
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

            if not done_scoring:
                n_id = torch.cat([src, dst]).unique()
                z, last_update = memory(n_id)
                assoc = torch.empty(g.num_nodes, dtype=torch.long)
                assoc[n_id] = torch.arange(n_id.size(0))

                for i in range(src.size(0)):
                    if max_eval_edges > 0 and evaluated >= max_eval_edges:
                        done_scoring = True
                        break

                    s, d, ts = int(src[i]), int(dst[i]), int(t[i])
                    key = (np.int64(s), np.int64(d), np.int64(ts))
                    if key not in test_ns:
                        skipped += 1
                        continue

                    neg_dsts = test_ns[key]
                    z_s = z[assoc[src[i]]].unsqueeze(0)
                    z_d = z[assoc[dst[i]]].unsqueeze(0)
                    pos_score = link_pred(z_s, z_d).item()

                    neg_tensor = torch.tensor(neg_dsts, dtype=torch.long)
                    z_neg, _ = memory(neg_tensor)
                    z_s_rep = z_s.expand(len(neg_dsts), -1)
                    neg_scores_batch = link_pred(z_s_rep, z_neg).tolist()

                    all_pos.append(pos_score)
                    all_neg.append(neg_scores_batch)
                    evaluated += 1

            memory.update_state(src, dst, t, msg)
            neighbor_loader.insert(src, dst)
            n_batch += 1

            if n_batch % 5000 == 0:
                print(f"    test batch {n_batch:,} | scored {evaluated:,}", flush=True)

            if done_scoring:
                break

    if skipped > 0:
        print(f"  Skipped {skipped:,} edges (not in neg samples)", flush=True)
    print(f"  Scored {evaluated:,} edges in {time.time()-t2:.0f}s", flush=True)

    y_pos = np.array(all_pos, dtype=np.float32)
    y_neg = np.array(all_neg, dtype=np.float32)
    result = evaluator.eval({
        "y_pred_pos": y_pos, "y_pred_neg": y_neg, "eval_metric": ["mrr"],
    })
    total_time = time.time() - t0
    print(f"  Test MRR: {result['mrr']:.4f}, Hits@10: {result['hits@10']:.4f} "
          f"(total {total_time:.0f}s)", flush=True)
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60, flush=True)
    print("Phase 5: Test Set Evaluation", flush=True)
    print("=" * 60, flush=True)

    # --- Part 1: Baselines ---
    print("\n--- BASELINES ---", flush=True)
    data = load_data()

    print(f"\nLoading test negative samples...", flush=True)
    t0 = time.time()
    with open("datasets/tgbl-flight/tgbl-flight_test_ns_v2.pkl", "rb") as f:
        test_ns = pickle.load(f)
    print(f"  Loaded {len(test_ns):,} entries in {time.time()-t0:.1f}s", flush=True)

    test_results = {}
    test_results["Random"] = eval_random_test(test_ns)
    test_results["HistFreq"] = eval_hist_freq_test(data, test_ns)
    test_results["Recency"] = eval_recency_test(data, test_ns)
    test_results["EdgeBank"] = eval_edgebank_test(data, test_ns)

    # Free baseline data
    del test_ns, data
    gc.collect()

    # --- Part 2: TGN ---
    print("\n--- TGN ---", flush=True)
    tgn_result = eval_tgn_test(max_eval_edges=100000)
    test_results["TGN"] = tgn_result

    # --- Summary ---
    print("\n" + "=" * 60, flush=True)
    print("TEST SET RESULTS — All Models", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Model':<22} {'MRR':>8} {'Hits@10':>8}", flush=True)
    print("-" * 40, flush=True)
    for name, res in test_results.items():
        print(f"{name:<22} {res['mrr']:>8.4f} {res['hits@10']:>8.4f}", flush=True)

    # Save
    os.makedirs("results", exist_ok=True)
    serializable = {
        k: {mk: float(mv) for mk, mv in v.items()} for k, v in test_results.items()
    }
    with open("results/test_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to results/test_results.json", flush=True)


if __name__ == "__main__":
    main()
