"""
Training and evaluation loop for TGN on tgbl-flight.

Training strategy for CPU-limited hardware:
  - Chronological batches via TemporalDataLoader
  - Random negative sampling during training (1 negative per positive)
  - TGB protocol for evaluation (pre-generated negatives, MRR metric)
  - Subsample support for development (train on N months)
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import TemporalDataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.tgn import build_tgn
from src.graph.temporal_graph import TemporalFlightGraph


def train_epoch(memory, gnn, link_pred, neighbor_loader, loader, optimizer,
                num_nodes, device="cpu"):
    """Train one epoch over chronological batches.

    For each batch of interactions (src, dst, t, msg):
      1. Get memory embeddings for src and dst
      2. Score positive edges and random negative edges
      3. Compute BPR loss
      4. Update model parameters
      5. Update memory with new interactions
      6. Update neighbor loader
    """
    memory.train()
    link_pred.train()
    if gnn is not None:
        gnn.train()

    memory.reset_state()
    neighbor_loader.reset_state()

    total_loss = 0
    num_edges = 0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Random negative destinations (same src, random dst)
        neg_dst = torch.randint(0, num_nodes, (src.size(0),), dtype=torch.long)

        # Get current memory for all involved nodes
        n_id = torch.cat([src, dst, neg_dst]).unique()
        z, last_update = memory(n_id)

        # Build embedding lookup
        assoc = torch.empty(num_nodes, dtype=torch.long)
        assoc[n_id] = torch.arange(n_id.size(0))

        if gnn is not None:
            n_id_gnn, edge_index, e_id = neighbor_loader(n_id)
            z_gnn, _ = memory(n_id_gnn)
            # For GNN we need edge features from the temporal data
            # Simplified: use zero edge features for neighbor edges
            edge_msg = torch.zeros(edge_index.size(1), msg.size(1))
            edge_t = torch.zeros(edge_index.size(1), dtype=torch.long)
            z_gnn = gnn(z_gnn, last_update, edge_index, edge_t, edge_msg)
            # Map back to original node indices
            assoc_gnn = torch.empty(num_nodes, dtype=torch.long)
            assoc_gnn[n_id_gnn] = torch.arange(n_id_gnn.size(0))
            z_src = z_gnn[assoc_gnn[src]]
            z_dst = z_gnn[assoc_gnn[dst]]
            z_neg = z_gnn[assoc_gnn[neg_dst]]
        else:
            z_src = z[assoc[src]]
            z_dst = z[assoc[dst]]
            z_neg = z[assoc[neg_dst]]

        # Score positive and negative edges
        pos_score = link_pred(z_src, z_dst)
        neg_score = link_pred(z_src, z_neg)

        # BPR loss: positive edges should score higher than negatives
        loss = nn.functional.softplus(neg_score - pos_score).mean()

        loss.backward()
        optimizer.step()

        # Update memory (detached from gradient graph)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

        total_loss += loss.detach().item() * src.size(0)
        num_edges += src.size(0)
        num_batches += 1

        # Detach memory to avoid backprop through time across batches
        memory.detach()

        if num_batches % 1000 == 0:
            avg = total_loss / num_edges if num_edges > 0 else 0
            print(f"    batch {num_batches:,} | edges {num_edges:,} | "
                  f"loss {avg:.4f}", flush=True)

    return total_loss / num_edges if num_edges > 0 else 0


@torch.no_grad()
def evaluate(memory, gnn, link_pred, neighbor_loader, loader, neg_samples,
             num_nodes, max_edges=0, device="cpu"):
    """Evaluate using TGB's One-vs-Many MRR protocol.

    For each val/test edge (src, dst, t):
      - Score the true destination and 20 pre-generated negatives (batched)
      - Compute MRR from the rank of the true destination

    Args:
        max_edges: if > 0, only score this many edges (for dev speed).
                   Memory is still updated for all processed batches.
    """
    from tgb.linkproppred.evaluate import Evaluator

    memory.eval()
    link_pred.eval()
    if gnn is not None:
        gnn.eval()

    evaluator = Evaluator(name="tgbl-flight")

    all_pos_scores = []
    all_neg_scores = []
    skipped = 0
    evaluated = 0
    num_batches = 0
    done_scoring = False

    for batch in loader:
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        if not done_scoring:
            n_id = torch.cat([src, dst]).unique()
            z, last_update = memory(n_id)
            assoc = torch.empty(num_nodes, dtype=torch.long)
            assoc[n_id] = torch.arange(n_id.size(0))

            for i in range(src.size(0)):
                if max_edges > 0 and evaluated >= max_edges:
                    done_scoring = True
                    break

                s, d, ts = int(src[i]), int(dst[i]), int(t[i])
                key = (np.int64(s), np.int64(d), np.int64(ts))
                if key not in neg_samples:
                    skipped += 1
                    continue

                neg_dsts = neg_samples[key]

                z_s = z[assoc[src[i]]].unsqueeze(0)
                z_d = z[assoc[dst[i]]].unsqueeze(0)
                pos_score = link_pred(z_s, z_d).item()

                # Batch all 20 negatives at once
                neg_tensor = torch.tensor(neg_dsts, dtype=torch.long)
                z_neg, _ = memory(neg_tensor)
                z_s_rep = z_s.expand(len(neg_dsts), -1)
                neg_scores_batch = link_pred(z_s_rep, z_neg).tolist()

                all_pos_scores.append(pos_score)
                all_neg_scores.append(neg_scores_batch)
                evaluated += 1

        # Always update memory for chronological consistency
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        num_batches += 1

        if num_batches % 5000 == 0:
            print(f"    eval batch {num_batches:,} | scored {evaluated:,}", flush=True)

        # If we're done scoring and don't need full memory walk, break
        if done_scoring:
            break

    if skipped > 0:
        print(f"  Skipped {skipped:,} edges (not in neg samples)", flush=True)
    print(f"  Scored {evaluated:,} edges across {num_batches:,} batches", flush=True)

    y_pos = np.array(all_pos_scores, dtype=np.float32)
    y_neg = np.array(all_neg_scores, dtype=np.float32)

    result = evaluator.eval({
        "y_pred_pos": y_pos,
        "y_pred_neg": y_neg,
        "eval_metric": ["mrr"],
    })
    return {"mrr": float(result["mrr"]), "hits@10": float(result["hits@10"])}


def run_training(
    train_months: int = 3,
    epochs: int = 2,
    batch_size: int = 200,
    lr: float = 0.0001,
    memory_dim: int = 100,
    time_dim: int = 100,
    use_gnn: bool = False,
    max_eval_edges: int = 0,
    output_dir: str = "results",
):
    """Full training pipeline.

    Args:
        train_months: number of months before val split to use for training
                      (0 = use all training data)
        epochs: number of training epochs
        batch_size: temporal batch size
        lr: learning rate
        memory_dim: TGN memory dimension
        time_dim: time encoding dimension
        use_gnn: whether to use graph attention embedding
        max_eval_edges: if > 0, subsample validation to this many edges
                        (0 = evaluate all, use for final results only)
        output_dir: where to save results
    """
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60, flush=True)
    print("TGN Training for tgbl-flight", flush=True)
    print("=" * 60, flush=True)

    # Load graph
    print("\nLoading temporal graph...", flush=True)
    g = TemporalFlightGraph()
    print(f"  {g}", flush=True)

    # Prepare training data
    if train_months > 0:
        # Use last N months of training data
        from datetime import datetime, timezone
        timestamps = np.array(g.timestamps)
        val_time = np.quantile(timestamps, 0.70)

        # Go back train_months months from val_time
        from datetime import timedelta
        val_dt = datetime.fromtimestamp(val_time, tz=timezone.utc)
        train_start_dt = val_dt - timedelta(days=30 * train_months)
        train_start_ts = int(train_start_dt.timestamp())

        train_data = g.get_subsample(train_start_ts, int(val_time))
        print(f"  Training subsample: last {train_months} months, "
              f"{train_data.src.size(0):,} edges", flush=True)
    else:
        train_data = g.get_split("train")
        print(f"  Full training set: {train_data.src.size(0):,} edges", flush=True)

    val_data = g.get_split("val")
    print(f"  Validation set: {val_data.src.size(0):,} edges", flush=True)

    # Build model
    print(f"\nBuilding TGN (memory_dim={memory_dim}, time_dim={time_dim}, "
          f"gnn={use_gnn})...", flush=True)
    components = build_tgn(
        num_nodes=g.num_nodes,
        raw_msg_dim=1,
        memory_dim=memory_dim,
        time_dim=time_dim,
        use_gnn=use_gnn,
    )
    memory = components["memory"]
    gnn = components["gnn"]
    link_pred = components["link_pred"]
    neighbor_loader = components["neighbor_loader"]

    # Count parameters
    n_params = sum(p.numel() for p in memory.parameters())
    n_params += sum(p.numel() for p in link_pred.parameters())
    if gnn is not None:
        n_params += sum(p.numel() for p in gnn.parameters())
    print(f"  Total parameters: {n_params:,}", flush=True)

    # Optimizer
    params = list(memory.parameters()) + list(link_pred.parameters())
    if gnn is not None:
        params += list(gnn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Data loaders
    train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
    val_loader = TemporalDataLoader(val_data, batch_size=batch_size)

    # Load negative samples for evaluation
    print("\nLoading validation negative samples...", flush=True)
    t0 = time.time()
    with open("datasets/tgbl-flight/tgbl-flight_val_ns_v2.pkl", "rb") as f:
        val_ns = pickle.load(f)
    print(f"  Loaded {len(val_ns):,} entries in {time.time()-t0:.1f}s", flush=True)

    # Training loop
    best_mrr = 0
    results_log = []

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---", flush=True)

        t_start = time.time()
        train_loss = train_epoch(
            memory, gnn, link_pred, neighbor_loader,
            train_loader, optimizer, g.num_nodes,
        )
        t_train = time.time() - t_start
        print(f"  Train loss: {train_loss:.4f} ({t_train:.1f}s)", flush=True)

        # Evaluate
        print("  Evaluating on validation set...", flush=True)
        t_start = time.time()
        val_result = evaluate(
            memory, gnn, link_pred, neighbor_loader,
            val_loader, val_ns, g.num_nodes, max_edges=max_eval_edges,
        )
        t_eval = time.time() - t_start
        print(f"  Val MRR: {val_result['mrr']:.4f}, "
              f"Hits@10: {val_result['hits@10']:.4f} ({t_eval:.1f}s)", flush=True)

        results_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mrr": val_result["mrr"],
            "val_hits10": val_result["hits@10"],
            "train_time": t_train,
            "eval_time": t_eval,
        })

        if val_result["mrr"] > best_mrr:
            best_mrr = val_result["mrr"]
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                "memory_state": memory.state_dict(),
                "link_pred_state": link_pred.state_dict(),
                "gnn_state": gnn.state_dict() if gnn else None,
                "epoch": epoch,
                "mrr": best_mrr,
            }, os.path.join(output_dir, "tgn_best.pt"))
            print(f"  New best MRR! Saved model.", flush=True)

    # Save training log
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tgn_training_log.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"Training complete! Best val MRR: {best_mrr:.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    return results_log


if __name__ == "__main__":
    run_training(train_months=3, epochs=2, batch_size=200, memory_dim=100,
                 time_dim=100, use_gnn=False, max_eval_edges=100000)
