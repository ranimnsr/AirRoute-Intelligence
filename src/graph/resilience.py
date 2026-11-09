"""
Network Resilience Analysis for tgbl-flight.

Graph analytics (NOT GNN) to study the robustness of the global flight network:
  - Build a weighted directed route graph from flight interactions
  - Compute centrality metrics (degree, betweenness, PageRank)
  - Detect communities (Louvain on undirected projection)
  - Simulate disruptions (sequential node removal)
"""

import os
import sys
import time
import json
import numpy as np
import networkx as nx
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def build_route_graph(sources, destinations, timestamps=None,
                      start_ts=None, end_ts=None):
    """Build a weighted directed graph from flight interactions.

    Each edge (u, v) has weight = number of flights on that route.
    Optionally filter to a time window.

    Returns:
        nx.DiGraph with 'weight' on edges and 'total_flights' on nodes
    """
    n = len(sources)

    route_counts = defaultdict(int)
    chunk = 5_000_000
    for start in range(0, n, chunk):
        end_idx = min(start + chunk, n)
        src_c = np.array(sources[start:end_idx], dtype=np.int32)
        dst_c = np.array(destinations[start:end_idx], dtype=np.int32)

        if timestamps is not None and (start_ts is not None or end_ts is not None):
            ts_c = np.array(timestamps[start:end_idx], dtype=np.int64)
            mask = np.ones(len(src_c), dtype=bool)
            if start_ts is not None:
                mask &= ts_c >= start_ts
            if end_ts is not None:
                mask &= ts_c < end_ts
            src_c = src_c[mask]
            dst_c = dst_c[mask]

        for s, d in zip(src_c.tolist(), dst_c.tolist()):
            route_counts[(s, d)] += 1

    G = nx.DiGraph()
    for (s, d), w in route_counts.items():
        G.add_edge(s, d, weight=w)

    # Node attribute: total flights (in + out)
    for node in G.nodes():
        total = sum(G[node][nbr]["weight"] for nbr in G.successors(node))
        total += sum(G[pred][node]["weight"] for pred in G.predecessors(node))
        G.nodes[node]["total_flights"] = total

    return G


def compute_centrality(G, top_k=20):
    """Compute centrality metrics for the route graph.

    Returns dict with:
      - degree_centrality: weighted in+out degree / max
      - betweenness_centrality: approximate betweenness (sampled)
      - pagerank: PageRank scores
      - top_airports: top-K by each metric
    """
    n = G.number_of_nodes()
    print(f"  Graph: {n:,} nodes, {G.number_of_edges():,} edges", flush=True)

    # Weighted degree centrality (normalized by max)
    print("  Computing degree centrality...", flush=True)
    w_degree = {}
    for node in G.nodes():
        out_w = sum(G[node][nbr]["weight"] for nbr in G.successors(node))
        in_w = sum(G[pred][node]["weight"] for pred in G.predecessors(node))
        w_degree[node] = out_w + in_w
    max_deg = max(w_degree.values()) if w_degree else 1
    degree_cent = {k: v / max_deg for k, v in w_degree.items()}

    # Betweenness centrality (approximate with k samples for speed)
    print("  Computing betweenness centrality (approximate)...", flush=True)
    k_samples = min(200, n)
    bc = nx.betweenness_centrality(G, k=k_samples, weight=None, normalized=True)

    # PageRank
    print("  Computing PageRank...", flush=True)
    pr = nx.pagerank(G, weight="weight", max_iter=100)

    # Top airports
    top_degree = sorted(degree_cent, key=degree_cent.get, reverse=True)[:top_k]
    top_bc = sorted(bc, key=bc.get, reverse=True)[:top_k]
    top_pr = sorted(pr, key=pr.get, reverse=True)[:top_k]

    return {
        "degree_centrality": degree_cent,
        "betweenness_centrality": bc,
        "pagerank": pr,
        "top_degree": top_degree,
        "top_betweenness": top_bc,
        "top_pagerank": top_pr,
    }


def detect_communities(G, resolution=1.0):
    """Detect communities using Louvain on the undirected projection.

    Returns dict mapping node → community_id, and community sizes.
    """
    print("  Detecting communities (Louvain)...", flush=True)
    G_undir = G.to_undirected()
    communities = nx.community.louvain_communities(
        G_undir, weight="weight", resolution=resolution, seed=42,
    )

    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    comm_sizes = defaultdict(int)
    for c in node_to_comm.values():
        comm_sizes[c] += 1

    print(f"  Found {len(communities)} communities", flush=True)
    top5 = sorted(comm_sizes.items(), key=lambda x: -x[1])[:5]
    for cid, size in top5:
        print(f"    Community {cid}: {size} airports", flush=True)

    return node_to_comm, dict(comm_sizes)


def simulate_disruption(G, removal_order, max_remove=50):
    """Simulate sequential node removal and measure network impact.

    For each step (removing 1 more node):
      - Number of strongly connected components
      - Size of largest strongly connected component
      - Fraction of nodes in largest SCC (connectivity)

    Args:
        G: the original directed graph
        removal_order: list of node IDs to remove (most critical first)
        max_remove: how many nodes to remove

    Returns list of dicts with metrics at each step.
    """
    print(f"  Simulating disruption (removing up to {max_remove} nodes)...", flush=True)
    G_sim = G.copy()
    original_n = G_sim.number_of_nodes()

    results = []

    # Baseline (no removal)
    sccs = list(nx.strongly_connected_components(G_sim))
    largest_scc = max(len(c) for c in sccs)
    results.append({
        "removed": 0,
        "num_scc": len(sccs),
        "largest_scc": largest_scc,
        "connectivity": largest_scc / original_n,
    })

    for i, node in enumerate(removal_order[:max_remove]):
        if node in G_sim:
            G_sim.remove_node(node)

        remaining = G_sim.number_of_nodes()
        if remaining == 0:
            results.append({
                "removed": i + 1,
                "num_scc": 0,
                "largest_scc": 0,
                "connectivity": 0,
            })
            break

        sccs = list(nx.strongly_connected_components(G_sim))
        largest_scc = max(len(c) for c in sccs) if sccs else 0
        results.append({
            "removed": i + 1,
            "num_scc": len(sccs),
            "largest_scc": largest_scc,
            "connectivity": largest_scc / original_n,
        })

        if (i + 1) % 10 == 0:
            print(f"    Removed {i+1}: {len(sccs)} SCCs, "
                  f"largest={largest_scc} ({largest_scc/original_n:.1%})", flush=True)

    return results


def run_resilience_analysis(time_window_months=3):
    """Full resilience analysis pipeline.

    Uses the last N months of training data to build the route graph,
    then computes centrality, communities, and disruption simulation.
    """
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60, flush=True)
    print("Network Resilience Analysis", flush=True)
    print("=" * 60, flush=True)

    # Load data
    d = "data/processed"
    sources = np.load(os.path.join(d, "sources.npy"), mmap_mode="r")
    destinations = np.load(os.path.join(d, "destinations.npy"), mmap_mode="r")
    timestamps = np.load(os.path.join(d, "timestamps.npy"), mmap_mode="r")
    train_mask = np.load(os.path.join(d, "train_mask.npy"), mmap_mode="r")
    train_end = int(np.sum(train_mask))

    # Load node mapping for airport codes
    import csv
    node_map = {}
    with open(os.path.join(d, "node_mapping.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            code, nid = row[0], int(row[1])
            node_map[nid] = code

    # Time window: last N months of training data
    from datetime import datetime, timezone, timedelta
    val_time = np.quantile(np.array(timestamps[:train_end]), 0.999)
    end_dt = datetime.fromtimestamp(val_time, tz=timezone.utc)
    start_dt = end_dt - timedelta(days=30 * time_window_months)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print(f"\nBuilding route graph ({time_window_months} months: "
          f"{start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')})...",
          flush=True)
    t0 = time.time()
    G = build_route_graph(sources[:train_end], destinations[:train_end],
                          timestamps[:train_end], start_ts, end_ts)
    print(f"  Built in {time.time()-t0:.1f}s", flush=True)
    print(f"  {G.number_of_nodes():,} airports, {G.number_of_edges():,} routes, "
          f"{sum(d['weight'] for _, _, d in G.edges(data=True)):,} flights", flush=True)

    # Centrality
    print("\nComputing centrality metrics...", flush=True)
    t0 = time.time()
    cent = compute_centrality(G, top_k=20)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    print("\nTop 10 airports by metric:")
    print(f"  {'Rank':<5} {'Degree':<12} {'Betweenness':<12} {'PageRank':<12}")
    print("  " + "-" * 40)
    for i in range(10):
        d_code = node_map.get(cent["top_degree"][i], str(cent["top_degree"][i]))
        b_code = node_map.get(cent["top_betweenness"][i], str(cent["top_betweenness"][i]))
        p_code = node_map.get(cent["top_pagerank"][i], str(cent["top_pagerank"][i]))
        print(f"  {i+1:<5} {d_code:<12} {b_code:<12} {p_code:<12}")

    # Communities
    print("\nCommunity detection...", flush=True)
    t0 = time.time()
    node_to_comm, comm_sizes = detect_communities(G)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Disruption simulation (remove by betweenness, then by degree, then random)
    print("\nDisruption simulation...", flush=True)
    t0 = time.time()

    # Strategy 1: Remove by betweenness (targeted attack)
    bc_order = sorted(cent["betweenness_centrality"],
                      key=cent["betweenness_centrality"].get, reverse=True)
    print("  Strategy: targeted (betweenness)", flush=True)
    targeted_results = simulate_disruption(G, bc_order, max_remove=50)

    # Strategy 2: Random removal
    rng = np.random.RandomState(42)
    random_order = list(G.nodes())
    rng.shuffle(random_order)
    print("  Strategy: random", flush=True)
    random_results = simulate_disruption(G, random_order, max_remove=50)

    print(f"  Disruption simulation done in {time.time()-t0:.1f}s", flush=True)

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "total_flights": sum(d["weight"] for _, _, d in G.edges(data=True)),
            "time_window_months": time_window_months,
        },
        "top_airports": {
            "by_degree": [node_map.get(n, str(n)) for n in cent["top_degree"]],
            "by_betweenness": [node_map.get(n, str(n)) for n in cent["top_betweenness"]],
            "by_pagerank": [node_map.get(n, str(n)) for n in cent["top_pagerank"]],
        },
        "communities": {
            "num_communities": len(comm_sizes),
            "sizes": sorted(comm_sizes.values(), reverse=True)[:20],
        },
        "disruption": {
            "targeted": targeted_results,
            "random": random_results,
        },
    }

    with open("results/resilience_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/resilience_results.json", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Resilience analysis complete!", flush=True)
    print(f"{'='*60}", flush=True)

    return results


if __name__ == "__main__":
    run_resilience_analysis(time_window_months=3)
