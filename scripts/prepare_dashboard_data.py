"""
Precompute all data needed for the Streamlit dashboard.

Produces results/dashboard_data.json with:
  - airport_meta: ICAO -> {lat, lon, name, city, country, node_id}
  - route_graph: {edges: [[src, dst, weight], ...]}
  - airport_stats: ICAO -> {degree_in, degree_out, total_flights, top_connections, community}
  - predictions: ICAO -> [{dst, score}, ...] top-K by historical frequency
"""

import sys, os, json, time, csv
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "processed")
RESULTS = os.path.join(BASE, "results")


def load_node_mapping():
    """Load ICAO code <-> node ID mapping."""
    id_to_code = {}
    code_to_id = {}
    with open(os.path.join(DATA, "node_mapping.csv"), encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            code, nid = row[0], int(row[1])
            id_to_code[nid] = code
            code_to_id[code] = nid
    return id_to_code, code_to_id


def load_airport_metadata(id_to_code):
    """Load airport lat/lon/name from airportsdata."""
    import airportsdata
    db = airportsdata.load("ICAO")

    meta = {}
    found = 0
    for nid, code in id_to_code.items():
        info = db.get(code, {})
        if info and info.get("lat") is not None:
            meta[code] = {
                "lat": info["lat"],
                "lon": info["lon"],
                "name": info.get("name", code),
                "city": info.get("city", ""),
                "country": info.get("country", ""),
                "node_id": nid,
            }
            found += 1
        else:
            meta[code] = {
                "lat": None,
                "lon": None,
                "name": code,
                "city": "",
                "country": "",
                "node_id": nid,
            }
    print(f"  Airport metadata: {found}/{len(id_to_code)} with coordinates")
    return meta


def build_route_data(id_to_code, time_window_months=3):
    """Build route graph from the training data (3-month window)."""
    sources = np.load(os.path.join(DATA, "sources.npy"), mmap_mode="r")
    destinations = np.load(os.path.join(DATA, "destinations.npy"), mmap_mode="r")
    timestamps = np.load(os.path.join(DATA, "timestamps.npy"), mmap_mode="r")
    train_mask = np.load(os.path.join(DATA, "train_mask.npy"), mmap_mode="r")
    train_end = int(np.sum(train_mask))

    # Time window
    from datetime import datetime, timezone, timedelta
    val_time = np.quantile(np.array(timestamps[:train_end]), 0.999)
    end_dt = datetime.fromtimestamp(val_time, tz=timezone.utc)
    start_dt = end_dt - timedelta(days=30 * time_window_months)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print(f"  Time window: {start_dt.strftime('%Y-%m')} to {end_dt.strftime('%Y-%m')}")

    # Count routes in chunks
    route_counts = defaultdict(int)
    chunk = 5_000_000
    for start in range(0, train_end, chunk):
        end_idx = min(start + chunk, train_end)
        src_c = np.array(sources[start:end_idx], dtype=np.int32)
        dst_c = np.array(destinations[start:end_idx], dtype=np.int32)
        ts_c = np.array(timestamps[start:end_idx], dtype=np.int64)

        mask = (ts_c >= start_ts) & (ts_c < end_ts)
        src_c = src_c[mask]
        dst_c = dst_c[mask]

        for s, d in zip(src_c.tolist(), dst_c.tolist()):
            route_counts[(s, d)] += 1

        print(f"    Processed {end_idx:,}/{train_end:,} edges...", flush=True)

    # Convert to ICAO codes
    edges = []
    for (s, d), w in route_counts.items():
        sc = id_to_code.get(s, str(s))
        dc = id_to_code.get(d, str(d))
        edges.append([sc, dc, w])

    total_flights = sum(w for _, _, w in edges)
    print(f"  Routes: {len(edges):,}, Flights: {total_flights:,}")
    return edges, route_counts


def compute_airport_stats(edges, route_counts, id_to_code, resilience_results):
    """Compute per-airport statistics."""
    # Degree and flight counts
    degree_out = defaultdict(int)
    degree_in = defaultdict(int)
    flights_out = defaultdict(int)
    flights_in = defaultdict(int)
    connections = defaultdict(lambda: defaultdict(int))

    for src, dst, w in edges:
        degree_out[src] += 1
        degree_in[dst] += 1
        flights_out[src] += w
        flights_in[dst] += w
        connections[src][dst] = w

    all_airports = set(degree_out.keys()) | set(degree_in.keys())

    # Load community info from resilience results
    # (we don't have per-airport community IDs saved, so skip for now)

    stats = {}
    for code in all_airports:
        d_out = degree_out.get(code, 0)
        d_in = degree_in.get(code, 0)
        f_out = flights_out.get(code, 0)
        f_in = flights_in.get(code, 0)

        # Top connections (by flight count)
        conns = connections.get(code, {})
        top_conns = sorted(conns.items(), key=lambda x: -x[1])[:20]

        stats[code] = {
            "degree_out": d_out,
            "degree_in": d_in,
            "degree_total": d_out + d_in,
            "flights_out": f_out,
            "flights_in": f_in,
            "flights_total": f_out + f_in,
            "top_connections": [{"dst": dst, "flights": cnt} for dst, cnt in top_conns],
        }

    print(f"  Stats computed for {len(stats):,} airports")
    return stats


def compute_predictions(route_counts, id_to_code, top_k=20):
    """Compute top-K predicted destinations per airport using historical frequency.

    This uses HistFreq (our best MRR model at 0.738) as the prediction engine.
    """
    # Group by source
    src_dsts = defaultdict(list)
    for (s, d), count in route_counts.items():
        sc = id_to_code.get(s, str(s))
        dc = id_to_code.get(d, str(d))
        src_dsts[sc].append((dc, count))

    predictions = {}
    for src, dsts in src_dsts.items():
        sorted_dsts = sorted(dsts, key=lambda x: -x[1])[:top_k]
        total = sum(c for _, c in dsts)
        predictions[src] = [
            {"dst": dst, "flights": cnt, "score": round(cnt / total, 4)}
            for dst, cnt in sorted_dsts
        ]

    print(f"  Predictions for {len(predictions):,} source airports")
    return predictions


def build_map_edges(edges, airport_meta, min_flights=50):
    """Build edge list for the world map (only routes with enough flights)."""
    map_edges = []
    for src, dst, w in edges:
        if w >= min_flights:
            src_meta = airport_meta.get(src, {})
            dst_meta = airport_meta.get(dst, {})
            if src_meta.get("lat") and dst_meta.get("lat"):
                map_edges.append({
                    "src": src, "dst": dst, "flights": w,
                    "src_lat": src_meta["lat"], "src_lon": src_meta["lon"],
                    "dst_lat": dst_meta["lat"], "dst_lon": dst_meta["lon"],
                })
    print(f"  Map edges (>={min_flights} flights): {len(map_edges):,}")
    return map_edges


def main():
    print("=" * 60)
    print("Preparing dashboard data")
    print("=" * 60)

    t0 = time.time()

    print("\n1. Loading node mapping...")
    id_to_code, code_to_id = load_node_mapping()
    print(f"  {len(id_to_code):,} airports in dataset")

    print("\n2. Loading airport metadata (coordinates)...")
    airport_meta = load_airport_metadata(id_to_code)

    print("\n3. Building route graph (3-month window)...")
    edges, route_counts = build_route_data(id_to_code)

    print("\n4. Computing airport statistics...")
    resilience = json.load(open(os.path.join(RESULTS, "resilience_results.json")))
    stats = compute_airport_stats(edges, route_counts, id_to_code, resilience)

    print("\n5. Computing predictions (HistFreq)...")
    predictions = compute_predictions(route_counts, id_to_code)

    print("\n6. Building map edge data...")
    map_edges = build_map_edges(edges, airport_meta, min_flights=50)

    # Save
    dashboard_data = {
        "airport_meta": airport_meta,
        "airport_stats": stats,
        "predictions": predictions,
        "map_edges": map_edges,
        "graph_summary": {
            "num_airports": len(set(s for s, _, _ in edges) | set(d for _, d, _ in edges)),
            "num_routes": len(edges),
            "total_flights": sum(w for _, _, w in edges),
        },
    }

    out_path = os.path.join(RESULTS, "dashboard_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path} ({size_mb:.1f} MB)")
    print(f"Total time: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
