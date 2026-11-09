"""
Preprocessing pipeline for tgbl-flight.

Converts the raw CSV into numpy arrays compatible with TGB's expected format.
This replaces TGB's built-in preprocessor which requires more RAM than available.

Pipeline:
  1. Build airport_code → integer ID mapping from the node features CSV
  2. Read the edge CSV in chunks, converting string codes to integer IDs
  3. Encode edge features (typecode) as integers
  4. Save everything as .npy files for fast reloading
  5. Compute and save the chronological train/val/test masks

The resulting arrays match what TGB's LinkPropPredDataset.full_data provides:
  - sources: int array of source node IDs
  - destinations: int array of destination node IDs
  - timestamps: int array of UNIX timestamps
  - edge_feat: float array of encoded edge features
"""

import os
import numpy as np
import pandas as pd
from collections import Counter


def build_node_mapping(node_feat_path: str, edge_csv_path: str) -> dict:
    """Build a mapping from ICAO airport codes to contiguous integer IDs.

    Uses first-seen order from the chronological CSV, matching TGB's
    own mapping convention. This ensures our node IDs are compatible
    with TGB's pre-generated negative samples.

    Returns:
        dict with keys:
        - code_to_id: {str: int} mapping
        - id_to_code: {int: str} reverse mapping
        - node_features: np.ndarray of shape (num_nodes, num_features)
        - feature_names: list of feature column names
    """
    nf = pd.read_csv(node_feat_path)

    # Assign IDs by first-seen order (matching TGB's convention)
    code_to_id = {}
    uid = 0
    for chunk in pd.read_csv(edge_csv_path, chunksize=5_000_000, usecols=["src", "dst"]):
        for src, dst in zip(chunk["src"], chunk["dst"]):
            if src not in code_to_id:
                code_to_id[src] = uid
                uid += 1
            if dst not in code_to_id:
                code_to_id[dst] = uid
                uid += 1

    id_to_code = {i: code for code, i in code_to_id.items()}

    # Build node feature matrix
    # Features: type (one-hot), continent (one-hot), longitude, latitude
    nf_indexed = nf.set_index("airport_code")

    # Encode airport type
    type_categories = sorted(nf["type"].unique())
    type_to_idx = {t: i for i, t in enumerate(type_categories)}

    # Encode continent (with "NA" for missing)
    continent_categories = sorted(nf["continent"].dropna().unique())
    continent_to_idx = {c: i for i, c in enumerate(continent_categories)}

    num_nodes = len(code_to_id)
    num_type = len(type_categories)
    num_continent = len(continent_categories) + 1  # +1 for missing

    # Feature vector: [type_onehot(7), continent_onehot(7), longitude, latitude]
    num_features = num_type + num_continent + 2
    node_features = np.zeros((num_nodes, num_features), dtype=np.float32)

    for code, i in code_to_id.items():
        if code in nf_indexed.index:
            row = nf_indexed.loc[code]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Type one-hot
            t_idx = type_to_idx.get(row["type"], 0)
            node_features[i, t_idx] = 1.0

            # Continent one-hot
            if pd.notna(row["continent"]):
                c_idx = continent_to_idx[row["continent"]]
                node_features[i, num_type + c_idx] = 1.0
            else:
                node_features[i, num_type + num_continent - 1] = 1.0  # missing

            # Longitude and latitude (normalized to [-1, 1])
            node_features[i, num_type + num_continent] = row["longitude"] / 180.0
            node_features[i, num_type + num_continent + 1] = row["latitude"] / 90.0

    feature_names = (
        [f"type_{t}" for t in type_categories]
        + [f"continent_{c}" for c in continent_categories]
        + ["continent_missing"]
        + ["longitude_norm", "latitude_norm"]
    )

    return {
        "code_to_id": code_to_id,
        "id_to_code": id_to_code,
        "node_features": node_features,
        "feature_names": feature_names,
        "type_categories": type_categories,
        "continent_categories": continent_categories,
    }


def build_typecode_mapping(edge_csv_path: str, max_types: int = 200) -> dict:
    """Build a mapping from aircraft typecodes to integer IDs.

    Keeps the top `max_types` most frequent codes; everything else
    maps to a single "rare" bucket. Missing typecodes get ID 0.

    Returns dict with code_to_id mapping and num_codes.
    """
    counter = Counter()
    for chunk in pd.read_csv(edge_csv_path, chunksize=10_000_000, usecols=["typecode"]):
        tc = chunk["typecode"].dropna()
        counter.update(tc.value_counts().to_dict())

    top_codes = [code for code, _ in counter.most_common(max_types)]
    # 0 = missing, 1 = rare, 2+ = specific typecodes
    tc_to_id = {code: i + 2 for i, code in enumerate(top_codes)}

    return {
        "tc_to_id": tc_to_id,
        "num_codes": len(top_codes) + 2,  # +2 for missing and rare
        "top_codes": top_codes,
    }


def preprocess_edges(
    edge_csv_path: str,
    code_to_id: dict,
    tc_to_id: dict,
    output_dir: str,
) -> dict:
    """Convert the edge CSV to numpy arrays, processing in chunks.

    Saves to output_dir:
      - sources.npy, destinations.npy, timestamps.npy, edge_feat.npy

    Returns dict with array shapes and the output paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # First pass: count total edges to pre-allocate
    n_total = 0
    for chunk in pd.read_csv(edge_csv_path, chunksize=10_000_000, usecols=["src"]):
        n_total += len(chunk)

    sources = np.empty(n_total, dtype=np.int32)
    destinations = np.empty(n_total, dtype=np.int32)
    timestamps = np.empty(n_total, dtype=np.int64)
    edge_feat = np.empty(n_total, dtype=np.int32)

    offset = 0
    for i, chunk in enumerate(pd.read_csv(edge_csv_path, chunksize=5_000_000)):
        n = len(chunk)
        sources[offset : offset + n] = chunk["src"].map(code_to_id).values
        destinations[offset : offset + n] = chunk["dst"].map(code_to_id).values
        timestamps[offset : offset + n] = chunk["timestamp"].values

        # Encode typecodes: 0=missing, 1=rare, 2+=known
        tc_encoded = np.ones(n, dtype=np.int32)  # default: rare
        mask_null = chunk["typecode"].isna()
        tc_encoded[mask_null.values] = 0
        known_mask = ~mask_null
        if known_mask.any():
            known_tc = chunk.loc[known_mask, "typecode"]
            tc_encoded[known_mask.values] = known_tc.map(
                lambda x: tc_to_id.get(x, 1)
            ).values

        edge_feat[offset : offset + n] = tc_encoded
        offset += n
        print(f"  Chunk {i}: {offset:,} / {n_total:,}", flush=True)

    # Verify chronological order
    assert np.all(timestamps[:-1] <= timestamps[1:]), "Timestamps not sorted!"

    # Save arrays
    np.save(os.path.join(output_dir, "sources.npy"), sources)
    np.save(os.path.join(output_dir, "destinations.npy"), destinations)
    np.save(os.path.join(output_dir, "timestamps.npy"), timestamps)
    np.save(os.path.join(output_dir, "edge_feat.npy"), edge_feat)

    return {
        "n_edges": n_total,
        "sources_shape": sources.shape,
        "timestamps_range": (int(timestamps[0]), int(timestamps[-1])),
    }


def compute_temporal_split(
    timestamps: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple:
    """Compute chronological train/val/test masks.

    Uses timestamp quantiles, matching TGB's generate_splits():
      val_time = quantile(timestamps, 1 - val_ratio - test_ratio)
      test_time = quantile(timestamps, 1 - test_ratio)
      train: t <= val_time
      val:   val_time < t <= test_time
      test:  t > test_time

    Returns (train_mask, val_mask, test_mask) as boolean arrays.
    """
    val_time, test_time = np.quantile(
        timestamps,
        [(1 - val_ratio - test_ratio), (1 - test_ratio)],
    )
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time

    return train_mask, val_mask, test_mask


def run_full_preprocessing(
    raw_dir: str = "datasets/tgbl-flight",
    output_dir: str = "data/processed",
):
    """Run the complete preprocessing pipeline.

    This is the main entry point. Call it once; subsequent runs
    skip if output files already exist.
    """
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    edge_csv = os.path.join(raw_dir, "tgbl-flight_edgelist_v2.csv")
    node_csv = os.path.join(raw_dir, "airport_node_feat_v2.csv")

    if os.path.exists(os.path.join(output_dir, "sources.npy")):
        print("Preprocessed files already exist. Loading...")
        return load_preprocessed(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Node mapping
    print("Step 1/4: Building node mapping...", flush=True)
    node_info = build_node_mapping(node_csv, edge_csv)
    np.save(os.path.join(output_dir, "node_features.npy"), node_info["node_features"])

    # Save mapping as CSV for interpretability
    mapping_df = pd.DataFrame(
        [(code, idx) for code, idx in node_info["code_to_id"].items()],
        columns=["airport_code", "node_id"],
    )
    mapping_df.to_csv(os.path.join(output_dir, "node_mapping.csv"), index=False)

    # Step 2: Typecode mapping
    print("Step 2/4: Building typecode mapping...", flush=True)
    tc_info = build_typecode_mapping(edge_csv, max_types=200)

    tc_df = pd.DataFrame(
        [(code, idx) for code, idx in tc_info["tc_to_id"].items()],
        columns=["typecode", "type_id"],
    )
    tc_df.to_csv(os.path.join(output_dir, "typecode_mapping.csv"), index=False)

    # Step 3: Convert edges
    print("Step 3/4: Converting edges to numpy arrays...", flush=True)
    edge_info = preprocess_edges(
        edge_csv, node_info["code_to_id"], tc_info["tc_to_id"], output_dir
    )

    # Step 4: Compute split
    print("Step 4/4: Computing temporal split...", flush=True)
    timestamps = np.load(os.path.join(output_dir, "timestamps.npy"))
    train_mask, val_mask, test_mask = compute_temporal_split(timestamps)
    np.save(os.path.join(output_dir, "train_mask.npy"), train_mask)
    np.save(os.path.join(output_dir, "val_mask.npy"), val_mask)
    np.save(os.path.join(output_dir, "test_mask.npy"), test_mask)

    print(f"\nPreprocessing complete!", flush=True)
    print(f"  Edges: {edge_info['n_edges']:,}", flush=True)
    print(f"  Nodes: {len(node_info['code_to_id']):,}", flush=True)
    print(f"  Node features: {node_info['node_features'].shape}", flush=True)
    print(f"  Train: {train_mask.sum():,} ({100*train_mask.mean():.1f}%)", flush=True)
    print(f"  Val: {val_mask.sum():,} ({100*val_mask.mean():.1f}%)", flush=True)
    print(f"  Test: {test_mask.sum():,} ({100*test_mask.mean():.1f}%)", flush=True)

    return load_preprocessed(output_dir)


def load_preprocessed(output_dir: str = "data/processed") -> dict:
    """Load preprocessed numpy arrays."""
    return {
        "sources": np.load(os.path.join(output_dir, "sources.npy")),
        "destinations": np.load(os.path.join(output_dir, "destinations.npy")),
        "timestamps": np.load(os.path.join(output_dir, "timestamps.npy")),
        "edge_feat": np.load(os.path.join(output_dir, "edge_feat.npy")),
        "node_features": np.load(os.path.join(output_dir, "node_features.npy")),
        "train_mask": np.load(os.path.join(output_dir, "train_mask.npy")),
        "val_mask": np.load(os.path.join(output_dir, "val_mask.npy")),
        "test_mask": np.load(os.path.join(output_dir, "test_mask.npy")),
        "node_mapping": pd.read_csv(
            os.path.join(output_dir, "node_mapping.csv")
        ),
    }


if __name__ == "__main__":
    run_full_preprocessing()
