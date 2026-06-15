#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build HGS Preposs files for one large online graph and one requested feature count.

CLI contract:
  --number_of_features 3
      writes psdfeature_test.json and edge_index_test.pkl

  --number_of_features 10
      writes psdfeature_test_10features.json and edge_index_test.pkl

This script keeps the feature definitions aligned with the code shared by Bharat:

3 features:
  1) eccentricity
  2) log_degree
  3) clustering_coefficient

10 features:
  1) log_degree
  2) clustering_coefficient
  3) eccentricity
  4) log_triangles
  5) log_median_neighbor_degree
  6) log_std_neighbor_degree
  7) betweenness_centrality
  8) eigenvector_centrality
  9) closeness_centrality
 10) degree_centrality

For very large graphs, exact eccentricity, exact betweenness, and exact
closeness are often infeasible. Default mode is approximate. Use:
  --large_graph_mode exact
only for smaller graphs/subgraphs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


ONLINE_GRAPHS: Dict[str, Dict[str, str]] = {
    "com-dblp": {
        "tag": "Com_DBLP",
        "family": "Com-DBLP",
        "instance": "Com-DBLP",
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "filename": "com-dblp.ungraph.txt.gz",
    },
    "com-amazon": {
        "tag": "Com_Amazon",
        "family": "Com-Amazon",
        "instance": "Com-Amazon",
        "url": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "filename": "com-amazon.ungraph.txt.gz",
    },
    "web-google": {
        "tag": "Web_Google",
        "family": "Web-Google",
        "instance": "Web-Google",
        "url": "https://snap.stanford.edu/data/web-Google.txt.gz",
        "filename": "web-Google.txt.gz",
    },
    "as-skitter": {
        "tag": "AS_Skitter",
        "family": "AS-Skitter",
        "instance": "AS-Skitter",
        "url": "https://snap.stanford.edu/data/as-skitter.txt.gz",
        "filename": "as-skitter.txt.gz",
    },
    "ca-dblp-2012": {
        "tag": "ca_DBLP_2012",
        "family": "ca-DBLP-2012",
        "instance": "ca-DBLP-2012",
        "url": "https://nrvis.com/download/data/ca/ca-dblp-2012.zip",
        "filename": "ca-dblp-2012.zip",
    },
}


def normalize_graph_dataset(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    aliases = {
        "dblp": "com-dblp",
        "comdblp": "com-dblp",
        "com-dblp": "com-dblp",
        "amazon": "com-amazon",
        "comamazon": "com-amazon",
        "com-amazon": "com-amazon",
        "google": "web-google",
        "webgoogle": "web-google",
        "web-google": "web-google",
        "skitter": "as-skitter",
        "asskitter": "as-skitter",
        "as-skitter": "as-skitter",
        "cadblp2012": "ca-dblp-2012",
        "ca-dblp-2012": "ca-dblp-2012",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported graph_dataset={name}. Supported: {sorted(ONLINE_GRAPHS)}")
    return aliases[key]


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"Using cached file: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url}")
    print(f"        to {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def parse_edge_lines(lines: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    src: List[int] = []
    dst: List[int] = []
    for line in tqdm(lines, desc="Reading edge list"):
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            u = int(parts[0])
            v = int(parts[1])
        except ValueError:
            continue
        if u != v:
            src.append(u)
            dst.append(v)

    if not src:
        raise ValueError("No edges found.")
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def parse_downloaded_graph(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return parse_edge_lines(f)

    if path.suffix == ".zip":
        extract_dir = path.parent / f"{path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        marker = extract_dir / ".extracted"
        if not marker.exists():
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)
            marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

        candidates = []
        for pat in ("*.edges", "*.mtx", "*.txt"):
            candidates.extend(extract_dir.rglob(pat))
        candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
        if not candidates:
            raise FileNotFoundError(f"No edge-list-like file found inside {path}")
        candidates = sorted(candidates, key=lambda p: (0 if p.suffix == ".edges" else 1, len(str(p))))
        with candidates[0].open("r", encoding="utf-8", errors="ignore") as f:
            return parse_edge_lines(f)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return parse_edge_lines(f)


def remap_to_simple_undirected(src_raw: np.ndarray, dst_raw: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
    u = np.minimum(src_raw, dst_raw)
    v = np.maximum(src_raw, dst_raw)
    mask = u != v
    u = u[mask]
    v = v[mask]

    nodes = np.unique(np.concatenate([u, v]))
    u_idx = np.searchsorted(nodes, u)
    v_idx = np.searchsorted(nodes, v)

    print("Removing duplicate undirected edges ...")
    df = pd.DataFrame({"u": u_idx, "v": v_idx})
    df = df.drop_duplicates(ignore_index=True)
    edge_pairs = df[["u", "v"]].to_numpy(dtype=np.int64)

    n = int(nodes.size)
    m = int(edge_pairs.shape[0])
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
    return edge_pairs, n, m, float(density)


def build_nx_graph(n: int, edge_pairs: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for u, v in edge_pairs)
    return G


def safe_log_3(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    mask = vals > 0
    out[mask] = np.log(vals[mask])
    return out


def safe_log_10(values: np.ndarray, all_zero_fallback: float = -10.0) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    mask = vals > 0
    out = np.empty_like(vals, dtype=float)

    if np.any(mask):
        logs = np.zeros_like(vals, dtype=float)
        logs[mask] = np.log(vals[mask])
        min_log = logs[mask].min()
        out[mask] = logs[mask]
        out[~mask] = min_log - 1.0
    else:
        out.fill(all_zero_fallback)
    return out


def choose_pivots(G: nx.Graph, max_pivots: int, seed: int) -> List[int]:
    if G.number_of_nodes() == 0:
        return []
    rng = random.Random(seed)
    deg = dict(G.degree())
    top_deg = sorted(deg, key=deg.get, reverse=True)[: max_pivots // 2]
    remaining = [n for n in G.nodes() if n not in set(top_deg)]
    sample_k = max(0, max_pivots - len(top_deg))
    random_part = rng.sample(remaining, min(sample_k, len(remaining))) if remaining else []
    return list(dict.fromkeys(top_deg + random_part))


def approximate_eccentricity_by_pivots(G: nx.Graph, max_pivots: int, seed: int) -> np.ndarray:
    ecc = np.zeros(G.number_of_nodes(), dtype=float)
    for comp_id, comp in enumerate(nx.connected_components(G)):
        comp_list = list(comp)
        if len(comp_list) == 1:
            ecc[comp_list[0]] = 0.0
            continue
        H = G.subgraph(comp_list)
        pivots = choose_pivots(H, max_pivots=max_pivots, seed=seed + comp_id)
        for p in pivots:
            lengths = nx.single_source_shortest_path_length(H, p)
            for node, dist in lengths.items():
                if dist > ecc[node]:
                    ecc[node] = float(dist)
    return ecc


def exact_eccentricity_per_component(G: nx.Graph) -> np.ndarray:
    ecc = np.zeros(G.number_of_nodes(), dtype=float)
    comps = list(nx.connected_components(G))
    for comp in tqdm(comps, desc="Exact eccentricity by component"):
        H = G.subgraph(comp)
        if H.number_of_nodes() == 1:
            n = next(iter(comp))
            ecc[n] = 0.0
        else:
            ecd = nx.eccentricity(H)
            for n, v in ecd.items():
                ecc[n] = float(v)
    return ecc


def approx_closeness_by_pivots(G: nx.Graph, max_pivots: int, seed: int) -> np.ndarray:
    n = G.number_of_nodes()
    counts = np.zeros(n, dtype=float)
    dist_sums = np.zeros(n, dtype=float)

    for comp_id, comp in enumerate(nx.connected_components(G)):
        comp_list = list(comp)
        if len(comp_list) == 1:
            continue
        H = G.subgraph(comp_list)
        pivots = choose_pivots(H, max_pivots=max_pivots, seed=seed + 10_000 + comp_id)
        for p in pivots:
            lengths = nx.single_source_shortest_path_length(H, p)
            for node, dist in lengths.items():
                if node != p:
                    dist_sums[node] += float(dist)
                    counts[node] += 1.0

    closeness = np.zeros(n, dtype=float)
    mask = counts > 0
    closeness[mask] = counts[mask] / np.maximum(dist_sums[mask], 1.0)
    return closeness


def median_neighbor_degree(G: nx.Graph, degrees: Dict[int, int]) -> np.ndarray:
    med = np.zeros(G.number_of_nodes(), dtype=float)
    for n in tqdm(G.nodes(), desc="Median neighbor degree"):
        nbrs = list(G.neighbors(n))
        if nbrs:
            med[n] = float(np.median([degrees[v] for v in nbrs]))
    return med


def std_neighbor_degree(G: nx.Graph, degrees: Dict[int, int]) -> np.ndarray:
    st = np.zeros(G.number_of_nodes(), dtype=float)
    for n in tqdm(G.nodes(), desc="Std neighbor degree"):
        nbrs = list(G.neighbors(n))
        if len(nbrs) > 1:
            st[n] = float(np.std([degrees[v] for v in nbrs], ddof=0))
    return st


def compute_3_features(G: nx.Graph, large_graph_mode: str, eccentricity_pivots: int, seed: int) -> List[List[float]]:
    n = G.number_of_nodes()
    deg_dict = dict(G.degree())
    deg = np.array([deg_dict[i] for i in range(n)], dtype=float)

    print("Computing 3-feature log-degree ...")
    log_degree = safe_log_3(deg)

    print("Computing 3-feature clustering coefficient ...")
    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict[i] for i in range(n)], dtype=float)

    if large_graph_mode == "exact":
        print("Computing exact eccentricity. This can be very slow on large graphs.")
        eccentricity = exact_eccentricity_per_component(G)
    else:
        print(f"Computing approximate eccentricity using {eccentricity_pivots} pivots per component.")
        eccentricity = approximate_eccentricity_by_pivots(G, max_pivots=eccentricity_pivots, seed=seed)

    feats = np.stack([eccentricity, log_degree, clustering], axis=1)
    return feats.astype(float).tolist()


def compute_10_features(
    G: nx.Graph,
    large_graph_mode: str,
    eccentricity_pivots: int,
    closeness_pivots: int,
    betweenness_k: int,
    seed: int,
) -> List[List[float]]:
    n = G.number_of_nodes()
    deg_dict = dict(G.degree())
    deg = np.array([deg_dict[i] for i in range(n)], dtype=float)

    print("Computing 10-feature log-degree ...")
    log_degree = safe_log_10(deg)

    print("Computing 10-feature clustering coefficient ...")
    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict[i] for i in range(n)], dtype=float)

    if large_graph_mode == "exact":
        print("Computing exact eccentricity, betweenness, and closeness. This can be infeasible.")
        eccentricity = exact_eccentricity_per_component(G)
        bc_dict = nx.betweenness_centrality(G, normalized=True)
        betweenness = np.array([bc_dict[i] for i in range(n)], dtype=float)
        cl_dict = nx.closeness_centrality(G)
        closeness = np.array([cl_dict[i] for i in range(n)], dtype=float)
    else:
        print(f"Computing approximate eccentricity using {eccentricity_pivots} pivots per component.")
        eccentricity = approximate_eccentricity_by_pivots(G, max_pivots=eccentricity_pivots, seed=seed)

        print(f"Computing approximate betweenness with k={betweenness_k}.")
        k = min(max(1, int(betweenness_k)), n)
        bc_dict = nx.betweenness_centrality(G, k=k, normalized=True, seed=seed)
        betweenness = np.array([bc_dict.get(i, 0.0) for i in range(n)], dtype=float)

        print(f"Computing approximate closeness using {closeness_pivots} pivots per component.")
        closeness = approx_closeness_by_pivots(G, max_pivots=closeness_pivots, seed=seed)

    print("Computing triangle counts ...")
    triangles_dict = nx.triangles(G)
    triangles = np.array([triangles_dict[i] for i in range(n)], dtype=float)
    log_tri = safe_log_10(triangles)

    print("Computing neighbor degree features ...")
    med_nbr_deg = median_neighbor_degree(G, deg_dict)
    log_med_nbr_deg = safe_log_10(med_nbr_deg)

    std_nbr_deg = std_neighbor_degree(G, deg_dict)
    log_std_nbr_deg = safe_log_10(std_nbr_deg)

    print("Computing eigenvector centrality ...")
    try:
        ec_dict = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        eigenvector = np.array([float(ec_dict.get(i, 0.0)) for i in range(n)], dtype=float)
    except Exception as exc:
        print(f"Eigenvector centrality failed ({exc}); using degree-normalized fallback.")
        max_deg = max(float(deg.max()), 1.0)
        eigenvector = deg / max_deg

    print("Computing degree centrality ...")
    dc_dict = nx.degree_centrality(G)
    degree_cent = np.array([dc_dict[i] for i in range(n)], dtype=float)

    feats = np.stack(
        [
            log_degree,
            clustering,
            eccentricity,
            log_tri,
            log_med_nbr_deg,
            log_std_nbr_deg,
            betweenness,
            eigenvector,
            closeness,
            degree_cent,
        ],
        axis=1,
    )
    return feats.astype(float).tolist()


def feature_json_name(number_of_features: int) -> str:
    if number_of_features == 3:
        return "psdfeature_test.json"
    if number_of_features == 10:
        return "psdfeature_test_10features.json"
    raise ValueError("--number_of_features must be 3 or 10")


def build_preposs_for_graph(
    graph_dataset: str,
    number_of_features: int,
    out_root: Path,
    data_root: Path,
    large_graph_mode: str = "approx",
    eccentricity_pivots: int = 64,
    closeness_pivots: int = 64,
    betweenness_k: int = 256,
    seed: int = 42,
    force_download: bool = False,
    force_rebuild: bool = False,
) -> Path:
    graph_key = normalize_graph_dataset(graph_dataset)
    meta = ONLINE_GRAPHS[graph_key]
    out_dir = out_root / meta["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_path = out_dir / feature_json_name(number_of_features)
    edge_path = out_dir / "edge_index_test.pkl"
    metadata_path = out_dir / f"preprocess_metadata_{number_of_features}features.json"

    if feature_path.exists() and edge_path.exists() and metadata_path.exists() and not force_rebuild:
        print("Requested Preposs files already exist. Use --force_rebuild to regenerate.")
        print(f"  {feature_path}")
        print(f"  {edge_path}")
        return out_dir

    raw_path = download_file(
        meta["url"],
        data_root / "raw" / graph_key / meta["filename"],
        force=force_download,
    )
    src_raw, dst_raw = parse_downloaded_graph(raw_path)
    edge_pairs, n, m, density = remap_to_simple_undirected(src_raw, dst_raw)
    print(f"Processed graph: N={n:,}, M={m:,}, density={density:.8e}")

    G = build_nx_graph(n, edge_pairs)

    start = time.time()
    if number_of_features == 3:
        features = compute_3_features(
            G=G,
            large_graph_mode=large_graph_mode,
            eccentricity_pivots=eccentricity_pivots,
            seed=seed,
        )
        feature_names = ["eccentricity", "log_degree", "clustering_coefficient"]
    elif number_of_features == 10:
        features = compute_10_features(
            G=G,
            large_graph_mode=large_graph_mode,
            eccentricity_pivots=eccentricity_pivots,
            closeness_pivots=closeness_pivots,
            betweenness_k=betweenness_k,
            seed=seed,
        )
        feature_names = [
            "log_degree",
            "clustering_coefficient",
            "eccentricity",
            "log_triangles",
            "log_median_neighbor_degree",
            "log_std_neighbor_degree",
            "betweenness_centrality",
            "eigenvector_centrality",
            "closeness_centrality",
            "degree_centrality",
        ]
    else:
        raise ValueError("--number_of_features must be 3 or 10")

    print(f"Writing requested feature file: {feature_path}")
    with feature_path.open("w", encoding="utf-8") as f:
        json.dump([features], f)

    print(f"Writing edge-index file: {edge_path}")
    with edge_path.open("wb") as f:
        pickle.dump([edge_pairs], f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "dataset_key": graph_key,
        "graph_family": meta["family"],
        "instance_name": meta["instance"],
        "num_nodes": n,
        "num_edges": m,
        "density": density,
        "number_of_features_requested": number_of_features,
        "feature_file": str(feature_path),
        "edge_index_file": str(edge_path),
        "feature_names": feature_names,
        "large_graph_mode": large_graph_mode,
        "eccentricity_pivots": eccentricity_pivots,
        "closeness_pivots": closeness_pivots,
        "betweenness_k": betweenness_k,
        "seed": seed,
        "raw_path": str(raw_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_feature_build": time.time() - start,
        "warning": (
            "For large_graph_mode=approx, eccentricity, betweenness, and closeness are scalable approximations. "
            "Use large_graph_mode=exact only on smaller graphs/subgraphs."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Also write/update a generic metadata file for quick inspection.
    (out_dir / "preprocess_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done. Output directory: {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dataset", required=True, help="com-dblp, com-amazon, web-google, as-skitter, ca-dblp-2012")
    ap.add_argument("--number_of_features", type=int, required=True, choices=[3, 10])
    ap.add_argument("--out_root", required=True, help="Example: ~/AWS_poc/data/Preposs")
    ap.add_argument("--data_root", default="~/AWS_poc/data/online_graphs")
    ap.add_argument("--large_graph_mode", default="approx", choices=["approx", "exact"])
    ap.add_argument("--eccentricity_pivots", type=int, default=64)
    ap.add_argument("--closeness_pivots", type=int, default=64)
    ap.add_argument("--betweenness_k", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_download", action="store_true")
    ap.add_argument("--force_rebuild", action="store_true")
    args = ap.parse_args()

    build_preposs_for_graph(
        graph_dataset=args.graph_dataset,
        number_of_features=args.number_of_features,
        out_root=Path(args.out_root).expanduser().resolve(),
        data_root=Path(args.data_root).expanduser().resolve(),
        large_graph_mode=args.large_graph_mode,
        eccentricity_pivots=args.eccentricity_pivots,
        closeness_pivots=args.closeness_pivots,
        betweenness_k=args.betweenness_k,
        seed=args.seed,
        force_download=args.force_download,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
