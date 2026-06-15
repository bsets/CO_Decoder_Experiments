#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end HGS large-graph decoder runner.

This script accepts --number_of_features 3 or 10, ensures the matching HGS
feature JSON exists, loads the requested trained HGS model, and runs decoder R,
decoder P, or both.

For --number_of_features 3:
  reads/builds psdfeature_test.json

For --number_of_features 10:
  reads/builds psdfeature_test_10features.json

In both cases:
  reads/builds edge_index_test.pkl
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix

from build_hgs_large_graph_preposs import build_preposs_for_graph, normalize_graph_dataset, ONLINE_GRAPHS


def getclicnum(adjmatrix, dis, walkerstart=0, thresholdloopnodes=50):
    """
    Original HGS/EGN-style greedy decoder logic formerly imported from Sampler.py.

    Returns:
      (clique_size, clique_nodes)
    """
    _sorted, indices = torch.sort(dis.squeeze(), descending=True)
    initiaprd = 0.0 * indices
    initiaprd = initiaprd.cpu().numpy()

    for walker in range(min(thresholdloopnodes, adjmatrix.get_shape()[0])):
        if walker < walkerstart:
            initiaprd[indices[walker]] = 0.0

    initiaprd[indices[walkerstart]] = 1.0

    for clq in range(walkerstart + 1, min(thresholdloopnodes, adjmatrix.get_shape()[0])):
        initiaprd[indices[clq]] = 1.0
        binary_vec = np.reshape(initiaprd, (-1, 1))
        zor_o = (
            np.sum(binary_vec) ** 2
            - np.sum(binary_vec)
            - np.sum(binary_vec * (adjmatrix.dot(binary_vec)))
        )
        if zor_o < 0.0001:
            pass
        else:
            initiaprd[indices[clq]] = 0.0

    clique_nodes = np.where(initiaprd == 1)[0].tolist()
    clique_size = len(clique_nodes)
    return clique_size, clique_nodes



DATASET_TAGS = {
    "com-dblp": ("Com_DBLP", "Com-DBLP", "Com-DBLP"),
    "com-amazon": ("Com_Amazon", "Com-Amazon", "Com-Amazon"),
    "web-google": ("Web_Google", "Web-Google", "Web-Google"),
    "as-skitter": ("AS_Skitter", "AS-Skitter", "AS-Skitter"),
    "ca-dblp-2012": ("ca_DBLP_2012", "ca-DBLP-2012", "ca-DBLP-2012"),
}


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def feature_json_name(number_of_features: int) -> str:
    if number_of_features == 3:
        return "psdfeature_test.json"
    if number_of_features == 10:
        return "psdfeature_test_10features.json"
    raise ValueError("--number_of_features must be 3 or 10")


def reported_feature_count(number_of_features: int) -> int:
    # User reporting convention: 3-feature model is reported as 2,
    # 10-feature model is reported as 9, because one constant feature
    # is not counted in the experiment table.
    if number_of_features == 3:
        return 2
    if number_of_features == 10:
        return 9
    raise ValueError("--number_of_features must be 3 or 10")


def model_name_feature_check(model_path: Path, number_of_features: int) -> None:
    lower = model_path.name.lower()
    if number_of_features == 3 and "10features" in lower:
        print("WARNING: --number_of_features 3 was selected, but model filename contains '10features'.")
    if number_of_features == 10 and "3features" in lower:
        print("WARNING: --number_of_features 10 was selected, but model filename contains '3features'.")


def get_preposs_files(preposs_root: Path, graph_key: str, number_of_features: int) -> Tuple[Path, Path, Path]:
    tag, _, _ = DATASET_TAGS[graph_key]
    pre_dir = preposs_root / tag
    feature_json = pre_dir / feature_json_name(number_of_features)
    edge_pkl = pre_dir / "edge_index_test.pkl"
    metadata_json = pre_dir / f"preprocess_metadata_{number_of_features}features.json"
    if not metadata_json.exists():
        metadata_json = pre_dir / "preprocess_metadata.json"
    return feature_json, edge_pkl, metadata_json


def ensure_preposs(
    graph_dataset: str,
    number_of_features: int,
    preposs_root: Path,
    data_root: Path,
    preprocess_if_missing: bool,
    large_graph_mode: str,
    eccentricity_pivots: int,
    closeness_pivots: int,
    betweenness_k: int,
    seed: int,
    force_download: bool,
    force_rebuild: bool,
) -> Tuple[Path, Path, Path]:
    graph_key = normalize_graph_dataset(graph_dataset)
    feature_json, edge_pkl, metadata_json = get_preposs_files(preposs_root, graph_key, number_of_features)

    missing = [p for p in [feature_json, edge_pkl] if not p.exists()]
    if missing or force_rebuild:
        if not preprocess_if_missing and not force_rebuild:
            raise FileNotFoundError(
                "Required Preposs files are missing. Either run build_hgs_large_graph_preposs.py first "
                "or pass --preprocess_if_missing.\nMissing:\n" + "\n".join(str(p) for p in missing)
            )

        build_preposs_for_graph(
            graph_dataset=graph_dataset,
            number_of_features=number_of_features,
            out_root=preposs_root,
            data_root=data_root,
            large_graph_mode=large_graph_mode,
            eccentricity_pivots=eccentricity_pivots,
            closeness_pivots=closeness_pivots,
            betweenness_k=betweenness_k,
            seed=seed,
            force_download=force_download,
            force_rebuild=force_rebuild,
        )
        feature_json, edge_pkl, metadata_json = get_preposs_files(preposs_root, graph_key, number_of_features)

    return feature_json, edge_pkl, metadata_json


def load_dataset(feature_json: Path, edge_pkl: Path) -> List[Data]:
    with feature_json.open("r", encoding="utf-8") as f:
        psd_features = json.load(f)
    with edge_pkl.open("rb") as f:
        edge_index_list = pickle.load(f)

    if len(psd_features) != len(edge_index_list):
        raise ValueError(
            f"Mismatch: {feature_json} has {len(psd_features)} feature arrays, "
            f"but {edge_pkl} has {len(edge_index_list)} edge arrays."
        )

    dataset = []
    for feats, eidx_raw in zip(psd_features, edge_index_list):
        raw = torch.tensor(eidx_raw, dtype=torch.long)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(f"Expected edge array shape (E,2), got {tuple(raw.shape)}")
        edge_index = to_undirected(raw.t().contiguous())
        x = torch.tensor(feats, dtype=torch.float32)
        dataset.append(Data(x=x, edge_index=edge_index))
    return dataset


def infer_model_input_dim(model: torch.nn.Module) -> Optional[int]:
    if hasattr(model, "in_proj") and hasattr(model.in_proj, "in_features"):
        return int(model.in_proj.in_features)
    if hasattr(model, "gc1") and hasattr(model.gc1, "mlp") and hasattr(model.gc1.mlp, "in_features"):
        return int(model.gc1.mlp.in_features)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    return None


def forward_model(model, x, adj_ts):
    try:
        return model(x, adj_ts, moment=1, device="cpu")
    except TypeError:
        return model(x, adj_ts, moment=1)


def get_node_scores(out, num_nodes: int) -> np.ndarray:
    if not torch.is_tensor(out):
        out = torch.tensor(out)
    out = out.detach().cpu().float()

    if out.numel() == num_nodes:
        return out.reshape(num_nodes).numpy()

    if out.ndim >= 2 and out.shape[0] == num_nodes:
        flat = out.reshape(num_nodes, -1)
        if flat.shape[1] == 1:
            return flat[:, 0].numpy()
        return flat[:, -1].numpy()

    raise ValueError(f"Could not convert model output shape {tuple(out.shape)} into N={num_nodes} scores")


def extract_clique_size(result) -> int:
    if torch.is_tensor(result) and result.numel() == 1:
        return int(result.item())
    if isinstance(result, np.ndarray):
        if result.size == 1:
            return int(result.reshape(-1)[0].item())
        if result.ndim == 1:
            return int(result.size)
    if isinstance(result, (int, np.integer)):
        return int(result)
    if isinstance(result, (float, np.floating)):
        return int(result)
    if isinstance(result, (tuple, list)):
        for elem in result:
            try:
                return extract_clique_size(elem)
            except Exception:
                pass
        lens = []
        for elem in result:
            if isinstance(elem, (list, tuple, set)):
                lens.append(len(elem))
            elif torch.is_tensor(elem) and elem.ndim == 1:
                lens.append(int(elem.numel()))
            elif isinstance(elem, np.ndarray) and elem.ndim == 1:
                lens.append(int(elem.size))
        if lens:
            return max(lens)
    raise TypeError(f"Unsupported clique result type: {type(result)} -> {result!r}")


def build_neighbor_sets(adj_coo) -> List[set]:
    adj_csr = adj_coo.tocsr()
    nbrs = []
    for i in tqdm(range(adj_csr.shape[0]), desc="Building neighbor sets"):
        s = set(adj_csr.getrow(i).indices.tolist())
        s.discard(i)
        nbrs.append(s)
    return nbrs


def greedy_clique_with_anchor_and_skips(ranked_nodes, neighbor_sets, skip_count: int) -> List[int]:
    if len(ranked_nodes) == 0:
        return []
    anchor = int(ranked_nodes[0])
    clique = [anchor]
    excluded = set(int(v) for v in ranked_nodes[1:1 + skip_count])
    for node in ranked_nodes[1:]:
        node = int(node)
        if node in excluded:
            continue
        if all(node in neighbor_sets[u] for u in clique):
            clique.append(node)
    return clique


def run_decoder_r(dataset, model, walkers_ratio, min_walkers, graph_timeout_sec, candidate_limit):
    cliques, times = [], []
    model.eval()
    model.cpu()

    with torch.no_grad():
        for g in tqdm(dataset, desc="Decoder R/original"):
            t0 = time.time()
            x = g.x.cpu().float()
            eix = g.edge_index.cpu()
            adj_coo = to_scipy_sparse_matrix(eix, num_nodes=g.num_nodes)
            adj_ts = sparse_mx_to_torch_sparse_tensor(adj_coo).cpu()
            out = forward_model(model, x, adj_ts)

            n = int(g.num_nodes)
            k = min(max(min_walkers, int(math.floor(walkers_ratio * n))), n)
            threshold = n if candidate_limit is None else min(candidate_limit, n)

            best = 0
            for s in range(k):
                if time.time() - t0 >= graph_timeout_sec:
                    print(f"[TIMEOUT] R reached {graph_timeout_sec}s; best={best}")
                    break
                if s >= threshold:
                    break
                csize = extract_clique_size(getclicnum(adj_coo, out, walkerstart=s, thresholdloopnodes=threshold))
                best = max(best, csize)

            cliques.append(int(best))
            times.append(float(min(time.time() - t0, graph_timeout_sec)))
    return cliques, times


def run_decoder_p(dataset, model, walkers_ratio, min_walkers, graph_timeout_sec, candidate_limit):
    cliques, times = [], []
    model.eval()
    model.cpu()

    with torch.no_grad():
        for g in tqdm(dataset, desc="Decoder P/modified"):
            t0 = time.time()
            x = g.x.cpu().float()
            eix = g.edge_index.cpu()
            adj_coo = to_scipy_sparse_matrix(eix, num_nodes=g.num_nodes)
            adj_ts = sparse_mx_to_torch_sparse_tensor(adj_coo).cpu()
            out = forward_model(model, x, adj_ts)

            n = int(g.num_nodes)
            k = min(max(min_walkers, int(math.floor(walkers_ratio * n))), n)
            threshold = n if candidate_limit is None else min(candidate_limit, n)

            scores = get_node_scores(out, n)
            ranked_nodes = np.argsort(-scores, kind="stable")[:threshold]
            neighbor_sets = build_neighbor_sets(adj_coo)

            best = 0
            for s in range(k):
                if time.time() - t0 >= graph_timeout_sec:
                    print(f"[TIMEOUT] P reached {graph_timeout_sec}s; best={best}")
                    break
                if s >= len(ranked_nodes):
                    break
                clique = greedy_clique_with_anchor_and_skips(ranked_nodes, neighbor_sets, skip_count=s)
                best = max(best, len(clique))

            cliques.append(int(best))
            times.append(float(min(time.time() - t0, graph_timeout_sec)))
    return cliques, times


def normalize_decoder(decoder: str) -> List[str]:
    d = decoder.lower().strip()
    if d in {"r", "original", "decoder_original"}:
        return ["R"]
    if d in {"p", "modified", "decoder_highest_prob_node_always_included"}:
        return ["P"]
    if d in {"both", "all"}:
        return ["R", "P"]
    raise ValueError("--decoder must be R, P, or both")


def append_excel(out_xlsx: Path, rows: List[Dict]) -> pd.DataFrame:
    cols = [
        "S.N.",
        "Graph family",
        "Instances",
        "Number of features",
        "Decoder",
        "Local Machine /AWS",
        "Compute Instance",
        "Total Time Taken (s)",
        "Average Time Taken (s)",
        "Average Clique Size",
        "Median Clique Size",
        "Cost ($)",
    ]
    new = pd.DataFrame(rows)[cols]
    if out_xlsx.exists():
        old = pd.read_excel(out_xlsx)
        for c in cols:
            if c not in old.columns:
                old[c] = np.nan
        df = pd.concat([old[cols], new], ignore_index=True)
    else:
        df = new
    df["S.N."] = np.arange(1, len(df) + 1)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dataset", required=True)
    ap.add_argument("--number_of_features", type=int, required=True, choices=[3, 10])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--trained_on", default="bhoslib_and_dimacs")
    ap.add_argument("--data_root", default="~/AWS_poc/data/online_graphs")
    ap.add_argument("--preposs_root", default="~/AWS_poc/data/Preposs")
    ap.add_argument("--results_root", default="~/AWS_poc/results")
    ap.add_argument("--decoder", default="both")
    ap.add_argument("--num_passes_as_percentage_of_total_nodes", "--walkers_ratio", dest="walkers_ratio", type=float, default=0.05)
    ap.add_argument("--min_walkers", type=int, default=20)
    ap.add_argument("--candidate_limit", type=int, default=5000)
    ap.add_argument("--graph_timeout_sec", type=int, default=21600)
    ap.add_argument("--run_label", default="aws")
    ap.add_argument("--hourly_rate_usd", type=float, default=0.714)
    ap.add_argument("--instance_type", default="c7i.4xlarge")
    ap.add_argument("--out_xlsx", default=None)

    # Preprocessing options.
    ap.add_argument("--preprocess_if_missing", action="store_true")
    ap.add_argument("--force_download", action="store_true")
    ap.add_argument("--force_rebuild", action="store_true")
    ap.add_argument("--large_graph_mode", default="approx", choices=["approx", "exact"])
    ap.add_argument("--eccentricity_pivots", type=int, default=64)
    ap.add_argument("--closeness_pivots", type=int, default=64)
    ap.add_argument("--betweenness_k", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    graph_key = normalize_graph_dataset(args.graph_dataset)
    _, family, instance = DATASET_TAGS[graph_key]

    data_root = Path(args.data_root).expanduser().resolve()
    preposs_root = Path(args.preposs_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    out_xlsx = Path(args.out_xlsx).expanduser().resolve() if args.out_xlsx else results_root / "hgs_large_graph_decoder_summary.xlsx"

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model_name_feature_check(model_path, args.number_of_features)

    feature_json, edge_pkl, preprocess_metadata = ensure_preposs(
        graph_dataset=args.graph_dataset,
        number_of_features=args.number_of_features,
        preposs_root=preposs_root,
        data_root=data_root,
        preprocess_if_missing=args.preprocess_if_missing,
        large_graph_mode=args.large_graph_mode,
        eccentricity_pivots=args.eccentricity_pivots,
        closeness_pivots=args.closeness_pivots,
        betweenness_k=args.betweenness_k,
        seed=args.seed,
        force_download=args.force_download,
        force_rebuild=args.force_rebuild,
    )

    print(f"Feature file: {feature_json}")
    print(f"Edge file   : {edge_pkl}")

    dataset = load_dataset(feature_json, edge_pkl)

    print(f"Loading model: {model_path}")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.cpu()
    model.eval()

    input_dim = infer_model_input_dim(model)
    if input_dim is not None and input_dim != args.number_of_features:
        print(
            f"WARNING: model input_dim appears to be {input_dim}, but "
            f"--number_of_features is {args.number_of_features}."
        )

    meta = {}
    if preprocess_metadata.exists():
        meta = json.loads(preprocess_metadata.read_text(encoding="utf-8"))

    rows = []
    raw_rows = []

    for dec in normalize_decoder(args.decoder):
        print(f"Running decoder {dec} on {instance}, model={model_path.name}")
        if dec == "R":
            cliques, times = run_decoder_r(dataset, model, args.walkers_ratio, args.min_walkers, args.graph_timeout_sec, args.candidate_limit)
        else:
            cliques, times = run_decoder_p(dataset, model, args.walkers_ratio, args.min_walkers, args.graph_timeout_sec, args.candidate_limit)

        total_time = float(np.sum(times))
        avg_time = float(np.mean(times))
        avg_clique = float(np.mean(cliques))
        med_clique = float(np.median(cliques))
        cost = total_time / 3600.0 * args.hourly_rate_usd

        row = {
            "S.N.": None,
            "Graph family": family,
            "Instances": instance,
            "Number of features": reported_feature_count(args.number_of_features),
            "Decoder": dec,
            "Local Machine /AWS": "AWS",
            "Compute Instance": args.instance_type,
            "Total Time Taken (s)": total_time,
            "Average Time Taken (s)": avg_time,
            "Average Clique Size": avg_clique,
            "Median Clique Size": med_clique,
            "Cost ($)": cost,
        }
        rows.append(row)
        raw_rows.append({
            **row,
            "model_path": str(model_path),
            "model_file": model_path.name,
            "model_input_dim": input_dim,
            "file_feature_count": args.number_of_features,
            "graph_dataset": graph_key,
            "feature_json": str(feature_json),
            "edge_pkl": str(edge_pkl),
            "preprocess_metadata": str(preprocess_metadata),
            "number_of_nodes": meta.get("num_nodes"),
            "number_of_edges": meta.get("num_edges"),
            "density": meta.get("density"),
            "walkers_ratio": args.walkers_ratio,
            "candidate_limit": args.candidate_limit,
            "graph_timeout_sec": args.graph_timeout_sec,
            "run_label": args.run_label,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    df = append_excel(out_xlsx, rows)
    print(f"Wrote Excel: {out_xlsx}")
    print(df.tail(len(rows)).to_string(index=False))

    raw_path = results_root / "hgs_large_graph_decoder_raw_metrics.csv"
    results_root.mkdir(parents=True, exist_ok=True)
    new_raw = pd.DataFrame(raw_rows)
    if raw_path.exists():
        old_raw = pd.read_csv(raw_path)
        full_raw = pd.concat([old_raw, new_raw], ignore_index=True)
    else:
        full_raw = new_raw
    full_raw.to_csv(raw_path, index=False)

    run_meta_dir = results_root / "metadata"
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "outputs": {
            "excel": str(out_xlsx),
            "raw_csv": str(raw_path),
        },
    }
    meta_path = run_meta_dir / f"hgs_large_graph_decoder_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
