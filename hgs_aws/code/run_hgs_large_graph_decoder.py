#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end HGS large-graph decoder runner.

This script accepts --number_of_features 3 or 10, ensures the matching HGS
feature JSON exists, loads the requested trained HGS model, and runs decoder R,
decoder P, or both. Includes inner decoder progress bars, checkpoint/resume support, and
best-clique-size trajectory logging to Excel every checkpoint interval.

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
import hashlib
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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




def safe_slug(text: str) -> str:
    """Return a filesystem-safe label."""
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def checkpoint_paths(
    checkpoint_dir: Path,
    run_label: str,
    graph_key: str,
    number_of_features: int,
    decoder: str,
    graph_idx: int,
    walkers_ratio: float,
    candidate_limit: Optional[int],
) -> Tuple[Path, Path]:
    """Return JSON checkpoint path and CSV progress path for one decoder/graph."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "run_label": run_label,
        "graph_key": graph_key,
        "number_of_features": number_of_features,
        "decoder": decoder,
        "graph_idx": graph_idx,
        "walkers_ratio": walkers_ratio,
        "candidate_limit": candidate_limit,
    }
    digest = hashlib.sha1(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    stem = "_".join([
        safe_slug(run_label),
        safe_slug(graph_key),
        f"{number_of_features}features",
        f"decoder_{decoder}",
        f"graph{graph_idx}",
        f"cand{candidate_limit if candidate_limit is not None else 'all'}",
        f"wr{str(walkers_ratio).replace('.', 'p')}",
        digest,
    ])
    return checkpoint_dir / f"{stem}.checkpoint.json", checkpoint_dir / f"{stem}.progress.csv"


def write_checkpoint_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write a checkpoint JSON atomically to reduce corruption risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path, planned_runs: int) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        ckpt = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Could not read checkpoint {path}: {exc}. Starting from 0.")
        return None
    if int(ckpt.get("planned_runs", -1)) != int(planned_runs):
        print(
            f"WARNING: Checkpoint {path} has planned_runs={ckpt.get('planned_runs')} "
            f"but current planned_runs={planned_runs}. Starting from 0."
        )
        return None
    return ckpt


def append_progress_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "run_label",
        "graph_key",
        "graph_family",
        "instance",
        "decoder",
        "graph_idx",
        "number_of_features_cli",
        "number_of_features_reported",
        "completed_runs",
        "planned_runs",
        "decoder_run_interval",
        "next_start",
        "best_clique_size",
        "max_clique_size_so_far",
        "elapsed_sec_total",
        "status",
        "walkers_ratio",
        "candidate_limit",
        "graph_timeout_sec",
        "checkpoint_json",
        "progress_csv",
    ]
    exists = path.exists()
    if exists:
        try:
            with path.open("r", encoding="utf-8") as f:
                existing_header = f.readline().strip().split(",")
            if existing_header != fieldnames:
                old_df = pd.read_csv(path)
                for c in fieldnames:
                    if c not in old_df.columns:
                        old_df[c] = np.nan
                old_df[fieldnames].to_csv(path, index=False)
        except Exception as exc:
            backup = path.with_suffix(path.suffix + f".backup_{int(time.time())}")
            print(f"WARNING: Could not normalize old progress CSV {path}: {exc}. Backing it up to {backup}.")
            os.replace(path, backup)
            exists = False

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def append_progress_excel(path: Optional[Path], row: Dict[str, Any]) -> None:
    """
    Maintain a combined Excel trajectory table for plotting best clique size vs.
    completed decoder runs across all graph/feature/decoder cases.

    The file is rewritten every checkpoint interval. This is intentionally simple
    and robust for overnight AWS runs. Duplicate rows from resume/retry are
    de-duplicated by run identity + completed_runs, keeping the latest row.
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "timestamp_utc",
        "case_label",
        "run_label",
        "graph_key",
        "graph_family",
        "instance",
        "decoder",
        "graph_idx",
        "number_of_features_cli",
        "number_of_features_reported",
        "completed_runs",
        "planned_runs",
        "decoder_run_interval",
        "max_clique_size_so_far",
        "best_clique_size",
        "elapsed_sec_total",
        "status",
        "walkers_ratio",
        "candidate_limit",
        "graph_timeout_sec",
        "checkpoint_json",
        "progress_csv",
    ]
    new = pd.DataFrame([{k: row.get(k) for k in cols}])
    if path.exists():
        try:
            old = pd.read_excel(path, sheet_name="progress_by_interval")
        except Exception:
            old = pd.read_excel(path)
        for c in cols:
            if c not in old.columns:
                old[c] = np.nan
        df = pd.concat([old[cols], new], ignore_index=True)
    else:
        df = new

    dedup_cols = [
        "run_label",
        "graph_key",
        "graph_idx",
        "number_of_features_cli",
        "decoder",
        "completed_runs",
        "walkers_ratio",
        "candidate_limit",
    ]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")
    df = df.sort_values(
        ["run_label", "graph_key", "number_of_features_cli", "decoder", "graph_idx", "completed_runs"],
        kind="stable",
    )

    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        df.to_excel(writer, sheet_name="progress_by_interval", index=False)


def save_progress_checkpoint(
    checkpoint_path: Path,
    progress_csv_path: Path,
    *,
    decoder: str,
    graph_idx: int,
    graph_key: str,
    number_of_features: int,
    run_label: str,
    planned_runs: int,
    next_start: int,
    best_clique_size: int,
    best_clique_nodes: List[int],
    elapsed_sec_total: float,
    status: str,
    walkers_ratio: float,
    candidate_limit: Optional[int],
    graph_timeout_sec: int,
    graph_family: str,
    instance: str,
    progress_xlsx_path: Optional[Path] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp_utc": now,
        "run_label": run_label,
        "graph_key": graph_key,
        "graph_family": graph_family,
        "instance": instance,
        "decoder": decoder,
        "graph_idx": graph_idx,
        "number_of_features": number_of_features,
        "number_of_features_cli": number_of_features,
        "number_of_features_reported": reported_feature_count(number_of_features),
        "planned_runs": int(planned_runs),
        "completed_runs": int(next_start),
        "decoder_run_interval": int(next_start),
        "next_start": int(next_start),
        "best_clique_size": int(best_clique_size),
        "max_clique_size_so_far": int(best_clique_size),
        "best_clique_nodes": [int(x) for x in best_clique_nodes],
        "elapsed_sec_total": float(elapsed_sec_total),
        "status": status,
        "walkers_ratio": float(walkers_ratio),
        "candidate_limit": None if candidate_limit is None else int(candidate_limit),
        "graph_timeout_sec": int(graph_timeout_sec),
        "checkpoint_json": str(checkpoint_path),
        "progress_csv": str(progress_csv_path),
        "case_label": (
            f"{instance} | features={reported_feature_count(number_of_features)} "
            f"| decoder={decoder} | {run_label}"
        ),
    }
    write_checkpoint_atomic(checkpoint_path, payload)
    append_progress_csv(progress_csv_path, payload)
    append_progress_excel(progress_xlsx_path, payload)


def result_size_and_nodes(result) -> Tuple[int, List[int]]:
    """Extract clique size and clique-node list when available."""
    size = extract_clique_size(result)
    nodes: List[int] = []
    if isinstance(result, (tuple, list)):
        for elem in result:
            if isinstance(elem, (list, tuple, set)):
                cand = [int(x) for x in elem]
                if len(cand) == size:
                    nodes = cand
                    break
            elif isinstance(elem, np.ndarray) and elem.ndim == 1 and elem.size == size:
                nodes = [int(x) for x in elem.tolist()]
                break
            elif torch.is_tensor(elem) and elem.ndim == 1 and elem.numel() == size:
                nodes = [int(x) for x in elem.detach().cpu().tolist()]
                break
    return int(size), nodes

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



def run_decoder_r(
    dataset,
    model,
    walkers_ratio,
    min_walkers,
    graph_timeout_sec,
    candidate_limit,
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume: bool,
    graph_key: str,
    number_of_features: int,
    run_label: str,
    graph_family: str,
    instance: str,
    progress_xlsx_path: Optional[Path],
):
    cliques, times, details = [], [], []
    model.eval()
    model.cpu()

    with torch.no_grad():
        for graph_idx, g in enumerate(tqdm(dataset, desc="Decoder R/original graphs"), start=1):
            t0 = time.time()
            x = g.x.cpu().float()
            eix = g.edge_index.cpu()
            adj_coo = to_scipy_sparse_matrix(eix, num_nodes=g.num_nodes)
            adj_ts = sparse_mx_to_torch_sparse_tensor(adj_coo).cpu()
            out = forward_model(model, x, adj_ts)

            n = int(g.num_nodes)
            k = min(max(min_walkers, int(math.floor(walkers_ratio * n))), n)
            threshold = n if candidate_limit is None else min(candidate_limit, n)
            planned_runs = min(k, threshold)
            checkpoint_path, progress_csv_path = checkpoint_paths(
                checkpoint_dir, run_label, graph_key, number_of_features, "R", graph_idx, walkers_ratio, candidate_limit
            )

            best = 0
            best_nodes: List[int] = []
            start_run = 0
            base_elapsed = 0.0
            resumed = False
            if resume:
                ckpt = load_checkpoint(checkpoint_path, planned_runs)
                if ckpt:
                    start_run = min(max(int(ckpt.get("next_start", 0)), 0), planned_runs)
                    best = int(ckpt.get("best_clique_size", 0))
                    best_nodes = [int(x) for x in ckpt.get("best_clique_nodes", [])]
                    base_elapsed = float(ckpt.get("elapsed_sec_total", 0.0))
                    resumed = start_run > 0
                    print(
                        f"[RESUME] R graph {graph_idx}: resuming from inner run {start_run:,}/{planned_runs:,}; "
                        f"best={best}; previous_elapsed={base_elapsed:.2f}s; checkpoint={checkpoint_path}"
                    )

            print(
                f"[R] Graph {graph_idx}/{len(dataset)}: n={n:,}, "
                f"walkers_ratio={walkers_ratio}, k={k:,}, "
                f"candidate_limit={candidate_limit}, threshold={threshold:,}, "
                f"planned decoder runs={planned_runs:,}, start={start_run:,}"
            )

            status = "running"
            completed = start_run
            if start_run >= planned_runs:
                status = "finished"
                print(f"[R] Checkpoint already finished: {completed:,}/{planned_runs:,}; best={best}")
            else:
                inner = tqdm(
                    range(start_run, planned_runs),
                    total=planned_runs,
                    initial=start_run,
                    desc="Decoder R/original inner runs",
                    unit="run",
                    mininterval=5,
                    dynamic_ncols=True,
                    leave=True,
                )
                for s in inner:
                    elapsed_total = base_elapsed + (time.time() - t0)
                    if elapsed_total >= graph_timeout_sec:
                        status = "timeout"
                        print(f"[TIMEOUT] R reached {graph_timeout_sec}s after {completed:,}/{planned_runs:,} runs; best={best}")
                        break
                    result = getclicnum(adj_coo, out, walkerstart=s, thresholdloopnodes=threshold)
                    csize, cnodes = result_size_and_nodes(result)
                    completed = s + 1
                    if csize > best:
                        best = csize
                        best_nodes = cnodes
                    elapsed_total = base_elapsed + (time.time() - t0)
                    inner.set_postfix(best=best, elapsed_sec=int(elapsed_total), refresh=False)

                    if checkpoint_every > 0 and (completed % checkpoint_every == 0):
                        save_progress_checkpoint(
                            checkpoint_path,
                            progress_csv_path,
                            decoder="R",
                            graph_idx=graph_idx,
                            graph_key=graph_key,
                            number_of_features=number_of_features,
                            run_label=run_label,
                            planned_runs=planned_runs,
                            next_start=completed,
                            best_clique_size=best,
                            best_clique_nodes=best_nodes,
                            elapsed_sec_total=elapsed_total,
                            status="running",
                            walkers_ratio=walkers_ratio,
                            candidate_limit=candidate_limit,
                            graph_timeout_sec=graph_timeout_sec,
                            graph_family=graph_family,
                            instance=instance,
                            progress_xlsx_path=progress_xlsx_path,
                        )

                if completed >= planned_runs:
                    status = "finished"

            elapsed_total = base_elapsed + (time.time() - t0)
            save_progress_checkpoint(
                checkpoint_path,
                progress_csv_path,
                decoder="R",
                graph_idx=graph_idx,
                graph_key=graph_key,
                number_of_features=number_of_features,
                run_label=run_label,
                planned_runs=planned_runs,
                next_start=completed,
                best_clique_size=best,
                best_clique_nodes=best_nodes,
                elapsed_sec_total=elapsed_total,
                status=status,
                walkers_ratio=walkers_ratio,
                candidate_limit=candidate_limit,
                graph_timeout_sec=graph_timeout_sec,
                graph_family=graph_family,
                instance=instance,
                progress_xlsx_path=progress_xlsx_path,
            )

            print(
                f"[R] Status={status}; completed {completed:,}/{planned_runs:,} inner runs; "
                f"best clique size={best}; elapsed_total={elapsed_total:.2f}s; checkpoint={checkpoint_path}"
            )
            cliques.append(int(best))
            times.append(float(min(elapsed_total, graph_timeout_sec)))
            details.append({
                "decoder_status": status,
                "completed_inner_runs": completed,
                "planned_inner_runs": planned_runs,
                "checkpoint_path": str(checkpoint_path),
                "progress_csv_path": str(progress_csv_path),
                "resumed_from_checkpoint": resumed,
            })
    return cliques, times, details


def run_decoder_p(
    dataset,
    model,
    walkers_ratio,
    min_walkers,
    graph_timeout_sec,
    candidate_limit,
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume: bool,
    graph_key: str,
    number_of_features: int,
    run_label: str,
    graph_family: str,
    instance: str,
    progress_xlsx_path: Optional[Path],
):
    cliques, times, details = [], [], []
    model.eval()
    model.cpu()

    with torch.no_grad():
        for graph_idx, g in enumerate(tqdm(dataset, desc="Decoder P/modified graphs"), start=1):
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
            planned_runs = min(k, len(ranked_nodes))
            checkpoint_path, progress_csv_path = checkpoint_paths(
                checkpoint_dir, run_label, graph_key, number_of_features, "P", graph_idx, walkers_ratio, candidate_limit
            )

            best = 0
            best_nodes: List[int] = []
            start_run = 0
            base_elapsed = 0.0
            resumed = False
            if resume:
                ckpt = load_checkpoint(checkpoint_path, planned_runs)
                if ckpt:
                    start_run = min(max(int(ckpt.get("next_start", 0)), 0), planned_runs)
                    best = int(ckpt.get("best_clique_size", 0))
                    best_nodes = [int(x) for x in ckpt.get("best_clique_nodes", [])]
                    base_elapsed = float(ckpt.get("elapsed_sec_total", 0.0))
                    resumed = start_run > 0
                    print(
                        f"[RESUME] P graph {graph_idx}: resuming from inner run {start_run:,}/{planned_runs:,}; "
                        f"best={best}; previous_elapsed={base_elapsed:.2f}s; checkpoint={checkpoint_path}"
                    )

            print("[P] Building neighbor sets before inner decoder loop...")
            neighbor_sets = build_neighbor_sets(adj_coo)

            print(
                f"[P] Graph {graph_idx}/{len(dataset)}: n={n:,}, "
                f"walkers_ratio={walkers_ratio}, k={k:,}, "
                f"candidate_limit={candidate_limit}, threshold={threshold:,}, "
                f"planned decoder runs={planned_runs:,}, start={start_run:,}"
            )

            status = "running"
            completed = start_run
            if start_run >= planned_runs:
                status = "finished"
                print(f"[P] Checkpoint already finished: {completed:,}/{planned_runs:,}; best={best}")
            else:
                inner = tqdm(
                    range(start_run, planned_runs),
                    total=planned_runs,
                    initial=start_run,
                    desc="Decoder P/modified inner runs",
                    unit="run",
                    mininterval=5,
                    dynamic_ncols=True,
                    leave=True,
                )
                for s in inner:
                    elapsed_total = base_elapsed + (time.time() - t0)
                    if elapsed_total >= graph_timeout_sec:
                        status = "timeout"
                        print(f"[TIMEOUT] P reached {graph_timeout_sec}s after {completed:,}/{planned_runs:,} runs; best={best}")
                        break
                    clique = greedy_clique_with_anchor_and_skips(ranked_nodes, neighbor_sets, skip_count=s)
                    completed = s + 1
                    if len(clique) > best:
                        best = len(clique)
                        best_nodes = [int(x) for x in clique]
                    elapsed_total = base_elapsed + (time.time() - t0)
                    inner.set_postfix(best=best, elapsed_sec=int(elapsed_total), refresh=False)

                    if checkpoint_every > 0 and (completed % checkpoint_every == 0):
                        save_progress_checkpoint(
                            checkpoint_path,
                            progress_csv_path,
                            decoder="P",
                            graph_idx=graph_idx,
                            graph_key=graph_key,
                            number_of_features=number_of_features,
                            run_label=run_label,
                            planned_runs=planned_runs,
                            next_start=completed,
                            best_clique_size=best,
                            best_clique_nodes=best_nodes,
                            elapsed_sec_total=elapsed_total,
                            status="running",
                            walkers_ratio=walkers_ratio,
                            candidate_limit=candidate_limit,
                            graph_timeout_sec=graph_timeout_sec,
                            graph_family=graph_family,
                            instance=instance,
                            progress_xlsx_path=progress_xlsx_path,
                        )

                if completed >= planned_runs:
                    status = "finished"

            elapsed_total = base_elapsed + (time.time() - t0)
            save_progress_checkpoint(
                checkpoint_path,
                progress_csv_path,
                decoder="P",
                graph_idx=graph_idx,
                graph_key=graph_key,
                number_of_features=number_of_features,
                run_label=run_label,
                planned_runs=planned_runs,
                next_start=completed,
                best_clique_size=best,
                best_clique_nodes=best_nodes,
                elapsed_sec_total=elapsed_total,
                status=status,
                walkers_ratio=walkers_ratio,
                candidate_limit=candidate_limit,
                graph_timeout_sec=graph_timeout_sec,
                graph_family=graph_family,
                instance=instance,
                progress_xlsx_path=progress_xlsx_path,
            )

            print(
                f"[P] Status={status}; completed {completed:,}/{planned_runs:,} inner runs; "
                f"best clique size={best}; elapsed_total={elapsed_total:.2f}s; checkpoint={checkpoint_path}"
            )
            cliques.append(int(best))
            times.append(float(min(elapsed_total, graph_timeout_sec)))
            details.append({
                "decoder_status": status,
                "completed_inner_runs": completed,
                "planned_inner_runs": planned_runs,
                "checkpoint_path": str(checkpoint_path),
                "progress_csv_path": str(progress_csv_path),
                "resumed_from_checkpoint": resumed,
            })
    return cliques, times, details


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
    ap.add_argument("--candidate_limit", type=int, default=1000)
    ap.add_argument("--graph_timeout_sec", type=int, default=10800)
    ap.add_argument("--checkpoint_every", type=int, default=10, help="Write checkpoint/progress files every N inner decoder runs. Use 0 to disable periodic checkpoints.")
    ap.add_argument("--checkpoint_dir", default=None, help="Directory for checkpoint JSON/progress CSV files. Default: <results_root>/checkpoints")
    ap.add_argument("--progress_xlsx", default=None, help="Combined Excel file for best-clique-vs-decoder-runs trajectory. Default: <results_root>/hgs_decoder_progress_by_interval.xlsx")
    ap.add_argument("--no_resume", action="store_true", help="Do not resume from existing checkpoint files; start inner decoder loops from 0.")
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
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else results_root / "checkpoints"
    progress_xlsx = Path(args.progress_xlsx).expanduser().resolve() if args.progress_xlsx else results_root / "hgs_decoder_progress_by_interval.xlsx"

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
            cliques, times, decoder_details = run_decoder_r(
                dataset,
                model,
                args.walkers_ratio,
                args.min_walkers,
                args.graph_timeout_sec,
                args.candidate_limit,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=args.checkpoint_every,
                resume=not args.no_resume,
                graph_key=graph_key,
                number_of_features=args.number_of_features,
                run_label=args.run_label,
                graph_family=family,
                instance=instance,
                progress_xlsx_path=progress_xlsx,
            )
        else:
            cliques, times, decoder_details = run_decoder_p(
                dataset,
                model,
                args.walkers_ratio,
                args.min_walkers,
                args.graph_timeout_sec,
                args.candidate_limit,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=args.checkpoint_every,
                resume=not args.no_resume,
                graph_key=graph_key,
                number_of_features=args.number_of_features,
                run_label=args.run_label,
                graph_family=family,
                instance=instance,
                progress_xlsx_path=progress_xlsx,
            )

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
        completed_inner_runs = int(np.sum([d.get("completed_inner_runs", 0) for d in decoder_details])) if decoder_details else None
        planned_inner_runs = int(np.sum([d.get("planned_inner_runs", 0) for d in decoder_details])) if decoder_details else None
        decoder_statuses = ";".join(str(d.get("decoder_status")) for d in decoder_details) if decoder_details else None
        checkpoint_files = ";".join(str(d.get("checkpoint_path")) for d in decoder_details) if decoder_details else None
        progress_csv_files = ";".join(str(d.get("progress_csv_path")) for d in decoder_details) if decoder_details else None
        resumed_from_checkpoint = any(bool(d.get("resumed_from_checkpoint")) for d in decoder_details) if decoder_details else False
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
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_dir": str(checkpoint_dir),
            "completed_inner_runs": completed_inner_runs,
            "planned_inner_runs": planned_inner_runs,
            "decoder_statuses": decoder_statuses,
            "checkpoint_files": checkpoint_files,
            "progress_csv_files": progress_csv_files,
            "progress_xlsx": str(progress_xlsx),
            "resumed_from_checkpoint": resumed_from_checkpoint,
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
            "checkpoint_dir": str(checkpoint_dir),
            "progress_xlsx": str(progress_xlsx),
        },
    }
    meta_path = run_meta_dir / f"hgs_large_graph_decoder_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
