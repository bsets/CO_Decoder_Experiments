from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve
import urllib.request

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism helps reproducibility. It may slow training.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------


def download_file(url: str, dest: Path, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} -> {dest} [attempt {attempt}/{retries}]")
            tmp = dest.with_suffix(dest.suffix + ".partial")
            if tmp.exists():
                tmp.unlink()
            urlretrieve(url, tmp)
            tmp.rename(dest)
            return dest
        except Exception as exc:
            print(f"Download failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(5)

    return dest


# ---------------------------------------------------------------------
# TU Dortmund dataset loader for IMDB-BINARY and COLLAB
# ---------------------------------------------------------------------


TU_URLS = {
    "imdb_binary": "https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip",
    "collab": "https://www.chrsmrrs.com/graphkerneldatasets/COLLAB.zip",
}


def load_tu_graphs(
    dataset_name: str,
    data_root: Path,
    limit: Optional[int] = None,
) -> List[nx.Graph]:
    key = dataset_name.lower()
    if key not in TU_URLS:
        raise ValueError(f"Unsupported TU dataset: {dataset_name}")

    zip_path = data_root / "raw" / f"{dataset_name}.zip"
    extract_dir = data_root / "raw" / dataset_name

    download_file(TU_URLS[key], zip_path)

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    nested_dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    base = nested_dirs[0] if nested_dirs else extract_dir

    prefix = {
        "imdb_binary": "IMDB-BINARY",
        "collab": "COLLAB",
    }[key]

    indicator_file = base / f"{prefix}_graph_indicator.txt"
    edges_file = base / f"{prefix}_A.txt"

    if not indicator_file.exists() or not edges_file.exists():
        raise FileNotFoundError(f"Expected TU files not found in {base}")

    node_to_graph: Dict[int, int] = {}
    with indicator_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            node_to_graph[idx] = int(line.strip())

    graph_ids = sorted(set(node_to_graph.values()))
    if limit is not None:
        graph_ids = graph_ids[:limit]
    graph_id_set = set(graph_ids)

    graphs_by_id: Dict[int, nx.Graph] = {gid: nx.Graph() for gid in graph_ids}
    local_id_maps: Dict[int, Dict[int, int]] = {gid: {} for gid in graph_ids}

    for global_node, gid in node_to_graph.items():
        if gid not in graph_id_set:
            continue
        local_idx = len(local_id_maps[gid])
        local_id_maps[gid][global_node] = local_idx
        graphs_by_id[gid].add_node(local_idx)

    with edges_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.replace(" ", "").strip().split(",")
            if len(parts) != 2:
                continue

            u_global, v_global = int(parts[0]), int(parts[1])
            gid_u = node_to_graph.get(u_global)
            gid_v = node_to_graph.get(v_global)

            if gid_u != gid_v or gid_u not in graph_id_set:
                continue

            u = local_id_maps[gid_u][u_global]
            v = local_id_maps[gid_u][v_global]
            if u != v:
                graphs_by_id[gid_u].add_edge(u, v)

    graphs: List[nx.Graph] = []
    for gid in graph_ids:
        G = nx.convert_node_labels_to_integers(graphs_by_id[gid])
        G.graph["dataset"] = key
        G.graph["source_graph_id"] = gid
        graphs.append(G)

    return graphs


# ---------------------------------------------------------------------
# SNAP Twitter ego-network loader
# ---------------------------------------------------------------------


def load_twitter_ego_graphs(data_root: Path, limit: Optional[int] = None) -> List[nx.Graph]:
    url = "https://snap.stanford.edu/data/twitter.tar.gz"
    tar_path = data_root / "raw" / "twitter.tar.gz"
    extract_dir = data_root / "raw" / "twitter"

    download_file(url, tar_path)

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path} -> {extract_dir}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)

    edge_files = sorted(extract_dir.rglob("*.edges"))
    if limit is not None:
        edge_files = edge_files[:limit]

    graphs: List[nx.Graph] = []
    for edge_file in edge_files:
        ego_id = edge_file.stem
        G = nx.Graph()

        with edge_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                u, v = parts
                G.add_edge(u, v)

        # The SNAP ego node is not listed in the edge file; add it and connect
        # it to every node appearing in the ego network.
        ego_node = f"ego_{ego_id}"
        for node in list(G.nodes()):
            G.add_edge(ego_node, node)

        G = nx.convert_node_labels_to_integers(G)
        G.graph["dataset"] = "twitter"
        G.graph["source_graph_id"] = ego_id
        graphs.append(G)

    return graphs


# ---------------------------------------------------------------------
# RB-style compatibility graph generator
# ---------------------------------------------------------------------


def generate_forced_rb_clique_graph(
    n_variables: int,
    domain_size: int,
    r: float,
    p: float,
    seed: int,
) -> nx.Graph:
    rng = random.Random(seed)
    n = n_variables
    d = domain_size

    planted = {i: rng.randrange(d) for i in range(n)}
    variable_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = min(len(variable_pairs), int(round(r * n * math.log(max(n, 2)))))
    constrained_pairs = rng.sample(variable_pairs, m)

    forbidden = {}
    for i, j in constrained_pairs:
        all_tuples = [(a, b) for a in range(d) for b in range(d)]
        planted_tuple = (planted[i], planted[j])
        all_tuples.remove(planted_tuple)
        t = min(len(all_tuples), int(round(p * d * d)))
        forbidden[(i, j)] = set(rng.sample(all_tuples, t))

    G = nx.Graph()
    for i in range(n):
        for a in range(d):
            G.add_node((i, a), variable=i, value=a)

    for i in range(n):
        for j in range(i + 1, n):
            blocked = forbidden.get((i, j), set())
            for a in range(d):
                for b in range(d):
                    if (a, b) not in blocked:
                        G.add_edge((i, a), (j, b))

    G = nx.convert_node_labels_to_integers(G)
    G.graph["planted_clique_size"] = n
    G.graph["n_variables"] = n
    G.graph["domain_size"] = d
    G.graph["r"] = r
    G.graph["p"] = p
    G.graph["seed"] = seed
    return G


def generate_rb_dataset(name: str, count: int, seed: int) -> List[nx.Graph]:
    rng = random.Random(seed)
    graphs: List[nx.Graph] = []

    if name == "rb_small":
        specs = [(20, 10), (25, 12), (30, 15)]
    elif name == "rb_large":
        specs = [(40, 20), (50, 20), (60, 25)]
    else:
        raise ValueError(name)

    for idx in range(count):
        n, d = specs[idx % len(specs)]
        G = generate_forced_rb_clique_graph(
            n_variables=n,
            domain_size=d,
            r=0.8,
            p=0.25,
            seed=rng.randint(0, 10_000_000),
        )
        G.graph["dataset"] = name
        G.graph["source_graph_id"] = idx
        graphs.append(G)

    return graphs


# ---------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------


def assign_splits(
    graphs: List[nx.Graph],
    seed: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> Dict[str, List[nx.Graph]]:
    rng = random.Random(seed)
    idx = list(range(len(graphs)))
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return {
        "train": [graphs[i] for i in train_idx],
        "validation": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }


# ---------------------------------------------------------------------
# Dense GCN model
# ---------------------------------------------------------------------


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return self.linear(a_norm @ x)


class EGNStyleGCN(nn.Module):
    def __init__(self, in_dim: int = 3, hidden_dim: int = 64, layers: int = 3):
        super().__init__()

        gcn_layers = []
        dims = [in_dim] + [hidden_dim] * layers
        for i in range(layers):
            gcn_layers.append(DenseGCNLayer(dims[i], dims[i + 1]))

        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.gcn_layers:
            h = torch.relu(layer(h, a_norm))
        logits = self.out(h).squeeze(-1)
        return torch.sigmoid(logits)


def graph_to_tensors(
    G: nx.Graph,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = G.number_of_nodes()

    adj_np = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32)
    adj = torch.tensor(adj_np, dtype=torch.float32, device=device)

    deg = adj.sum(dim=1)
    deg_norm = deg / max(float(n - 1), 1.0)
    density = torch.full((n,), nx.density(G), dtype=torch.float32, device=device)

    x = torch.stack(
        [
            torch.ones(n, dtype=torch.float32, device=device),
            deg_norm,
            density,
        ],
        dim=1,
    )

    a_tilde = adj + torch.eye(n, device=device)
    d = a_tilde.sum(dim=1)
    d_inv_sqrt = torch.pow(d.clamp(min=1.0), -0.5)
    a_norm = d_inv_sqrt[:, None] * a_tilde * d_inv_sqrt[None, :]

    return x, adj, a_norm


# ---------------------------------------------------------------------
# EGN-style maximum-clique loss and author-style decoder
# ---------------------------------------------------------------------


def maxclique_egn_loss(probs: torch.Tensor, adj: torch.Tensor, beta: float) -> torch.Tensor:
    n = probs.shape[0]
    pp = probs[:, None] * probs[None, :]

    upper = torch.triu(torch.ones((n, n), device=probs.device), diagonal=1)
    edge_mask = upper * adj
    nonedge_mask = upper * (1.0 - adj)

    expected_edges = (edge_mask * pp).sum()
    expected_nonedges = (nonedge_mask * pp).sum()

    return -expected_edges + beta * expected_nonedges


def loss_np(p: np.ndarray, adj: np.ndarray, beta: float) -> float:
    n = len(p)
    pp = np.outer(p, p)
    upper = np.triu(np.ones((n, n), dtype=np.float32), k=1)
    edge_mask = upper * adj
    nonedge_mask = upper * (1.0 - adj)
    return float(-(edge_mask * pp).sum() + beta * (nonedge_mask * pp).sum())


def repair_to_clique(selected: List[int], probs: np.ndarray, adj: np.ndarray) -> List[int]:
    if len(selected) == 0:
        return [int(np.argmax(probs))]

    ordered = sorted(selected, key=lambda i: probs[i], reverse=True)
    clique: List[int] = []
    for v in ordered:
        if all(adj[v, u] > 0.5 for u in clique):
            clique.append(v)

    if len(clique) == 0:
        clique = [int(np.argmax(probs))]

    return clique


def egn_author_decoder(probs: np.ndarray, adj: np.ndarray, beta: float) -> List[int]:
    """
    EGN-style conditional-expectation decoder.

    It sorts nodes by predicted probability and derandomizes by checking
    whether fixing p_i to 1 or 0 gives lower unsupervised loss. A final greedy
    repair guarantees the returned set is a valid clique.
    """
    p_work = probs.copy()
    order = np.argsort(-probs)

    for i in order:
        p1 = p_work.copy()
        p0 = p_work.copy()
        p1[i] = 1.0
        p0[i] = 0.0

        if loss_np(p1, adj, beta) <= loss_np(p0, adj, beta):
            p_work[i] = 1.0
        else:
            p_work[i] = 0.0

    selected = [int(i) for i in np.where(p_work >= 0.5)[0]]
    return repair_to_clique(selected, probs, adj)


def is_valid_clique(nodes: List[int], adj: np.ndarray) -> bool:
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adj[nodes[i], nodes[j]] < 0.5:
                return False
    return True


def exact_clique_size_if_small(G: nx.Graph, max_nodes: int) -> Tuple[Optional[int], str]:
    if G.number_of_nodes() > max_nodes:
        return None, "not_computed_large_graph"

    max_size = 0
    for clique in nx.find_cliques(G):
        max_size = max(max_size, len(clique))

    return max_size, "exact_networkx_find_cliques"


# ---------------------------------------------------------------------
# Results and cost schemas
# ---------------------------------------------------------------------


@dataclass
class GraphResult:
    dataset: str
    graph_id: str
    split: str
    algorithm: str
    decoder: str
    num_nodes: int
    num_edges: int
    density: float
    clique_size: int
    is_valid_clique: bool
    reference_clique_size: Optional[int]
    reference_type: str
    approx_ratio: Optional[float]
    model_time_seconds: float
    decode_time_seconds: float
    total_time_seconds: float
    seed: int
    epoch_selected: int


@dataclass
class CostRecord:
    run_id: str
    dataset: str
    split: str
    algorithm: str
    decoder: str
    operation: str
    instance_type: str
    region: str
    device: str
    seconds: float
    ec2_hourly_usd: float
    ec2_cost_per_second_usd: float
    estimated_compute_cost_usd: float
    seed: int
    notes: str


# ---------------------------------------------------------------------
# EC2 context and operation-level cost tracking
# ---------------------------------------------------------------------


def _imds_v2_token(timeout_seconds: float = 0.2) -> Optional[str]:
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except Exception:
        return None


def _read_ec2_metadata(
    path: str,
    token: Optional[str],
    timeout_seconds: float = 0.2,
) -> Optional[str]:
    try:
        headers = {}
        if token:
            headers["X-aws-ec2-metadata-token"] = token

        req = urllib.request.Request(
            f"http://169.254.169.254/latest/{path}",
            headers=headers,
        )

        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except Exception:
        return None


def detect_ec2_context(cost_cfg: dict) -> dict:
    manual_instance_type = cost_cfg.get("instance_type")
    manual_region = cost_cfg.get("region")

    token = _imds_v2_token()

    instance_type = manual_instance_type
    if not instance_type:
        instance_type = _read_ec2_metadata("meta-data/instance-type", token)

    region = manual_region
    if not region:
        identity_doc = _read_ec2_metadata("dynamic/instance-identity/document", token)
        if identity_doc:
            try:
                region = json.loads(identity_doc).get("region")
            except json.JSONDecodeError:
                region = None

    return {
        "instance_type": instance_type or "unknown",
        "region": region or "unknown",
    }


def make_cost_record(
    run_id: str,
    dataset: str,
    split: str,
    algorithm: str,
    decoder: str,
    operation: str,
    instance_type: str,
    region: str,
    device: str,
    seconds: float,
    ec2_hourly_usd: float,
    seed: int,
    notes: str = "",
) -> CostRecord:
    cost_per_second = ec2_hourly_usd / 3600.0
    estimated_cost = seconds * cost_per_second

    return CostRecord(
        run_id=run_id,
        dataset=dataset,
        split=split,
        algorithm=algorithm,
        decoder=decoder,
        operation=operation,
        instance_type=instance_type,
        region=region,
        device=device,
        seconds=float(seconds),
        ec2_hourly_usd=float(ec2_hourly_usd),
        ec2_cost_per_second_usd=float(cost_per_second),
        estimated_compute_cost_usd=float(estimated_cost),
        seed=seed,
        notes=notes,
    )


def write_cost_outputs(
    output_dir: Path,
    cost_records: List[CostRecord],
    run_id: str,
    run_wall_time_seconds: float,
    ec2_hourly_usd: float,
    ec2_context: dict,
    cost_tags: dict,
) -> None:
    if not cost_records:
        return

    cost_df = pd.DataFrame([asdict(r) for r in cost_records])
    cost_df.to_csv(output_dir / "operation_costs.csv", index=False)

    summary = (
        cost_df.groupby(["dataset", "algorithm", "decoder", "operation"], dropna=False)
        .agg(
            seconds=("seconds", "sum"),
            estimated_compute_cost_usd=("estimated_compute_cost_usd", "sum"),
            rows=("operation", "count"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "operation_costs_summary.csv", index=False)

    total_core_seconds = float(cost_df["seconds"].sum())
    total_core_cost = float(cost_df["estimated_compute_cost_usd"].sum())
    run_wall_cost = run_wall_time_seconds * (ec2_hourly_usd / 3600.0)

    run_cost_summary = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance_type": ec2_context.get("instance_type", "unknown"),
        "region": ec2_context.get("region", "unknown"),
        "ec2_hourly_usd": ec2_hourly_usd,
        "ec2_cost_per_second_usd": ec2_hourly_usd / 3600.0,
        "run_wall_time_seconds": run_wall_time_seconds,
        "estimated_run_wall_clock_ec2_cost_usd": run_wall_cost,
        "core_operation_seconds": total_core_seconds,
        "estimated_core_operation_ec2_cost_usd": total_core_cost,
        "core_operation_fraction_of_run_wall_time": (
            total_core_seconds / run_wall_time_seconds
            if run_wall_time_seconds > 0
            else None
        ),
        "core_operation_fraction_of_run_wall_clock_ec2_cost": (
            total_core_cost / run_wall_cost
            if run_wall_cost > 0
            else None
        ),
        "aws_total_tagged_cost_usd": None,
        "core_operation_fraction_of_aws_total_tagged_cost": None,
        "cost_tags": cost_tags,
        "notes": [
            "This estimates EC2 compute cost for measured training, inference, and decoding time.",
            "It excludes EBS storage, data transfer, idle instance time outside this Python run, snapshots, CloudWatch, and other AWS charges.",
            "Fill aws_total_tagged_cost_usd later from AWS Cost Explorer to compute the fraction of total tagged AWS cost.",
        ],
    }

    (output_dir / "run_cost_summary.json").write_text(
        json.dumps(run_cost_summary, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_graphs: List[nx.Graph],
    val_graphs: List[nx.Graph],
    device: torch.device,
    epochs: int,
    lr: float,
    beta: float,
) -> Tuple[nn.Module, int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_graphs)
        train_losses = []

        for G in train_graphs:
            x, adj, a_norm = graph_to_tensors(G, device)
            optimizer.zero_grad()
            probs = model(x, a_norm)
            loss = maxclique_egn_loss(probs, adj, beta)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for G in val_graphs:
                x, adj, a_norm = graph_to_tensors(G, device)
                probs = model(x, a_norm)
                loss = maxclique_egn_loss(probs, adj, beta)
                val_losses.append(float(loss.detach().cpu()))

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else mean_train

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch={epoch:04d} train_loss={mean_train:.4f} val_loss={mean_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_val_loss


def evaluate_graphs(
    model: nn.Module,
    graphs: List[nx.Graph],
    split: str,
    device: torch.device,
    beta: float,
    seed: int,
    epoch_selected: int,
    exact_max_nodes: int,
) -> List[GraphResult]:
    model.eval()
    results: List[GraphResult] = []

    with torch.no_grad():
        for G in tqdm(graphs, desc=f"evaluate {split}"):
            dataset = G.graph.get("dataset", "unknown")
            graph_id = str(G.graph.get("source_graph_id", "unknown"))

            x, adj_t, a_norm = graph_to_tensors(G, device)
            adj = adj_t.detach().cpu().numpy()

            sync_if_cuda(device)
            t0 = time.time()
            probs_t = model(x, a_norm)
            sync_if_cuda(device)
            model_time = time.time() - t0

            probs = probs_t.detach().cpu().numpy()

            t1 = time.time()
            clique_nodes = egn_author_decoder(probs, adj, beta)
            decode_time = time.time() - t1

            ref, ref_type = exact_clique_size_if_small(G, exact_max_nodes)
            approx = None if ref is None or ref == 0 else len(clique_nodes) / ref

            results.append(
                GraphResult(
                    dataset=dataset,
                    graph_id=graph_id,
                    split=split,
                    algorithm="egn_style",
                    decoder="egn_author_conditional_expectation_repaired",
                    num_nodes=G.number_of_nodes(),
                    num_edges=G.number_of_edges(),
                    density=nx.density(G),
                    clique_size=len(clique_nodes),
                    is_valid_clique=is_valid_clique(clique_nodes, adj),
                    reference_clique_size=ref,
                    reference_type=ref_type,
                    approx_ratio=approx,
                    model_time_seconds=model_time,
                    decode_time_seconds=decode_time,
                    total_time_seconds=model_time + decode_time,
                    seed=seed,
                    epoch_selected=epoch_selected,
                )
            )

    return results


def load_dataset_by_name(
    name: str,
    data_root: Path,
    limit: Optional[int],
    seed: int,
) -> List[nx.Graph]:
    if name in {"imdb_binary", "collab"}:
        return load_tu_graphs(name, data_root=data_root, limit=limit)
    if name == "twitter":
        return load_twitter_ego_graphs(data_root=data_root, limit=limit)
    if name == "rb_small":
        return generate_rb_dataset("rb_small", count=limit or 30, seed=seed)
    if name == "rb_large":
        return generate_rb_dataset("rb_large", count=limit or 15, seed=seed)
    raise ValueError(f"Unknown dataset: {name}")


def optional_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_start_time = time.time()

    run_id = cfg["experiment"]["run_id"]
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    cost_cfg = cfg.get("cost", {})
    ec2_hourly_usd = optional_float(cost_cfg.get("ec2_hourly_usd"), default=0.0)
    cost_tags = cost_cfg.get("cost_tags", {})
    ec2_context = detect_ec2_context(cost_cfg)

    if ec2_hourly_usd <= 0:
        print(
            "WARNING: cost.ec2_hourly_usd is not set or is <= 0. "
            "Cost outputs will show $0. Set this in the YAML config."
        )

    output_dir = Path(cfg["output"]["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data"]["data_root"])
    dataset_names = cfg["data"]["datasets"]
    dataset_limit = cfg["data"].get("limit_per_dataset")

    beta = float(cfg["model"]["beta"])
    epochs = int(cfg["model"]["epochs"])
    lr = float(cfg["model"]["learning_rate"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    layers = int(cfg["model"]["layers"])
    exact_max_nodes = int(cfg["evaluation"]["exact_max_nodes"])

    requested_cuda = bool(cfg.get("compute", {}).get("use_cuda", True))
    device = torch.device("cuda" if torch.cuda.is_available() and requested_cuda else "cpu")

    print(f"Using device: {device}")
    print(f"EC2 context: {ec2_context}")
    print(f"EC2 hourly USD: {ec2_hourly_usd}")

    all_results: List[GraphResult] = []
    cost_records: List[CostRecord] = []

    for dataset_name in dataset_names:
        print(f"\n=== Dataset: {dataset_name} ===")
        graphs = load_dataset_by_name(
            dataset_name,
            data_root=data_root,
            limit=dataset_limit,
            seed=seed,
        )

        # RB-large is treated as OOD test in this first design.
        # The model is trained on RB-small and evaluated on RB-large.
        if dataset_name == "rb_large":
            splits = {"ood_test": graphs}
            rb_small_graphs = load_dataset_by_name(
                "rb_small",
                data_root=data_root,
                limit=dataset_limit,
                seed=seed,
            )
            train_splits = assign_splits(rb_small_graphs, seed=seed)
            training_note = "Training on RB-small, evaluating RB-large as OOD test."
        else:
            splits = assign_splits(graphs, seed=seed)
            train_splits = splits
            training_note = "Training on this dataset's train split."

        model = EGNStyleGCN(
            in_dim=3,
            hidden_dim=hidden_dim,
            layers=layers,
        ).to(device)

        sync_if_cuda(device)
        training_start = time.time()

        model, best_epoch, best_val_loss = train_model(
            model=model,
            train_graphs=train_splits["train"],
            val_graphs=train_splits["validation"],
            device=device,
            epochs=epochs,
            lr=lr,
            beta=beta,
        )

        sync_if_cuda(device)
        training_seconds = time.time() - training_start

        cost_records.append(
            make_cost_record(
                run_id=run_id,
                dataset=dataset_name,
                split="train",
                algorithm="egn_style",
                decoder="not_applicable_training",
                operation="training",
                instance_type=ec2_context["instance_type"],
                region=ec2_context["region"],
                device=str(device),
                seconds=training_seconds,
                ec2_hourly_usd=ec2_hourly_usd,
                seed=seed,
                notes=training_note,
            )
        )

        model_path = output_dir / f"egn_style_{dataset_name}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset": dataset_name,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "config": cfg,
            },
            model_path,
        )

        for split_name, split_graphs in splits.items():
            split_results = evaluate_graphs(
                model=model,
                graphs=split_graphs,
                split=split_name,
                device=device,
                beta=beta,
                seed=seed,
                epoch_selected=best_epoch,
                exact_max_nodes=exact_max_nodes,
            )

            all_results.extend(split_results)

            inference_seconds = sum(r.model_time_seconds for r in split_results)
            decoding_seconds = sum(r.decode_time_seconds for r in split_results)

            cost_records.append(
                make_cost_record(
                    run_id=run_id,
                    dataset=dataset_name,
                    split=split_name,
                    algorithm="egn_style",
                    decoder="egn_author_conditional_expectation_repaired",
                    operation="inference_forward_pass",
                    instance_type=ec2_context["instance_type"],
                    region=ec2_context["region"],
                    device=str(device),
                    seconds=inference_seconds,
                    ec2_hourly_usd=ec2_hourly_usd,
                    seed=seed,
                    notes="Sum of GNN forward-pass time over graphs in this split.",
                )
            )

            cost_records.append(
                make_cost_record(
                    run_id=run_id,
                    dataset=dataset_name,
                    split=split_name,
                    algorithm="egn_style",
                    decoder="egn_author_conditional_expectation_repaired",
                    operation="decoding",
                    instance_type=ec2_context["instance_type"],
                    region=ec2_context["region"],
                    device=str(device),
                    seconds=decoding_seconds,
                    ec2_hourly_usd=ec2_hourly_usd,
                    seed=seed,
                    notes="Sum of decoder-only time over graphs in this split.",
                )
            )

    if not all_results:
        raise RuntimeError("No graph results were generated. Check dataset config.")

    results_csv = output_dir / "egn_author_decoder_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for row in all_results:
            writer.writerow(asdict(row))

    df = pd.DataFrame([asdict(r) for r in all_results])
    summary = (
        df.groupby(["dataset", "split", "algorithm", "decoder"], dropna=False)
        .agg(
            graphs=("graph_id", "count"),
            mean_clique_size=("clique_size", "mean"),
            median_clique_size=("clique_size", "median"),
            valid_rate=("is_valid_clique", "mean"),
            mean_approx_ratio=("approx_ratio", "mean"),
            mean_model_time_seconds=("model_time_seconds", "mean"),
            mean_decode_time_seconds=("decode_time_seconds", "mean"),
            mean_total_time_seconds=("total_time_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "egn_author_decoder_summary.csv", index=False)

    run_wall_time_seconds = time.time() - run_start_time

    write_cost_outputs(
        output_dir=output_dir,
        cost_records=cost_records,
        run_id=run_id,
        run_wall_time_seconds=run_wall_time_seconds,
        ec2_hourly_usd=ec2_hourly_usd,
        ec2_context=ec2_context,
        cost_tags=cost_tags,
    )

    metadata = {
        "run_id": run_id,
        "seed": seed,
        "device": str(device),
        "ec2_context": ec2_context,
        "ec2_hourly_usd": ec2_hourly_usd,
        "ec2_cost_per_second_usd": ec2_hourly_usd / 3600.0,
        "run_wall_time_seconds": run_wall_time_seconds,
        "cost_tags": cost_tags,
        "config": cfg,
        "outputs": {
            "results_csv": str(results_csv),
            "summary_csv": str(output_dir / "egn_author_decoder_summary.csv"),
            "operation_costs_csv": str(output_dir / "operation_costs.csv"),
            "operation_costs_summary_csv": str(output_dir / "operation_costs_summary.csv"),
            "run_cost_summary_json": str(output_dir / "run_cost_summary.json"),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone. Results written to: {output_dir}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary CSV: {output_dir / 'egn_author_decoder_summary.csv'}")
    print(f"Cost records: {output_dir / 'operation_costs.csv'}")
    print(f"Cost summary: {output_dir / 'run_cost_summary.json'}")


if __name__ == "__main__":
    main()
