# QGINE_v10.py
# -----------------------------------------------------------------------------
# Hybrid model: AngleEquivariantGINE_v8 classical backbone
#               + QuantumReuploadingHead correction term (from QGINE_v2).
#
# Architecture:
#   final_pred = base_pred + q_scale * q_pred
#   where:
#     base_pred  = RealGINEop2_v8_Model (equivariant + 3-body + bond-GINE)
#     q_pred     = QuantumReuploadingHead(angle_head(g))
#     q_scale    = learned scalar, initialized at 0.0
#
# Training stages (all end-to-end, all stages update via backprop):
#   1. "base"    : freeze quantum head / angle_head / q_scale, train classical
#   2. "quantum" : optionally freeze base, train quantum components
#   3. "finetune": optional low-LR joint training of everything
#
# Key design decisions:
#   - q_scale init=0.0 → model starts as pure classical, quantum wakes up gradually
#   - quantum head kept on CPU; base on CUDA; QGINE_v10.forward unifies devices
#   - All V10 training features preserved: EMA, AMP/BF16, cosine scheduler,
#     noisy nodes, edge dropout, virtual node, torch.compile
#   - return_parts=True on forward exposes (pred, base_pred, q_pred, theta, g)
#     for per-epoch diagnostic logging of q_scale and quantum contribution
# -----------------------------------------------------------------------------

import os
import math
import random
import csv
import argparse
import contextlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, GINEConv
from torch_geometric.nn import radius_graph, knn_graph

import pennylane as qml
import time
import sys


# =============================================================================
# Logging utilities
# =============================================================================

_LOG_FILE = None  # set once in main()

def _init_run_log(path: str):
    global _LOG_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _LOG_FILE = open(path, "a", buffering=1)  # line-buffered

def _close_run_log():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None

def log(msg: str):
    """Print to stdout AND the run log file."""
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")

def log_separator(char="=", width=80):
    log(char * width)

def log_section(title: str):
    log_separator()
    log(f"  {title}")
    log_separator()

def _gpu_mem_str() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv  = torch.cuda.memory_reserved()  / 1024**3
        return f"GPU mem: {alloc:.2f}GB alloc / {resv:.2f}GB reserved"
    return "GPU: N/A"

def _param_count(model: "nn.Module") -> Dict[str, int]:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# =============================================================================
# Reproducibility
# =============================================================================

def seed_all(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def _init_fn(worker_id: int) -> None:
        s = int(base_seed) + int(worker_id)
        random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    return _init_fn


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Convenience wrapper to freeze/unfreeze an entire module."""
    for p in module.parameters():
        p.requires_grad = flag


# =============================================================================
# RBF encoding
# =============================================================================

class RBF(nn.Module):
    """Gaussian RBF: dist [E] -> [E, K], always FP32."""
    def __init__(self, r_min: float = 0.0, r_max: float = 5.0, n_bins: int = 32):
        super().__init__()
        centers = torch.linspace(r_min, r_max, n_bins, dtype=torch.float32)
        self.register_buffer("centers", centers)
        delta = (r_max - r_min) / max(n_bins - 1, 1)
        gamma = 1.0 / (2.0 * (delta ** 2 + 1e-12))
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        dist = dist.to(dtype=torch.float32)
        diff = dist - self.centers.view(1, -1)
        return torch.exp(-self.gamma * diff * diff)


# =============================================================================
# Cache key + meta helpers
# =============================================================================

@dataclass(frozen=True)
class Op2CacheKey:
    graph_type: str
    radius: float
    knn_k: int
    max_num_neighbors: int

    def dirname(self) -> str:
        return (
            f"op2_{self.graph_type}_rad{self.radius:g}"
            f"_k{self.knn_k}_mnn{self.max_num_neighbors}"
        )


def _write_meta(meta_path: str, meta: Dict[str, str]) -> None:
    with open(meta_path, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")


def _read_meta(meta_path: str) -> Optional[Dict[str, str]]:
    if not os.path.exists(meta_path):
        return None
    out: Dict[str, str] = {}
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _assert_cache_meta_matches(
    meta: Dict[str, str],
    *,
    graph_type: str,
    radius: float,
    knn_k: int,
    max_num_neighbors: int,
    n_graphs: int,
) -> None:
    def _get(k):
        if k not in meta:
            raise RuntimeError(f"[Cache] meta.txt missing key '{k}'.")
        return meta[k]

    exp = {
        "graph_type":        str(graph_type),
        "radius":            str(float(radius)),
        "knn_k":             str(int(knn_k)),
        "max_num_neighbors": str(int(max_num_neighbors)),
        "num_graphs":        str(int(n_graphs)),
        "format":            "edge_index+edge_dist+triplets_v1",
    }
    got = {
        "graph_type":        _get("graph_type"),
        "radius":            str(float(_get("radius"))),
        "knn_k":             str(int(_get("knn_k"))),
        "max_num_neighbors": str(int(_get("max_num_neighbors"))),
        "num_graphs":        str(int(_get("num_graphs"))),
        "format":            _get("format"),
    }
    if got != exp:
        raise RuntimeError(
            f"[Cache] meta mismatch.\n  expected: {exp}\n  got: {got}\n"
            "Delete cache_dir and rerun."
        )


# =============================================================================
# Triplet index builder
# =============================================================================

@torch.no_grad()
def build_triplet_index(
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (kj_idx, ji_idx) for all triplets (k->j, j->i) with k != i.
    Must be defined BEFORE build_or_load_edge_cache.
    """
    if edge_index.numel() == 0:
        z = torch.zeros((0,), dtype=torch.long)
        return z, z

    src, dst = edge_index
    E = edge_index.size(1)
    num_nodes = int(max(src.max().item(), dst.max().item())) + 1

    incoming: List[List[int]] = [[] for _ in range(num_nodes)]
    outgoing: List[List[int]] = [[] for _ in range(num_nodes)]
    for e in range(E):
        incoming[dst[e].item()].append(e)
        outgoing[src[e].item()].append(e)

    kj_list: List[int] = []
    ji_list: List[int] = []
    for j in range(num_nodes):
        for kj in incoming[j]:
            k = src[kj].item()
            for ji in outgoing[j]:
                i = dst[ji].item()
                if k != i:
                    kj_list.append(kj)
                    ji_list.append(ji)

    if not kj_list:
        z = torch.zeros((0,), dtype=torch.long)
        return z, z

    return (
        torch.tensor(kj_list, dtype=torch.long),
        torch.tensor(ji_list, dtype=torch.long),
    )


# =============================================================================
# Edge + triplet cache
# =============================================================================

@torch.no_grad()
def build_or_load_edge_cache(
    dataset: QM9,
    cache_dir: str,
    *,
    graph_type: str,
    radius: float,
    knn_k: int,
    max_num_neighbors: int,
) -> Tuple[List, List, List, List]:
    """Returns (edge_index_list, edge_dist_list, triplet_kj_list, triplet_ji_list)."""
    os.makedirs(cache_dir, exist_ok=True)
    ei_path  = os.path.join(cache_dir, "edge_index_list.pt")
    ed_path  = os.path.join(cache_dir, "edge_dist_list.pt")
    kj_path  = os.path.join(cache_dir, "triplet_kj_list.pt")
    ji_path  = os.path.join(cache_dir, "triplet_ji_list.pt")
    meta_path = os.path.join(cache_dir, "meta.txt")

    if all(os.path.exists(p) for p in [ei_path, ed_path, kj_path, ji_path]):
        meta = _read_meta(meta_path)
        if meta is None:
            raise RuntimeError("[Cache] Found tensors but missing meta.txt. Delete cache dir.")
        _assert_cache_meta_matches(
            meta,
            graph_type=graph_type, radius=radius,
            knn_k=knn_k, max_num_neighbors=max_num_neighbors,
            n_graphs=len(dataset),
        )
        ei_list = torch.load(ei_path, map_location="cpu", weights_only=True)
        ed_list = torch.load(ed_path, map_location="cpu", weights_only=True)
        kj_list = torch.load(kj_path, map_location="cpu", weights_only=True)
        ji_list = torch.load(ji_path, map_location="cpu", weights_only=True)
        if len(ei_list) != len(dataset):
            raise RuntimeError("[Cache] Length mismatch.")
        print(f"[Cache] Loaded edges+triplets from: {cache_dir}")
        return ei_list, ed_list, kj_list, ji_list

    print(f"[Cache] Building edges+triplets for {len(dataset)} molecules...")
    old_tf = getattr(dataset, "transform", None)
    dataset.transform = None

    ei_list: List[torch.Tensor] = []
    ed_list: List[torch.Tensor] = []
    kj_list: List[torch.Tensor] = []
    ji_list: List[torch.Tensor] = []

    for i in range(len(dataset)):
        data = dataset[i]
        pos  = data.pos.cpu()

        if graph_type == "radius":
            edge_index = radius_graph(
                pos, r=radius, batch=None, loop=False,
                max_num_neighbors=max_num_neighbors,
            )
        elif graph_type == "knn":
            edge_index = knn_graph(pos, k=knn_k, batch=None, loop=False)
        else:
            raise ValueError("graph_type must be 'radius' or 'knn'")

        if edge_index.numel() == 0:
            edge_dist = pos.new_empty((0,), dtype=torch.float16)
        else:
            s, d = edge_index
            edge_dist = (pos[s] - pos[d]).norm(dim=-1).to(torch.float16)

        kj_idx, ji_idx = build_triplet_index(edge_index)

        ei_list.append(edge_index.cpu())
        ed_list.append(edge_dist.cpu())
        kj_list.append(kj_idx.cpu())
        ji_list.append(ji_idx.cpu())

        if (i + 1) % 20000 == 0:
            print(f"[Cache]   {i+1}/{len(dataset)}")

    dataset.transform = old_tf
    torch.save(ei_list, ei_path)
    torch.save(ed_list, ed_path)
    torch.save(kj_list, kj_path)
    torch.save(ji_list, ji_path)
    _write_meta(meta_path, {
        "format":            "edge_index+edge_dist+triplets_v1",
        "graph_type":        str(graph_type),
        "radius":            str(float(radius)),
        "knn_k":             str(int(knn_k)),
        "max_num_neighbors": str(int(max_num_neighbors)),
        "num_graphs":        str(int(len(dataset))),
    })
    print(f"[Cache] Saved to: {cache_dir}")
    return ei_list, ed_list, kj_list, ji_list


# =============================================================================
# CachedQM9 dataset wrapper
# =============================================================================

class CachedQM9(torch.utils.data.Dataset):
    def __init__(
        self,
        base:             QM9,
        edge_index_list:  List[torch.Tensor],
        edge_dist_list:   List[torch.Tensor],
        triplet_kj_list:  List[torch.Tensor],
        triplet_ji_list:  List[torch.Tensor],
    ):
        self.base            = base
        self.edge_index_list = edge_index_list
        self.edge_dist_list  = edge_dist_list
        self.triplet_kj_list = triplet_kj_list
        self.triplet_ji_list = triplet_ji_list
        n = len(base)
        if not all(len(x) == n for x in [edge_index_list, edge_dist_list, triplet_kj_list, triplet_ji_list]):
            raise RuntimeError("CachedQM9: cache length mismatch.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        data = self.base[i]
        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_index_bond = data.edge_index
        else:
            data.edge_index_bond = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr_bond = data.edge_attr
        else:
            data.edge_attr_bond = None
        data.edge_attr  = None
        data.edge_index = self.edge_index_list[i]
        data.edge_dist  = self.edge_dist_list[i]
        data.triplet_kj = self.triplet_kj_list[i]
        data.triplet_ji = self.triplet_ji_list[i]
        return data


# =============================================================================
# Scatter helper
# =============================================================================

def _index_add(
    dst_index: torch.Tensor,
    src_values: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    out = torch.zeros(
        (dim_size,) + src_values.shape[1:],
        device=src_values.device,
        dtype=src_values.dtype,
    )
    out.index_add_(0, dst_index, src_values)
    return out


# =============================================================================
# EquiBlock3Body — 3-body angle-aware equivariant message passing
# =============================================================================

class EquiBlock3Body(nn.Module):
    """
    Scalar-vector equivariant block augmented with 3-body angular context.
    Scalar channel: s [N, H]
    Vector channel: v [N, H, 3]
    """
    def __init__(
        self,
        hidden:         int,
        edge_dim:       int,
        dropout:        float,
        angle_rbf_bins: int = 16,
    ):
        super().__init__()
        self.hidden         = int(hidden)
        self.edge_dim       = int(edge_dim)
        self.angle_rbf_bins = int(angle_rbf_bins)
        self.dropout        = float(dropout)

        self.triplet_mlp = nn.Sequential(
            nn.Linear(edge_dim + edge_dim + angle_rbf_bins, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.upd_s = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.gate_v = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.ln_s = nn.LayerNorm(hidden)

    def forward(
        self,
        s:            torch.Tensor,
        v:            torch.Tensor,
        edge_index:   torch.Tensor,
        edge_rbf:     torch.Tensor,
        edge_dir:     torch.Tensor,
        triplet_kj:   torch.Tensor,
        triplet_ji:   torch.Tensor,
        angle_feat:   torch.Tensor,
        *,
        edge_dropout_p: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, H = s.shape
        E    = edge_index.size(1)

        # Triplet aggregation
        if triplet_kj.numel() > 0:
            t_in = torch.cat([
                edge_rbf[triplet_kj],
                edge_rbf[triplet_ji],
                angle_feat,
            ], dim=-1)
            t_msg = self.triplet_mlp(t_in).to(dtype=s.dtype)
            angle_agg = torch.zeros((E, H), device=s.device, dtype=s.dtype)
            angle_agg.index_add_(0, triplet_ji, t_msg)
        else:
            angle_agg = torch.zeros((E, H), device=s.device, dtype=s.dtype)

        # Edge dropout
        if self.training and edge_dropout_p > 0.0 and E > 0:
            keep      = torch.rand(E, device=edge_index.device) >= edge_dropout_p
            edge_index = edge_index[:, keep]
            edge_rbf   = edge_rbf[keep]
            edge_dir   = edge_dir[keep]
            angle_agg  = angle_agg[keep]

        src, dst = edge_index
        msg_in = torch.cat([s[src], s[dst], edge_rbf, angle_agg], dim=-1)
        msg    = self.msg_mlp(msg_in)
        m_s, m_v_coeff = msg.chunk(2, dim=-1)

        agg_s = _index_add(dst, m_s, N)
        m_v   = m_v_coeff.unsqueeze(-1) * edge_dir.unsqueeze(1)
        agg_v = _index_add(dst, m_v.reshape(m_v.size(0), H * 3), N).view(N, H, 3)

        v_norm = torch.linalg.norm(v, dim=-1)
        ctx    = torch.cat([s, agg_s, v_norm], dim=-1)

        s = s + self.upd_s(ctx)
        v = v + self.gate_v(ctx).unsqueeze(-1) * agg_v
        s = self.ln_s(s)
        s = F.silu(s)
        s = F.dropout(s, p=self.dropout, training=self.training)
        return s, v


# =============================================================================
# Bond-GINE stack
# =============================================================================

class BondGINEStack(nn.Module):
    def __init__(
        self,
        hidden:     int,
        edge_dim:   int,
        num_layers: int,
        dropout:    float,
        train_eps:  bool = False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.dropout    = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=train_eps, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(
        self,
        s:              torch.Tensor,
        edge_index_bond: torch.Tensor,
        edge_attr_bond:  torch.Tensor,
    ) -> torch.Tensor:
        for conv, ln in zip(self.convs, self.norms):
            s = s + conv(s, edge_index_bond, edge_attr_bond)
            s = ln(s)
            s = F.silu(s)
            s = F.dropout(s, p=self.dropout, training=self.training)
        return s


# =============================================================================
# V8 Classical Model (modified to expose graph embedding)
# =============================================================================

class RealGINEop2_v8_Model(nn.Module):
    """
    V8 equivariant + 3-body + bond-GINE backbone.
    forward(..., return_graph_embed=True) returns (pred, g) where g is the
    pooled graph embedding before the prediction head — used by QGINE_v8.
    """
    def __init__(
        self,
        in_dim:           int,
        hidden:           int   = 256,
        layers:           int   = 4,
        dropout:          float = 0.1,
        rbf_bins:         int   = 32,
        rbf_max:          float = 5.0,
        readout:          str   = "add+mean",
        use_virtual_node: bool  = True,
        edge_dropout:     float = 0.0,
        noisy_node_std:   float = 0.0,
        head_type:        str   = "mlp",
        bond_edge_dim:    int   = 4,
        bond_gine_layers: int   = 1,
        bond_gine_mode:   str   = "pre",
        bond_use_dist:    bool  = False,
        bond_rbf_bins:    int   = 8,
        bond_rbf_max:     float = 2.0,
        bond_train_eps:   bool  = False,
        use_3body:        bool  = True,
        angle_rbf_bins:   int   = 16,
    ):
        super().__init__()
        assert readout      in ("add", "mean", "add+mean"), "readout must be add|mean|add+mean"
        assert head_type    in ("mlp", "dipole"),           "head_type must be mlp|dipole"
        assert bond_gine_mode in ("pre", "interleave"),     "bond_gine_mode must be pre|interleave"

        self.hidden           = int(hidden)
        self.layers           = int(layers)
        self.dropout          = float(dropout)
        self.readout          = readout
        self.edge_dropout     = float(edge_dropout)
        self.noisy_node_std   = float(noisy_node_std)
        self.head_type        = head_type
        self.use_virtual_node = bool(use_virtual_node)
        self.use_3body        = bool(use_3body)
        self.angle_rbf_bins   = int(angle_rbf_bins)

        if self.use_3body:
            self.angle_rbf = RBF(r_min=0.0, r_max=math.pi, n_bins=self.angle_rbf_bins)

        self.node_in  = nn.Linear(in_dim, hidden)
        self.rbf_bins = int(rbf_bins)
        self.rbf      = RBF(r_min=0.0, r_max=float(rbf_max), n_bins=self.rbf_bins)

        self.blocks = nn.ModuleList([
            EquiBlock3Body(
                hidden=hidden, edge_dim=self.rbf_bins,
                dropout=dropout, angle_rbf_bins=self.angle_rbf_bins,
            )
            for _ in range(layers)
        ])

        if self.use_virtual_node:
            self.vn_init    = nn.Parameter(torch.zeros(hidden))
            self.vn_to_node = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.node_to_vn = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.vn_dropout = nn.Dropout(self.dropout)

        self.bond_gine_layers = int(bond_gine_layers)
        self.bond_gine_mode   = bond_gine_mode
        self.bond_use_dist    = bool(bond_use_dist)
        bond_edge_dim         = int(bond_edge_dim)
        self.bond_rbf_bins    = int(bond_rbf_bins)

        if self.bond_use_dist:
            self.bond_rbf = RBF(r_min=0.0, r_max=float(bond_rbf_max), n_bins=self.bond_rbf_bins)
            bond_edge_dim = bond_edge_dim + self.bond_rbf_bins
        else:
            self.bond_rbf = None

        if self.bond_gine_layers > 0:
            n_gine = (
                self.bond_gine_layers
                if bond_gine_mode == "pre"
                else min(self.bond_gine_layers, layers)
            )
            self.bond_gine = BondGINEStack(
                hidden=hidden, edge_dim=bond_edge_dim,
                num_layers=n_gine, dropout=dropout, train_eps=bond_train_eps,
            )
        else:
            self.bond_gine = None

        # Compute graph embedding dimension for downstream use
        self.node_read_dim = 2 * hidden
        pool_dim = self.node_read_dim if readout in ("add", "mean") else 2 * self.node_read_dim
        self.g_dim = pool_dim + (hidden if self.use_virtual_node else 0)

        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.g_dim, hidden), nn.SiLU(),
                nn.Dropout(self.dropout), nn.Linear(hidden, 1),
            )
        else:
            self.charge_head = nn.Sequential(
                nn.Linear(self.node_read_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1),
            )
            self.dipole_head = nn.Sequential(
                nn.Linear(1, hidden), nn.SiLU(),
                nn.Dropout(self.dropout), nn.Linear(hidden, 1),
            )

    def _pool(self, nf: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "add":
            return global_add_pool(nf, batch)
        if self.readout == "mean":
            return global_mean_pool(nf, batch)
        return torch.cat([global_add_pool(nf, batch), global_mean_pool(nf, batch)], dim=-1)

    def forward(
        self,
        data,
        *,
        return_graph_embed: bool = False,
    ):
        x, batch = data.x, data.batch
        N = x.size(0)

        s = self.node_in(x)
        if self.training and self.noisy_node_std > 0.0:
            s = s + torch.randn_like(s) * self.noisy_node_std

        v          = torch.zeros((N, self.hidden, 3), device=s.device, dtype=s.dtype)
        edge_index = data.edge_index
        bond_ei    = getattr(data, "edge_index_bond", None)
        bond_ea    = getattr(data, "edge_attr_bond",  None)

        # Geometry + angle computation — always FP32
        _cuda = s.is_cuda
        _ctx  = torch.autocast(device_type="cuda", enabled=False) if _cuda else contextlib.nullcontext()
        with _ctx:
            pos32       = data.pos.to(dtype=torch.float32)
            dist32      = data.edge_dist.to(dtype=torch.float32)
            src, dst    = edge_index
            vec         = pos32[dst] - pos32[src]
            edge_dir    = vec / (dist32.unsqueeze(-1) + 1e-9)
            edge_rbf    = self.rbf(dist32)

            # Angle features
            if self.use_3body and hasattr(data, "triplet_kj"):
                tkj = data.triplet_kj.to(device=s.device)
                tji = data.triplet_ji.to(device=s.device)
                if tkj.numel() > 0:
                    cos_a     = (-edge_dir[tkj] * edge_dir[tji]).sum(-1).clamp(-1., 1.)
                    angle_f   = self.angle_rbf(torch.acos(cos_a))
                else:
                    angle_f = torch.zeros((0, self.angle_rbf_bins), device=s.device, dtype=torch.float32)
            else:
                tkj     = torch.zeros((0,), dtype=torch.long, device=s.device)
                tji     = torch.zeros((0,), dtype=torch.long, device=s.device)
                angle_f = torch.zeros((0, self.angle_rbf_bins), device=s.device, dtype=torch.float32)

            # Bond distance RBF (optional)
            if self.bond_gine_layers > 0 and self.bond_use_dist and bond_ei is not None:
                bs, bd    = bond_ei
                bdist     = torch.linalg.norm(pos32[bd] - pos32[bs], dim=-1)
                bond_rbf_out = self.bond_rbf(bdist)
            else:
                bond_rbf_out = None

        if self.bond_gine_layers > 0 and bond_ea is not None:
            bond_ea = bond_ea.to(device=s.device, dtype=s.dtype)
            if self.bond_use_dist and bond_rbf_out is not None:
                bond_ea = torch.cat([bond_ea, bond_rbf_out.to(dtype=s.dtype)], dim=-1)

        if self.use_virtual_node:
            ng = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            vn = self.vn_init.unsqueeze(0).expand(ng, -1)

        # Bond-GINE pre
        if self.bond_gine_layers > 0 and self.bond_gine_mode == "pre" and bond_ei is not None:
            s = self.bond_gine(s, bond_ei, bond_ea)

        for li, blk in enumerate(self.blocks):
            # Bond-GINE interleave
            if self.bond_gine_layers > 0 and self.bond_gine_mode == "interleave" and bond_ei is not None:
                if li < self.bond_gine.num_layers:
                    s = self.bond_gine.convs[li](s, bond_ei, bond_ea) + s
                    s = self.bond_gine.norms[li](s)
                    s = F.silu(s)
                    s = F.dropout(s, p=self.dropout, training=self.training)

            if self.use_virtual_node:
                s = s + self.vn_dropout(self.vn_to_node(vn[batch]))

            s, v = blk(
                s, v,
                edge_index=edge_index,
                edge_rbf=edge_rbf.to(dtype=s.dtype),
                edge_dir=edge_dir.to(dtype=s.dtype),
                triplet_kj=tkj,
                triplet_ji=tji,
                angle_feat=angle_f.to(dtype=s.dtype),
                edge_dropout_p=self.edge_dropout,
            )

            if self.use_virtual_node:
                vn = vn + self.node_to_vn(global_add_pool(s, batch))

        v_norm    = torch.linalg.norm(v, dim=-1)
        node_feat = torch.cat([s, v_norm], dim=-1)
        g         = self._pool(node_feat, batch)
        if self.use_virtual_node:
            g = torch.cat([g, vn], dim=-1)

        # Prediction head
        if self.head_type == "mlp":
            pred = self.head(g).view(-1)
        else:
            pos32 = data.pos.to(device=node_feat.device, dtype=torch.float32)
            q     = self.charge_head(node_feat).squeeze(-1)
            ng    = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            ones  = torch.ones((N,), device=q.device, dtype=torch.float32)
            q_sum  = _index_add(batch, q.unsqueeze(-1), ng).squeeze(-1)
            n_sum  = _index_add(batch, ones.unsqueeze(-1), ng).squeeze(-1).clamp_min(1.0)
            q_c    = q - (q_sum / n_sum)[batch]
            p_sum  = _index_add(batch, pos32, ng)
            p_c    = pos32 - (p_sum / n_sum.unsqueeze(-1))[batch]
            dip    = _index_add(batch, q_c.unsqueeze(-1) * p_c, ng)
            pred   = self.dipole_head(torch.linalg.norm(dip, dim=-1, keepdim=True)).view(-1)

        if return_graph_embed:
            return pred, g
        return pred


# =============================================================================
# Quantum head — QuantumReuploadingHead (from QGINE_v2, unchanged)
# =============================================================================

class QuantumReuploadingHead(nn.Module):
    """
    Trainable data re-uploading VQC.
    Input:  theta [B, p]   — rotation angles, typically in [-pi, pi]
    Output: scalar [B]     — learned linear combination of expectation values
    """
    def __init__(
        self,
        p:              int   = 4,
        Lq:             int   = 2,
        measure:        str   = "z+zz",
        device_name:    str   = "default.qubit",
        shots:          Optional[int] = None,
        diff_method:    str   = "backprop",
        broadcast_mode: str   = "native",
        init_std:       float = 0.02,
        readout_hidden: int   = 32,
    ):
        super().__init__()
        assert measure        in ("z", "z+zz"),              "measure must be 'z' or 'z+zz'"
        assert broadcast_mode in ("native", "expand", "off"), "broadcast_mode must be native|expand|off"
        if shots is not None and diff_method == "backprop":
            raise ValueError("shots != None incompatible with diff_method='backprop'")

        self.p              = int(p)
        self.Lq             = int(Lq)
        self.measure        = measure
        self.broadcast_mode = broadcast_mode

        # Variational parameters
        self.rot_weights = nn.Parameter(torch.zeros(self.Lq, self.p, 3))
        self.enc_scale   = nn.Parameter(torch.ones(self.Lq, self.p, 2))
        self.enc_bias    = nn.Parameter(torch.zeros(self.Lq, self.p, 2))
        with torch.no_grad():
            self.rot_weights.normal_(0.0, float(init_std))
            self.enc_scale.clamp_(0.5, 1.5)

        n_meas = self.p + (self.p if measure == "z+zz" else 0)
        self.n_meas = int(n_meas)

        self.readout = nn.Sequential(
            nn.Linear(self.n_meas, int(readout_hidden)),
            nn.SiLU(),
            nn.Linear(int(readout_hidden), 1),
        )

        dev = qml.device(device_name, wires=self.p, shots=shots)

        def _circuit(theta, rot_weights, enc_scale, enc_bias):
            theta2 = theta.unsqueeze(0) if theta.dim() == 1 else theta
            for k in range(self.p):
                qml.Hadamard(wires=k)
            for l in range(self.Lq):
                for k in range(self.p):
                    t0 = enc_scale[l, k, 0] * theta2[:, k] + enc_bias[l, k, 0]
                    t1 = enc_scale[l, k, 1] * theta2[:, k] + enc_bias[l, k, 1]
                    qml.RZ(t0, wires=k)
                    qml.RX(t1, wires=k)
                    w = rot_weights[l, k]
                    qml.Rot(w[0], w[1], w[2], wires=k)
                for k in range(self.p):
                    qml.CNOT(wires=[k, (k + 1) % self.p])
            obs = [qml.expval(qml.PauliZ(k)) for k in range(self.p)]
            if self.measure == "z+zz":
                obs += [
                    qml.expval(qml.PauliZ(k) @ qml.PauliZ((k + 1) % self.p))
                    for k in range(self.p)
                ]
            return tuple(obs)

        qnode = qml.QNode(_circuit, dev, interface="torch", diff_method=diff_method)
        if broadcast_mode == "expand":
            self.qnode = qml.transforms.broadcast_expand(qnode)
        else:
            self.qnode = qnode

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # Quantum head may be on CPU; move theta accordingly
        theta = theta.to(device=self.rot_weights.device, dtype=torch.float32)

        if self.broadcast_mode == "off":
            outs = []
            for b in range(theta.shape[0]):
                out_b = self.qnode(theta[b], self.rot_weights, self.enc_scale, self.enc_bias)
                outs.append(torch.stack(list(out_b)))
            feats = torch.stack(outs)
        else:
            out = self.qnode(theta, self.rot_weights, self.enc_scale, self.enc_bias)
            if isinstance(out, (tuple, list)):
                feats = torch.stack(list(out), dim=-1)
            else:
                feats = out
                if feats.dim() == 1:
                    feats = feats.unsqueeze(-1)
                # Handle transposed batch dimension from some PennyLane versions
                if feats.shape[0] != theta.shape[0] and feats.shape[-1] == theta.shape[0]:
                    feats = feats.transpose(0, 1)

        feats = feats.to(
            dtype=self.readout[0].weight.dtype,
            device=self.readout[0].weight.device,
        )
        return self.readout(feats).squeeze(-1)


# =============================================================================
# QGINE_v10 — hybrid model
# =============================================================================

class QGINE_v10(nn.Module):
    """
    Hybrid: RealGINEop2_v8_Model + QuantumReuploadingHead correction.

    final_pred = base_pred + q_scale * q_pred

    q_scale is initialized at 0.0 so the model starts as pure classical and
    the quantum component is activated gradually during training.

    Quantum head is deliberately kept on CPU (qml simulators don't use CUDA).
    Device unification happens in forward().
    """
    def __init__(
        self,
        # --- classical backbone args (mirror RealGINEop2_v8_Model) ---
        in_dim:           int,
        hidden:           int   = 256,
        layers:           int   = 4,
        dropout:          float = 0.1,
        rbf_bins:         int   = 32,
        rbf_max:          float = 5.0,
        readout:          str   = "add+mean",
        use_virtual_node: bool  = True,
        edge_dropout:     float = 0.0,
        noisy_node_std:   float = 0.0,
        head_type:        str   = "mlp",
        bond_edge_dim:    int   = 4,
        bond_gine_layers: int   = 1,
        bond_gine_mode:   str   = "pre",
        bond_use_dist:    bool  = False,
        bond_rbf_bins:    int   = 8,
        bond_rbf_max:     float = 2.0,
        bond_train_eps:   bool  = False,
        use_3body:        bool  = True,
        angle_rbf_bins:   int   = 16,
        # --- quantum head args ---
        q_p:              int   = 4,
        q_Lq:             int   = 2,
        q_measure:        str   = "z+zz",
        q_device_name:    str   = "default.qubit",
        q_shots:          Optional[int] = None,
        q_diff_method:    str   = "backprop",
        q_broadcast_mode: str   = "native",
        q_init_std:       float = 0.02,
        q_readout_hidden: int   = 32,
        q_scale_init:     float = 0.0,
    ):
        super().__init__()

        self.base = RealGINEop2_v8_Model(
            in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout,
            rbf_bins=rbf_bins, rbf_max=rbf_max, readout=readout,
            use_virtual_node=use_virtual_node,
            edge_dropout=edge_dropout, noisy_node_std=noisy_node_std,
            head_type=head_type,
            bond_edge_dim=bond_edge_dim, bond_gine_layers=bond_gine_layers,
            bond_gine_mode=bond_gine_mode, bond_use_dist=bond_use_dist,
            bond_rbf_bins=bond_rbf_bins, bond_rbf_max=bond_rbf_max,
            bond_train_eps=bond_train_eps,
            use_3body=use_3body, angle_rbf_bins=angle_rbf_bins,
        )

        # Project graph embedding (g_dim -> p) then scale to angle range
        self.angle_head = nn.Linear(self.base.g_dim, q_p, bias=True)

        self.qhead = QuantumReuploadingHead(
            p=q_p, Lq=q_Lq, measure=q_measure,
            device_name=q_device_name, shots=q_shots,
            diff_method=q_diff_method,
            broadcast_mode=q_broadcast_mode,
            init_std=q_init_std,
            readout_hidden=q_readout_hidden,
        )

        # q_scale init=0.0 → pure classical at epoch 0
        self.q_scale = nn.Parameter(torch.tensor(float(q_scale_init)))
        self._quantum_active: bool = False

    def forward(
        self,
        data,
        *,
        return_parts: bool = False,
    ):
        base_pred, g = self.base(data, return_graph_embed=True)

        # Project graph embedding to angle input for the quantum circuit
        theta = math.pi * torch.tanh(self.angle_head(g))  # [B, p] in (-pi, pi)

        if self._quantum_active:
            theta_q = theta.to(device=self.qhead.rot_weights.device, dtype=torch.float32)
            q_pred  = self.qhead(theta_q)
            if q_pred.device != base_pred.device:
                q_pred = q_pred.to(base_pred.device)
            pred = base_pred + self.q_scale * q_pred
        else:
            q_pred = torch.zeros_like(base_pred)
            pred = base_pred

        if return_parts:
            return pred, base_pred, q_pred, theta, g
        return pred


# =============================================================================
# Checkpoint loading: pretrained V8 classical → QGINE_v10
# =============================================================================

def load_v8_checkpoint_into_qgine(
    model: QGINE_v10,
    ckpt_path: str,
) -> Dict:
    """
    Load a checkpoint saved by the standalone V8 training script into the
    QGINE_v10 hybrid model. Only classical base weights are transferred;
    quantum components (angle_head, qhead, q_scale) are left at init values.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError("checkpoint missing 'model_state_dict'.")

    src = ckpt["model_state_dict"]
    # V8 checkpoint keys are flat (e.g. "node_in.weight").
    # In QGINE_v10 they live under "base.*".
    mapped = {"base." + k: v for k, v in src.items()}
    res = model.load_state_dict(mapped, strict=False)

    return {
        "loaded":          len(mapped),
        "missing_keys":    list(res.missing_keys),
        "unexpected_keys": list(res.unexpected_keys),
        "ckpt_meta": {
            k: ckpt.get(k)
            for k in ["hidden", "layers", "dropout", "readout",
                       "rbf_bins", "rbf_max", "best_val_mae", "best_epoch"]
        },
    }


# =============================================================================
# AMP helpers
# =============================================================================

def _autocast_ctx(device: torch.device, use_amp: bool, amp_dtype: str):
    if not (use_amp and device.type == "cuda"):
        return contextlib.nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _make_grad_scaler(use_amp: bool, amp_dtype: str, device: torch.device):
    enable = bool(use_amp and device.type == "cuda" and amp_dtype == "fp16")
    try:
        return torch.amp.GradScaler("cuda", enabled=enable)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enable)


# =============================================================================
# EMA
# =============================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


# =============================================================================
# Per-epoch diagnostic logging of quantum state
# =============================================================================

@torch.no_grad()
def log_quantum_state(
    model: "QGINE_v10",
    loader,
    device: torch.device,
    writer,
    epoch: int,
    stage: str,
):
    """
    Log q_scale, mean |q_pred|, mean |base_pred|, and theta stats.
    Runs on a single batch — cheap diagnostic, not a full pass.
    """
    model.eval()
    try:
        batch = next(iter(loader))
        batch = batch.to(device)
        pred, base_pred, q_pred, theta, _ = model(batch, return_parts=True)
        q_scale_val    = float(model.q_scale.item())
        q_pred_mean    = float(q_pred.abs().mean().item())
        base_pred_mean = float(base_pred.abs().mean().item())
        contrib        = float((model.q_scale * q_pred).abs().mean().item())
        contrib_pct    = 100.0 * contrib / (base_pred_mean + 1e-12)

        # theta stats — free, already computed
        theta_mean = float(theta.mean().item())
        theta_std  = float(theta.std().item())
        theta_min  = float(theta.min().item())
        theta_max  = float(theta.max().item())

        writer.writerow([
            stage, epoch, "q_diagnostic",
            q_scale_val, q_pred_mean, base_pred_mean, contrib,
            contrib_pct, theta_mean, theta_std, theta_min, theta_max,
        ])

        # Warn if quantum contribution is suspiciously low after warmup
        if stage == "quantum" and epoch > 10 and contrib_pct < 0.1:
            log(f"  [WARN] q contribution only {contrib_pct:.3f}% — quantum head may be stuck")

    except Exception as e:
        if epoch == 1:
            log(f"  [qdiag] skipped: {type(e).__name__}: {e}")


# =============================================================================
# Train / eval loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    opt,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    scaler,
    *,
    loss_type: str,
    huber_delta: float,
    ema: Optional[EMA],
) -> Tuple[float, float]:
    """Returns (avg_train_loss, grad_norm). grad_norm is free — clip_grad_norm_ computes it."""
    model.train()
    total, n_graphs = 0.0, 0
    use_fp16 = bool(use_amp and device.type == "cuda" and amp_dtype == "fp16")
    last_grad_norm = 0.0

    for data in loader:
        data = data.to(device)
        opt.zero_grad(set_to_none=True)

        with _autocast_ctx(device, use_amp, amp_dtype):
            pred_norm = model(data)
            y_norm    = data.y[:, 0].view(-1)
            loss = (
                F.smooth_l1_loss(pred_norm, y_norm, beta=float(huber_delta))
                if loss_type == "huber"
                else F.l1_loss(pred_norm, y_norm)
            )

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            opt.step()

        if ema is not None:
            ema.update(model)

        total    += float(loss.item()) * data.num_graphs
        n_graphs += data.num_graphs

    return total / max(n_graphs, 1), last_grad_norm


@torch.no_grad()
def eval_mae(
    model: nn.Module,
    loader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    use_amp: bool,
    amp_dtype: str,
    ema: Optional[EMA] = None,
) -> float:
    model.eval()
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)
    if ema is not None:
        ema.apply_to(model)

    total, n = 0.0, 0
    for data in loader:
        data = data.to(device)
        with _autocast_ctx(device, use_amp, amp_dtype):
            pred_norm = model(data)
        y_norm = data.y[:, 0].view(-1)
        total += ((pred_norm - y_norm).abs() * y_std).sum().item()
        n     += y_norm.numel()

    if ema is not None:
        ema.restore(model)
    return total / max(n, 1)


@torch.no_grad()
def mean_baseline_mae(loader, device, y_mean, y_std) -> float:
    y_mean_f = float(y_mean.to(device).item())
    y_std    = y_std.to(device)
    total, n = 0.0, 0
    for data in loader:
        data = data.to(device)
        y    = data.y[:, 0].view(-1) * y_std + y_mean_f
        total += (torch.full_like(y, y_mean_f) - y).abs().sum().item()
        n    += y.numel()
    return total / max(n, 1)


# =============================================================================
# Stage training — supports freeze/unfreeze pattern from QGINE_v2
# =============================================================================

def _make_scheduler(opt, scheduler_type, warmup_epochs, epochs):
    if scheduler_type == "cosine":
        if warmup_epochs > 0:
            warmup  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
            cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs))
            return torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_epochs])
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)


def run_stage(
    *,
    stage_name:    str,
    model:         "QGINE_v10",
    train_loader,
    val_loader,
    device:        torch.device,
    y_mean:        torch.Tensor,
    y_std:         torch.Tensor,
    epochs:        int,
    lr:            float,
    weight_decay:  float,
    patience:      int,
    scheduler_type: str,
    warmup_epochs: int,
    use_amp:       bool,
    amp_dtype:     str,
    loss_type:     str,
    huber_delta:   float,
    use_ema:       bool,
    ema_decay:     float,
    ckpt_path:     str,
    log_path:      str,
    diag_path:     str,
) -> float:
    """
    Train for one stage. Returns best validation MAE achieved this stage.
    """
    if epochs <= 0:
        log(f"[{stage_name}] skipped (epochs=0)")
        return float("inf")

    if stage_name in ("quantum", "finetune"):
        assert any(p.requires_grad for p in model.qhead.parameters()), "qhead params not trainable"
        assert any(p.requires_grad for p in model.angle_head.parameters()), "angle_head params not trainable"
        assert model.q_scale.requires_grad, "q_scale not trainable"

    params   = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    opt      = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler   = _make_grad_scaler(use_amp, amp_dtype, device)
    sched    = _make_scheduler(opt, scheduler_type, warmup_epochs, epochs)
    ema      = EMA(model, decay=ema_decay) if use_ema else None

    best_val, best_epoch, bad = float("inf"), -1, 0
    prev_train_loss = None
    stage_start = time.time()
    epoch_times: List[float] = []

    log_separator()
    log(f"  STAGE: {stage_name.upper()}")
    log(f"  Trainable params : {n_params:,}")
    log(f"  Epochs / Patience: {epochs} / {patience}")
    log(f"  LR / WD          : {lr:.2e} / {weight_decay:.2e}")
    log(f"  Scheduler        : {scheduler_type} (warmup={warmup_epochs})")
    log(f"  Loss             : {loss_type}" + (f" (delta={huber_delta})" if loss_type=="huber" else ""))
    log(f"  EMA              : {use_ema}" + (f" (decay={ema_decay})" if use_ema else ""))
    log(f"  {_gpu_mem_str()}")
    log_separator()

    with open(log_path, "a", newline="") as lf, open(diag_path, "a", newline="") as df:
        lwriter = csv.writer(lf)
        dwriter = csv.writer(df)

        for epoch in range(1, epochs + 1):
            ep_start = time.time()

            train_loss, grad_norm = train_one_epoch(
                model, train_loader, opt, device, use_amp, amp_dtype, scaler,
                loss_type=loss_type, huber_delta=huber_delta, ema=ema,
            )
            val_mae = eval_mae(model, val_loader, device, y_mean, y_std, use_amp, amp_dtype, ema=ema)

            if scheduler_type == "cosine":
                sched.step()
            else:
                sched.step(val_mae)

            ep_elapsed = time.time() - ep_start
            epoch_times.append(ep_elapsed)

            # Quantum diagnostics (every 10 epochs or first)
            if epoch % 10 == 0 or epoch == 1:
                log_quantum_state(model, train_loader, device, dwriter, epoch, stage_name)
            df.flush()

            improved = val_mae < best_val - 1e-6
            delta_str = ""
            if improved:
                if best_val < float("inf"):
                    delta_str = f" (↓{best_val - val_mae:.6f})"
                best_val, best_epoch, bad = val_mae, epoch, 0
                torch.save({
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "best_val_mae":     float(best_val),
                    "best_epoch":       int(best_epoch),
                    "stage":            stage_name,
                    "q_scale":          float(model.q_scale.item()),
                    "y_mean":           y_mean.detach().cpu(),
                    "y_std":            y_std.detach().cpu(),
                }, ckpt_path)
            else:
                bad += 1

            lr_now = opt.param_groups[0]["lr"]
            q_s    = float(model.q_scale.item())

            lwriter.writerow([stage_name, epoch, lr_now, train_loss, val_mae, best_val, q_s])
            lf.flush()

            # ETA calculation
            avg_ep_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])
            epochs_left = epochs - epoch
            eta_str = _fmt_time(avg_ep_time * epochs_left)

            # Loss spike detection
            spike_str = ""
            if prev_train_loss is not None and train_loss > prev_train_loss * 3.0 and epoch > 5:
                spike_str = f"  ⚠ LOSS SPIKE ({prev_train_loss:.4f}→{train_loss:.4f})"
            prev_train_loss = train_loss

            # Print every epoch for first 10, then every 5
            should_print = epoch <= 10 or epoch % 5 == 0 or improved or spike_str
            if should_print:
                patience_bar = f"patience={bad}/{patience}"
                log(
                    f"  [{stage_name}] ep={epoch:04d}/{epochs}"
                    f" | lr={lr_now:.2e}"
                    f" | train={train_loss:.5f}"
                    f" | val={val_mae:.6f}{delta_str}"
                    f" | best={best_val:.6f}@{best_epoch}"
                    f" | q_scale={q_s:.5f}"
                    f" | gnorm={grad_norm:.3f}"
                    f" | {patience_bar}"
                    f" | ep={ep_elapsed:.1f}s ETA={eta_str}"
                    + spike_str
                )

            if bad >= patience:
                log(f"  [{stage_name}] Early stop @ ep={epoch}")
                log(f"  [{stage_name}] Best val={best_val:.6f} @ epoch {best_epoch}")
                break

        stage_elapsed = time.time() - stage_start
        log_separator("-")
        log(f"  [{stage_name}] COMPLETE in {_fmt_time(stage_elapsed)}")
        log(f"  [{stage_name}] Best val MAE : {best_val:.6f} @ epoch {best_epoch}")
        log(f"  [{stage_name}] Final q_scale: {float(model.q_scale.item()):.6f}")
        log(f"  [{stage_name}] Avg epoch time: {sum(epoch_times)/len(epoch_times):.1f}s")
        log(f"  {_gpu_mem_str()}")
        log_separator("-")

    return best_val


# =============================================================================
# Main run
# =============================================================================

def run_one_config(
    *,
    train_ds, val_ds, test_ds,
    device, target_idx, out_dir,
    # classical
    hidden, layers, dropout, readout,
    y_mean, y_std, rbf_bins, rbf_max,
    split_seed, train_seed,
    batch_size_train, batch_size_eval, batch_size_quantum,
    num_workers_base, num_workers_quantum, num_workers_finetune,
    use_amp, amp_dtype, use_compile,
    scheduler_type, warmup_epochs,
    loss_type, huber_delta,
    use_virtual_node, head_type,
    edge_dropout, noisy_node_std,
    use_ema, ema_decay,
    bond_gine_layers, bond_gine_mode,
    bond_use_dist, bond_rbf_bins, bond_rbf_max, bond_train_eps,
    use_3body, angle_rbf_bins,
    # quantum
    q_p, q_Lq, q_measure, q_device_name, q_shots,
    q_diff_method, q_broadcast_mode, q_init_std,
    q_readout_hidden, q_scale_init,
    # stage control
    epochs_base, patience_base, lr_base,
    epochs_quantum, patience_quantum, lr_quantum,
    freeze_base_during_quantum,
    epochs_finetune, patience_finetune, lr_finetune,
    unfreeze_last_block_finetune,
    weight_decay,
    # optional pretrained base checkpoint
    init_v8_ckpt: Optional[str] = None,
    # skip base stage entirely by loading a previous checkpoint
    load_base_ckpt: Optional[str] = None,
):
    seed_all(train_seed)

    tag = (
        f"t{target_idx}_qv8_h{hidden}_L{layers}"
        f"_p{q_p}_Lq{q_Lq}_{q_measure}"
        f"_split{split_seed}_train{train_seed}"
    )
    ckpt_path = os.path.join(out_dir, f"best_{tag}.pt")
    log_path  = os.path.join(out_dir, f"log_{tag}.csv")
    diag_path = os.path.join(out_dir, f"qdiag_{tag}.csv")

    pin    = (device.type == "cuda")
    dl_gen = torch.Generator().manual_seed(train_seed)
    wfn    = make_worker_init_fn(train_seed)

    def _make_loader(ds, bs, shuffle, workers):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=(workers > 0),
            generator=(dl_gen if shuffle else None),
            worker_init_fn=(wfn if workers > 0 else None),
        )

    # Create base-stage loaders now so val_loader exists for the baseline print
    _bsq = batch_size_quantum if batch_size_quantum is not None else batch_size_train
    log(f"  batch_size_quantum={_bsq}")
    train_loader = _make_loader(train_ds, _bsq, True, num_workers_quantum)
    val_loader   = _make_loader(val_ds, batch_size_eval, False, num_workers_quantum)
    test_loader  = _make_loader(test_ds, batch_size_eval, False, num_workers_base)

    baseline_mae = mean_baseline_mae(val_loader, device, y_mean, y_std)
    log(f"  Mean baseline MAE (val): {baseline_mae:.6f}")

    # Determine bond edge dim from sample
    sample = train_ds[0]
    in_dim = sample.x.size(-1)
    bde    = 0
    if bond_gine_layers > 0:
        if not hasattr(sample, "edge_attr_bond") or sample.edge_attr_bond is None:
            raise RuntimeError("bond_gine_layers>0 but sample has no edge_attr_bond.")
        bde = int(sample.edge_attr_bond.size(-1))

    model = QGINE_v10(
        in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout,
        rbf_bins=rbf_bins, rbf_max=rbf_max, readout=readout,
        use_virtual_node=use_virtual_node,
        edge_dropout=edge_dropout, noisy_node_std=noisy_node_std,
        head_type=head_type,
        bond_edge_dim=bde, bond_gine_layers=bond_gine_layers,
        bond_gine_mode=bond_gine_mode, bond_use_dist=bond_use_dist,
        bond_rbf_bins=bond_rbf_bins, bond_rbf_max=bond_rbf_max,
        bond_train_eps=bond_train_eps,
        use_3body=use_3body, angle_rbf_bins=angle_rbf_bins,
        q_p=q_p, q_Lq=q_Lq, q_measure=q_measure,
        q_device_name=q_device_name, q_shots=q_shots,
        q_diff_method=q_diff_method, q_broadcast_mode=q_broadcast_mode,
        q_init_std=q_init_std, q_readout_hidden=q_readout_hidden,
        q_scale_init=q_scale_init,
    ).to(device)

    # Quantum head always on CPU
    if device.type == "cuda":
        model.qhead.to(torch.device("cpu"))
        # DO NOT move angle_head to CPU

    # --- Model summary ---
    pc = _param_count(model)
    base_pc  = _param_count(model.base)
    qhead_pc = _param_count(model.qhead)
    log_separator()
    log(f"  MODEL SUMMARY: {tag}")
    log(f"  Architecture   : hidden={hidden} layers={layers} readout={readout}")
    log(f"  3-body         : use_3body={use_3body} angle_rbf_bins={angle_rbf_bins}")
    log(f"  Bond GINE      : layers={bond_gine_layers} mode={bond_gine_mode} use_dist={bond_use_dist}")
    log(f"  Quantum head   : p={q_p} Lq={q_Lq} measure={q_measure} diff={q_diff_method}")
    log(f"  Total params   : {pc['total']:,}")
    log(f"  Base params    : {base_pc['total']:,}")
    log(f"  Quantum params : {qhead_pc['total']:,}")
    log(f"  Input node dim : {in_dim}  |  Bond edge dim: {bde}")
    log(f"  Device         : {device}  |  AMP: {use_amp} ({amp_dtype})")
    log(f"  {_gpu_mem_str()}")
    log_separator()

    # Optionally warm-start the classical base from a pre-trained V8 checkpoint
    if init_v8_ckpt:
        info = load_v8_checkpoint_into_qgine(model, init_v8_ckpt)
        print(
            f"[Init] loaded {info['loaded']} keys | "
            f"missing={len(info['missing_keys'])} | "
            f"unexpected={len(info['unexpected_keys'])}"
        )
        print(f"[Init] ckpt_meta={info['ckpt_meta']}")

    if use_compile:
        try:
            # dynamic=True is required: PyG's global_add_pool uses
            # new_zeros(batch.max()+1) — a data-dependent size that
            # inductor cannot guard on in static mode.
            model.base = torch.compile(model.base, dynamic=False, fullgraph=False)
            print("[compile] torch.compile applied to base model (dynamic=False, fullgraph=False)")
        except Exception as e:
            print(f"[compile] failed: {e}")

    # Write CSV headers
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["stage", "epoch", "lr", "train_l1_norm", "val_mae", "best_val", "q_scale"])
    with open(diag_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "stage", "epoch", "metric", "q_scale",
            "mean_abs_q_pred", "mean_abs_base_pred", "mean_abs_contribution",
            "contribution_pct", "theta_mean", "theta_std", "theta_min", "theta_max",
        ])

    def _stage_kw():
        """Build stage_kw from the current train_loader / val_loader bindings."""
        return dict(
            model=model, train_loader=train_loader, val_loader=val_loader,
            device=device, y_mean=y_mean, y_std=y_std,
            scheduler_type=scheduler_type, warmup_epochs=warmup_epochs,
            use_amp=use_amp, amp_dtype=amp_dtype,
            loss_type=loss_type, huber_delta=huber_delta,
            use_ema=use_ema, ema_decay=ema_decay,
            weight_decay=weight_decay,
            ckpt_path=ckpt_path, log_path=log_path, diag_path=diag_path,
        )

    model._quantum_active = False
    # ---- Stage 1: Base -------------------------------------------------------
    set_requires_grad(model.qhead, False)
    set_requires_grad(model.angle_head, False)
    model.q_scale.requires_grad = False
    with torch.no_grad():
        model.q_scale.zero_()
    set_requires_grad(model.base, True)

    if load_base_ckpt:
        log(f"  [base] SKIPPING base training — loading checkpoint: {load_base_ckpt}")
        ckpt = torch.load(load_base_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        best_base = float(ckpt.get("best_val_mae", ckpt.get("val_mae", 9999.0)))
        log(f"  [base] Loaded. best_val={best_base:.6f} | q_scale={float(model.q_scale.item()):.4f}")
    else:
        best_base = run_stage(
            stage_name="base",
            epochs=epochs_base, lr=lr_base, patience=patience_base,
            **_stage_kw(),
        )
        log(f"  [base] best_val={best_base:.6f} | q_scale={float(model.q_scale.item()):.4f}")

    model._quantum_active = True
    # ---- Stage 2: Quantum ----------------------------------------------------
    log_section("STAGE TRANSITION: base → quantum")
    log(f"  freeze_base_during_quantum={freeze_base_during_quantum}")
    train_loader = _make_loader(train_ds, batch_size_train, True,  num_workers_quantum)
    val_loader   = _make_loader(val_ds,   batch_size_eval,  False, num_workers_quantum)
    set_requires_grad(model.base, not freeze_base_during_quantum)
    set_requires_grad(model.qhead, True)
    set_requires_grad(model.angle_head, True)
    model.q_scale.requires_grad = True

    best_quantum = run_stage(
        stage_name="quantum",
        epochs=epochs_quantum, lr=lr_quantum, patience=patience_quantum,
        **_stage_kw(),
    )
    q_improvement = best_base - best_quantum
    log(f"  [quantum] best_val={best_quantum:.6f} | improvement over base: {q_improvement:.6f} ({100*q_improvement/best_base:.2f}%)")
    log(f"  [quantum] q_scale={float(model.q_scale.item()):.6f}")

    # ---- Stage 3: Finetune ---------------------------------------------------
    log_section("STAGE TRANSITION: quantum → finetune")
    log(f"  unfreeze_last_block_finetune={unfreeze_last_block_finetune}")
    train_loader = _make_loader(train_ds, batch_size_train, True,  num_workers_finetune)
    val_loader   = _make_loader(val_ds,   batch_size_eval,  False, num_workers_finetune)
    set_requires_grad(model.base, False)
    if unfreeze_last_block_finetune:
        set_requires_grad(model.base.blocks[-1], True)
        set_requires_grad(model.base.head, True)
        if model.base.bond_gine is not None:
            set_requires_grad(model.base.bond_gine.convs[-1], True)
            set_requires_grad(model.base.bond_gine.norms[-1], True)
    else:
        set_requires_grad(model.base, True)
    set_requires_grad(model.qhead, True)
    set_requires_grad(model.angle_head, True)
    model.q_scale.requires_grad = True

    best_finetune = run_stage(
        stage_name="finetune",
        epochs=epochs_finetune, lr=lr_finetune, patience=patience_finetune,
        **_stage_kw(),
    )
    ft_improvement = best_quantum - best_finetune
    log(f"  [finetune] best_val={best_finetune:.6f} | improvement over quantum: {ft_improvement:.6f}")
    log(f"  [finetune] q_scale={float(model.q_scale.item()):.6f}")

    # ---- Final test evaluation -----------------------------------------------
    best_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)
    if device.type == "cuda":
        model.qhead.to(torch.device("cpu"))

    y_m = best_ckpt["y_mean"].to(device)
    y_s = best_ckpt["y_std"].to(device)

    final_val  = eval_mae(model, val_loader,  device, y_m, y_s, use_amp, amp_dtype)
    final_test = eval_mae(model, test_loader, device, y_m, y_s, use_amp, amp_dtype)
    final_qscale = float(best_ckpt.get("q_scale", model.q_scale.item()))

    log_section("FINAL RESULTS")
    log(f"  Tag      : {tag}")
    log(f"  Val  MAE : {final_val:.6f}")
    log(f"  Test MAE : {final_test:.6f}")
    log(f"  q_scale  : {final_qscale:.6f}")
    log(f"  Best epoch (overall): {best_ckpt.get('best_epoch', -1)}")
    log(f"  Stage MAEs — base: {best_base:.6f} | quantum: {best_quantum:.6f} | finetune: {best_finetune:.6f}")
    total_q_gain = best_base - final_val
    log(f"  Total quantum+finetune gain: {total_q_gain:.6f} ({100*total_q_gain/best_base:.2f}% over base)")
    log(f"  Ckpt : {ckpt_path}")
    log(f"  Log  : {log_path}")
    log(f"  Qdiag: {diag_path}")
    log_separator()

    return {
        "tag":         tag,
        "val_mae":     final_val,
        "test_mae":    final_test,
        "q_scale":     final_qscale,
        "best_epoch":  int(best_ckpt.get("best_epoch", -1)),
        "ckpt_path":   ckpt_path,
        "log_path":    log_path,
        "diag_path":   diag_path,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.cache_size_limit = 64
    
    ap = argparse.ArgumentParser("QGINE_v10: V10 classical backbone + quantum correction")

    # Data
    ap.add_argument("--qm9_root",   type=str, default="./qm9_data")
    ap.add_argument("--cache_root", type=str, default="./op2_cache_v10")
    ap.add_argument("--exp_name",   type=str, default="qgine_v10")
    ap.add_argument("--target_idx", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--train_seed", type=int, default=0)

    # Classical backbone sweep
    ap.add_argument("--hiddens",     nargs="+", type=int, default=[256])
    ap.add_argument("--layers_list", nargs="+", type=int, default=[4])
    ap.add_argument("--dropout",     type=float, default=0.0)
    ap.add_argument("--readout",     type=str,   default="add+mean",
                    choices=["add", "mean", "add+mean"])

    # Graph construction
    ap.add_argument("--graph_type",        type=str,   default="radius", choices=["radius", "knn"])
    ap.add_argument("--radius",            type=float, default=5.0)
    ap.add_argument("--knn_k",             type=int,   default=12)
    ap.add_argument("--max_num_neighbors", type=int,   default=48)
    ap.add_argument("--rbf_bins",          type=int,   default=64)
    ap.add_argument("--rbf_max",           type=float, default=5.0)

    # Model features
    ap.add_argument("--head_type",        type=str,   default="mlp", choices=["mlp", "dipole"])
    ap.add_argument("--use_virtual_node", dest="use_virtual_node", action="store_true")
    ap.add_argument("--no_virtual_node",  dest="use_virtual_node", action="store_false")
    ap.set_defaults(use_virtual_node=True)
    ap.add_argument("--edge_dropout",     type=float, default=0.0)
    ap.add_argument("--noisy_node_std",   type=float, default=0.1)
    ap.add_argument("--no_3body",         action="store_true")
    ap.add_argument("--angle_rbf_bins",   type=int,   default=32)

    # Bond GINE
    ap.add_argument("--bond_gine_layers", type=int,   default=2)
    ap.add_argument("--bond_gine_mode",   type=str,   default="interleave", choices=["pre", "interleave"])
    ap.add_argument("--bond_use_dist",    dest="bond_use_dist", action="store_true")
    ap.add_argument("--no_bond_use_dist", dest="bond_use_dist", action="store_false")
    ap.set_defaults(bond_use_dist=True)
    ap.add_argument("--bond_rbf_bins",    type=int,   default=16)
    ap.add_argument("--bond_rbf_max",     type=float, default=2.0)
    ap.add_argument("--bond_train_eps",   action="store_true")

    # Quantum
    ap.add_argument("--q_p",              type=int,   default=4,
                    help="Number of qubits in the quantum circuit")
    ap.add_argument("--q_Lq",             type=int,   default=2,
                    help="Number of variational layers")
    ap.add_argument("--q_measure",        type=str,   default="z+zz",  choices=["z", "z+zz"])
    ap.add_argument("--q_device_name",    type=str,   default="default.qubit",
                    help="PennyLane device: default.qubit or lightning.qubit")
    ap.add_argument("--q_shots",          type=int,   default=None)
    ap.add_argument("--q_diff_method",    type=str,   default="backprop",
                    choices=["backprop", "parameter-shift", "adjoint"])
    ap.add_argument("--q_broadcast_mode", type=str,   default="native",
                    choices=["native", "expand", "off"])
    ap.add_argument("--q_init_std",       type=float, default=0.02)
    ap.add_argument("--q_readout_hidden", type=int,   default=32)
    ap.add_argument("--q_scale_init",     type=float, default=0.0,
                    help="Initial value for q_scale. 0.0 = pure classical at start.")

    # Optional pretrained classical checkpoint
    ap.add_argument("--init_v8_ckpt", type=str, default=None,
                    help="Path to a pretrained V8 checkpoint to warm-start the classical base.")

    # Skip base stage by loading an existing QGINE checkpoint
    ap.add_argument("--load_base_ckpt", type=str, default=None,
                    help="Path to a saved best_*.pt from a previous run. Skips base training "
                         "and loads weights directly. Use with --num_workers_quantum 0 to avoid deadlock.")

    # Training stages
    ap.add_argument("--epochs_base",     type=int,   default=200)
    ap.add_argument("--patience_base",   type=int,   default=80)
    ap.add_argument("--lr_base",         type=float, default=1e-3)

    ap.add_argument("--epochs_quantum",  type=int,   default=100)
    ap.add_argument("--patience_quantum",type=int,   default=30)
    ap.add_argument("--lr_quantum",      type=float, default=3e-4)
    ap.add_argument("--freeze_base_during_quantum", action="store_true",
                    help="If set, classical base weights are frozen during quantum stage.")

    ap.add_argument("--epochs_finetune",  type=int,   default=50)
    ap.add_argument("--patience_finetune",type=int,   default=20)
    ap.add_argument("--lr_finetune",      type=float, default=1e-4)
    ap.add_argument("--unfreeze_last_block_finetune", action="store_true",
                    help="If set, only unfreeze last EquiBlock during finetune (not full base).")

    ap.add_argument("--weight_decay",   type=float, default=1e-5)
    ap.add_argument("--scheduler",      type=str,   default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--warmup_epochs",  type=int,   default=10)
    ap.add_argument("--loss",           type=str,   default="l1",    choices=["l1", "huber"])
    ap.add_argument("--huber_delta",    type=float, default=1.0)
    ap.add_argument("--ema",    dest="ema", action="store_true")
    ap.add_argument("--no_ema", dest="ema", action="store_false")
    ap.set_defaults(ema=True)
    ap.add_argument("--ema_decay",      type=float, default=0.9999)

    # Hardware
    ap.add_argument("--amp",    dest="amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--amp_dtype",        type=str,   default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--compile",          action="store_true")
    ap.add_argument("--batch_size_train", type=int,   default=256)
    ap.add_argument("--batch_size_eval",  type=int,   default=512)
    ap.add_argument("--batch_size_quantum", type=int, default=None, help="Batch size for quantum stage. Defaults to batch_size_train. ""Recommend 16-32 when using a CPU simulator (lightning.qubit, default.qubit).")
    ap.add_argument("--num_workers_base",     type=int, default=16)
    ap.add_argument("--num_workers_quantum",  type=int, default=8)
    ap.add_argument("--num_workers_finetune", type=int, default=16)

    args = ap.parse_args()
    use_3body = not args.no_3body

    seed_all(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init run log file
    run_log_dir = os.path.join(
        "runs_qgine_v10", f"target{args.target_idx}", args.exp_name,
        f"split{args.split_seed}", f"train{args.train_seed}",
    )
    os.makedirs(run_log_dir, exist_ok=True)
    _init_run_log(os.path.join(run_log_dir, "run.log"))

    log_section("QGINE v10 — STARTUP")
    log(f"  exp_name   : {args.exp_name}")
    log(f"  target_idx : {args.target_idx}")
    log(f"  split_seed : {args.split_seed}  train_seed: {args.train_seed}")
    log(f"  Device     : {device}")
    if device.type == "cuda":
        log(f"  GPU        : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log(f"  VRAM       : {props.total_memory / 1024**3:.1f} GB")
    log(f"  PyTorch    : {torch.__version__}")
    log(f"  PennyLane  : {qml.__version__}")
    log(f"  Python     : {sys.version.split()[0]}")
    log_separator()

    base = QM9(args.qm9_root)
    key  = Op2CacheKey(
        graph_type=args.graph_type, radius=args.radius,
        knn_k=args.knn_k, max_num_neighbors=args.max_num_neighbors,
    )
    cache_dir = os.path.join(args.cache_root, key.dirname())

    ei_list, ed_list, kj_list, ji_list = build_or_load_edge_cache(
        base, cache_dir,
        graph_type=args.graph_type, radius=args.radius,
        knn_k=args.knn_k, max_num_neighbors=args.max_num_neighbors,
    )
    dataset = CachedQM9(base, ei_list, ed_list, kj_list, ji_list)

    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    gen     = torch.Generator().manual_seed(args.split_seed)
    perm    = torch.randperm(n, generator=gen).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    test_ds  = torch.utils.data.Subset(dataset, test_idx)

    t       = int(args.target_idx)
    y_sel   = base._data.y[:, t:t+1]
    train_y = y_sel[train_idx]
    y_mean  = train_y.mean()
    y_std   = train_y.std().clamp_min(1e-12)
    base._data.y = (y_sel - y_mean) / y_std

    out_dir = os.path.join(
        "runs_qgine_v10", f"target{t}", args.exp_name,
        f"split{args.split_seed}", f"train{args.train_seed}",
    )
    os.makedirs(out_dir, exist_ok=True)

    log(f"  [Cache]  dir={cache_dir}")
    log(f"  [Split]  train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    log(f"  [Norm]   y_mean={float(y_mean):.6f} y_std={float(y_std):.6f}")
    log(f"  [3Body]  use_3body={use_3body} angle_rbf_bins={args.angle_rbf_bins}")
    log(f"  [Quantum] p={args.q_p} Lq={args.q_Lq} measure={args.q_measure} "
        f"diff={args.q_diff_method} device={args.q_device_name}")
    log(f"  [Stages] base={args.epochs_base} quantum={args.epochs_quantum} "
        f"finetune={args.epochs_finetune}")
    log(f"  [Freeze] base_during_quantum={args.freeze_base_during_quantum} "
        f"last_block_finetune={args.unfreeze_last_block_finetune}")

    results = []
    for h in args.hiddens:
        for L in args.layers_list:
            print("\n" + "=" * 80)
            print(f"RUN: hidden={h} layers={L} target={t}")

            r = run_one_config(
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                device=device, target_idx=t, out_dir=out_dir,
                hidden=h, layers=L, dropout=args.dropout, readout=args.readout,
                y_mean=y_mean, y_std=y_std, rbf_bins=args.rbf_bins, rbf_max=args.rbf_max,
                split_seed=args.split_seed, train_seed=args.train_seed,
                batch_size_train=args.batch_size_train,
                batch_size_eval=args.batch_size_eval,
                batch_size_quantum=args.batch_size_quantum,
                num_workers_base=args.num_workers_base,
                num_workers_quantum=args.num_workers_quantum,
                num_workers_finetune=args.num_workers_finetune,
                use_amp=args.amp, amp_dtype=args.amp_dtype, use_compile=args.compile,
                scheduler_type=args.scheduler, warmup_epochs=args.warmup_epochs,
                loss_type=args.loss, huber_delta=args.huber_delta,
                use_virtual_node=args.use_virtual_node, head_type=args.head_type,
                edge_dropout=args.edge_dropout, noisy_node_std=args.noisy_node_std,
                use_ema=args.ema, ema_decay=args.ema_decay,
                bond_gine_layers=args.bond_gine_layers, bond_gine_mode=args.bond_gine_mode,
                bond_use_dist=args.bond_use_dist, bond_rbf_bins=args.bond_rbf_bins,
                bond_rbf_max=args.bond_rbf_max, bond_train_eps=args.bond_train_eps,
                use_3body=use_3body, angle_rbf_bins=args.angle_rbf_bins,
                q_p=args.q_p, q_Lq=args.q_Lq, q_measure=args.q_measure,
                q_device_name=args.q_device_name, q_shots=args.q_shots,
                q_diff_method=args.q_diff_method, q_broadcast_mode=args.q_broadcast_mode,
                q_init_std=args.q_init_std, q_readout_hidden=args.q_readout_hidden,
                q_scale_init=args.q_scale_init,
                epochs_base=args.epochs_base, patience_base=args.patience_base, lr_base=args.lr_base,
                epochs_quantum=args.epochs_quantum, patience_quantum=args.patience_quantum, lr_quantum=args.lr_quantum,
                freeze_base_during_quantum=args.freeze_base_during_quantum,
                epochs_finetune=args.epochs_finetune, patience_finetune=args.patience_finetune, lr_finetune=args.lr_finetune,
                unfreeze_last_block_finetune=args.unfreeze_last_block_finetune,
                weight_decay=args.weight_decay,
                init_v8_ckpt=args.init_v8_ckpt,
                load_base_ckpt=args.load_base_ckpt,
            )
            results.append(r)

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = min(results, key=lambda d: d["val_mae"])
    log_section("ALL RUNS COMPLETE")
    log(f"  BEST: {best['tag']}")
    log(f"  val={best['val_mae']:.6f} | test={best['test_mae']:.6f} | q_scale={best['q_scale']:.6f}")
    log(f"  summary: {summary_path}")
    _close_run_log()


if __name__ == "__main__":
    main()