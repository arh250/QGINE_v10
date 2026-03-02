import torch
import torch.nn.functional as F

@torch.no_grad()
def mean_pairwise_cosine_similarity(x: torch.Tensor, batch: torch.Tensor) -> float:
    """
    Compute mean pairwise cosine similarity of node embeddings, averaged per-graph.
    """
    x = F.normalize(x, p=2, dim=-1)

    sims = []
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    for g in range(num_graphs):
        idx = (batch == g).nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue

        xg = x[idx]                 # [Ng, d]
        sim = xg @ xg.t()           # [Ng, Ng]
        Ng = xg.size(0)

        mean_offdiag = (sim.sum() - sim.diag().sum()) / (Ng * (Ng - 1))
        sims.append(mean_offdiag.item())

    return float(sum(sims) / max(len(sims), 1))
from collections import defaultdict, deque
import math

def _to_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    # edge_index: [2, E]
    row, col = edge_index
    rev = torch.stack([col, row], dim=0)
    return torch.cat([edge_index, rev], dim=1)

def _choose_root_max_degree(edge_index: torch.Tensor, num_nodes: int) -> int:
    # Choose a stable "central-ish" root: argmax degree.
    deg = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.numel() > 0:
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
        deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    return int(torch.argmax(deg).item()) if num_nodes > 0 else 0

def _hop_distances(edge_index: torch.Tensor, num_nodes: int, root: int) -> torch.Tensor:
    """
    BFS hop distances from root. Returns [num_nodes] with -1 for unreachable.
    """
    dist = torch.full((num_nodes,), -1, dtype=torch.long)
    if num_nodes == 0:
        return dist

    edge_index = _to_undirected_edge_index(edge_index)
    adj = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adj[u].append(v)

    root = max(0, min(root, num_nodes - 1))
    q = deque([root])
    dist[root] = 0

    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist

def grad_norm_vs_hop_distance(
    node_grads: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[dict[int, float], float]:
    """
    node_grads: [num_nodes, dim] gradient at node embedding (use node_x.grad)
    edge_index: [2, E]
    batch: [num_nodes] graph id per node

    Returns:
      hop_to_mean_grad: dict hop -> mean ||grad|| over nodes at that hop (averaged across graphs)
      slope: linear slope of log(mean_grad) vs hop (more negative => stronger decay => more oversquashing)
    """
    grad_norm = node_grads.norm(p=2, dim=-1)  # [N]
    hop_buckets = defaultdict(list)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    for g in range(num_graphs):
        idx = (batch == g).nonzero(as_tuple=False).view(-1)
        if idx.numel() < 2:
            continue

        # Induce subgraph edge_index in local node ids
        node_map = {int(n.item()): i for i, n in enumerate(idx)}
        local_edges = []
        if edge_index.numel() > 0:
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            for u, v in zip(src, dst):
                if u in node_map and v in node_map:
                    local_edges.append((node_map[u], node_map[v]))

        if len(local_edges) == 0:
            continue

        local_ei = torch.tensor(local_edges, dtype=torch.long).t().contiguous()  # [2, e]
        num_nodes = idx.numel()

        root = _choose_root_max_degree(local_ei, num_nodes)
        hops = _hop_distances(local_ei, num_nodes, root)  # [num_nodes]

        local_grad = grad_norm[idx]  # [num_nodes]
        for h in hops.unique().tolist():
            h = int(h)
            if h < 0:
                continue
            mask = (hops == h)
            if mask.any():
                hop_buckets[h].append(float(local_grad[mask].mean().item()))

    hop_to_mean_grad = {h: float(sum(vals) / max(len(vals), 1)) for h, vals in hop_buckets.items()}

    # Fit slope: log(mean_grad + eps) vs hop
    eps = 1e-12
    xs = sorted(hop_to_mean_grad.keys())
    if len(xs) < 2:
        return hop_to_mean_grad, 0.0

    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor([math.log(hop_to_mean_grad[h] + eps) for h in xs], dtype=torch.float32)

    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum().item()
    slope = float((((x - x_mean) * (y - y_mean)).sum().item()) / (denom + 1e-12))

    return hop_to_mean_grad, slope
