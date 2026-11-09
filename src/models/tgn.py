"""
TGN (Temporal Graph Network) for temporal link prediction on tgbl-flight.

Architecture:
  1. TGNMemory: maintains a memory vector per node, updated via GRU
     after each interaction. Uses IdentityMessage (concatenate src memory,
     dst memory, edge feature, time encoding) and LastAggregator (keep
     only the most recent message per node).

  2. GraphAttentionEmbedding: refines memory vectors using 1-hop neighbor
     attention via TransformerConv. Optional — can be skipped for a
     faster memory-only model.

  3. LinkPredictor: MLP that takes concatenated (src_emb, dst_emb) and
     outputs a scalar interaction score.

Reference: Rossi et al., "Temporal Graph Networks for Deep Learning on
Dynamic Graphs", ICML 2020 Workshop.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import TGNMemory, IdentityMessage, LastAggregator


class GraphAttentionEmbedding(nn.Module):
    """Refine node memory with 1-hop attention over recent neighbors.

    Uses TransformerConv which computes multi-head attention between
    each node and its neighbors, with edge features (time encoding +
    raw message) as key/value bias.
    """

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int,
                 time_enc_dim: int):
        super().__init__()
        edge_dim = msg_dim + time_enc_dim
        self.conv = TransformerConv(
            in_channels, out_channels, heads=2, concat=False,
            edge_dim=edge_dim, dropout=0.1,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        # Compute relative time encoding for edges
        src_idx = edge_index[0]
        rel_t = (t - last_update[src_idx]).float().unsqueeze(-1)
        rel_t_enc = rel_t.cos()  # simple time encoding for edges
        edge_attr = torch.cat([msg, rel_t_enc], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(nn.Module):
    """MLP that scores (src, dst) pairs from their embeddings."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, z_src, z_dst):
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


def build_tgn(num_nodes: int, raw_msg_dim: int = 1, memory_dim: int = 100,
              time_dim: int = 100, use_gnn: bool = False):
    """Build TGN components.

    Args:
        num_nodes: number of nodes in the graph
        raw_msg_dim: dimensionality of edge features (1 for encoded typecode)
        memory_dim: hidden dimension for node memory vectors
        time_dim: dimension for time encoding
        use_gnn: if True, add GraphAttentionEmbedding layer

    Returns:
        dict with 'memory', 'gnn' (or None), 'link_pred', 'neighbor_loader'
    """
    from torch_geometric.nn.models.tgn import LastNeighborLoader

    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=raw_msg_dim,
        memory_dim=memory_dim,
        time_dim=time_dim,
        message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    )

    gnn = None
    embedding_dim = memory_dim
    if use_gnn:
        gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=memory_dim,
            msg_dim=raw_msg_dim,
            time_enc_dim=1,
        )

    link_pred = LinkPredictor(in_channels=2 * embedding_dim)

    neighbor_loader = LastNeighborLoader(num_nodes, size=10)

    return {
        "memory": memory,
        "gnn": gnn,
        "link_pred": link_pred,
        "neighbor_loader": neighbor_loader,
    }
