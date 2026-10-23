"""Edge-aware Graph Transformer encoders for drug and protein graphs."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


def scatter(src, index, dim=0, dim_size=None, reduce='sum'):
    """Pure-PyTorch scatter (drop-in for torch_scatter.scatter)."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_zeros(shape)
    idx_shape = [1] * src.dim()
    idx_shape[dim] = index.shape[0]
    index_exp = index.view(idx_shape).expand_as(src)
    if reduce == 'sum':
        out.scatter_add_(dim, index_exp, src)
    elif reduce == 'mean':
        count = src.new_zeros(shape)
        count.scatter_add_(dim, index_exp, torch.ones_like(src))
        out.scatter_add_(dim, index_exp, src)
        out = out / count.clamp(min=1)
    else:
        raise ValueError(f'Unsupported reduce: {reduce}')
    return out


class MultiHeadAttentionLayer(nn.Module):
    """score = (Q_i . K_j / sqrt(d)) * E_ij, softmax over incoming edges of each node."""

    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

    def forward(self, h, edge_index, e):
        num_nodes, num_edges = h.size(0), edge_index.size(1)
        Q_h = self.Q(h).view(num_nodes, self.num_heads, self.out_dim)
        K_h = self.K(h).view(num_nodes, self.num_heads, self.out_dim)
        V_h = self.V(h).view(num_nodes, self.num_heads, self.out_dim)
        proj_e = self.proj_e(e).view(num_edges, self.num_heads, self.out_dim)

        src, dst = edge_index[0], edge_index[1]
        score = (Q_h[dst] * K_h[src]) / (self.out_dim ** 0.5) * proj_e
        e_out = score
        score_scalar = score.sum(dim=-1).clamp(-5, 5)

        attn = torch.stack([pyg_softmax(score_scalar[:, k], dst, num_nodes=num_nodes)
                            for k in range(self.num_heads)], dim=1).unsqueeze(-1)
        h_out = scatter(V_h[src] * attn, dst, dim=0, dim_size=num_nodes)
        z = scatter(attn, dst, dim=0, dim_size=num_nodes)
        return h_out / (z + 1e-6), e_out


class GraphTransformerLayer(nn.Module):
    """Attention -> residual + LayerNorm -> FFN -> residual + LayerNorm, for both nodes and edges."""

    def __init__(self, in_dim, out_dim, num_heads, dropout, residual=True, use_bias=False):
        super().__init__()
        self.out_channels = out_dim
        self.dropout = dropout
        self.residual = residual

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        self.layer_norm1_h = nn.LayerNorm(out_dim)
        self.layer_norm1_e = nn.LayerNorm(out_dim)
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)
        self.layer_norm2_h = nn.LayerNorm(out_dim)
        self.layer_norm2_e = nn.LayerNorm(out_dim)

    def _ffn(self, x, l1, l2):
        return l2(F.dropout(F.relu(l1(x)), self.dropout, training=self.training))

    def forward(self, h, edge_index, e):
        h_in, e_in = h, e
        h, e = self.attention(h, edge_index, e)
        h = self.O_h(F.dropout(h.view(-1, self.out_channels), self.dropout, training=self.training))
        e = self.O_e(F.dropout(e.view(-1, self.out_channels), self.dropout, training=self.training))
        if self.residual:
            h, e = h_in + h, e_in + e
        h, e = self.layer_norm1_h(h), self.layer_norm1_e(e)

        h_in, e_in = h, e
        h = self._ffn(h, self.FFN_h_layer1, self.FFN_h_layer2)
        e = self._ffn(e, self.FFN_e_layer1, self.FFN_e_layer2)
        if self.residual:
            h, e = h_in + h, e_in + e
        return self.layer_norm2_h(h), self.layer_norm2_e(e)


class GraphTransformer(nn.Module):
    """Stack of GraphTransformerLayers with Laplacian positional encoding.

    Drug graphs: node_dim=44, edge_dim=10.  Protein pocket graphs: node_dim=41, edge_dim=5.
    """

    def __init__(self, n_layers, node_dim, edge_dim, hidden_dim, out_dim, n_heads,
                 in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8):
        super().__init__()
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.linear_e = nn.Linear(edge_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout) for _ in range(n_layers - 1)]
            + [GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout)])

    def forward(self, data):
        h = self.linear_h(data.x.float()) + self.embedding_lap_pos_enc(data.laplacian_pe.float())
        h = self.in_feat_dropout(h)
        e = self.linear_e(data.edge_attr.float())
        for layer in self.layers:
            h, e = layer(h, data.edge_index, e)
        return h
