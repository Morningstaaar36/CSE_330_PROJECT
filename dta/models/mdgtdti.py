"""MDGTDTInet: drug GT + protein GT/ESM cross-attention + bilinear interaction attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_transformer import GraphTransformer, scatter


class CrossAttention(nn.Module):
    """Scaled dot-product multi-head attention (query from ESM, key/value from protein graph)."""

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim, self.n_heads = hid_dim, n_heads
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.tensor([hid_dim // n_heads], dtype=torch.float)),
                             persistent=False)

    def _heads(self, x):
        b = x.shape[0]
        return x.view(b, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        Q, K, V = self._heads(self.w_q(query)), self._heads(self.w_k(key)), self._heads(self.w_v(value))
        energy = Q @ K.transpose(-1, -2) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        x = self.do(F.softmax(energy, dim=-1)) @ V
        x = x.permute(0, 2, 1, 3).contiguous().view(query.shape[0], -1, self.hid_dim)
        return self.fc(x)


def to_dense_batch(bg, feats):
    """[N_total, d] node features of a PyG Batch -> zero-padded [B, max_nodes, d]."""
    batch = bg.batch
    num_nodes = scatter(torch.ones_like(batch), batch, dim=0, dim_size=bg.num_graphs)
    max_nodes = int(num_nodes.max())
    cum = torch.cat([batch.new_zeros(1), num_nodes.cumsum(0)])
    idx = torch.arange(feats.size(0), device=feats.device) - cum[batch] + batch * max_nodes
    out = feats.new_zeros([bg.num_graphs * max_nodes] + list(feats.shape[1:]))
    out[idx] = feats
    return out.view([bg.num_graphs, max_nodes] + list(feats.shape[1:]))


class MDGTDTInet(nn.Module):
    def __init__(self, compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=4, esm_dim=320, out_dim=1):
        super().__init__()
        self.crossAttention = CrossAttention(hid_dim=compound_dim, n_heads=1, dropout=0.2)
        self.compound_gt = GraphTransformer(gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                            out_dim=compound_dim, n_heads=gt_heads)
        self.protein_gt = GraphTransformer(gt_layers, node_dim=41, edge_dim=5, hidden_dim=protein_dim,
                                           out_dim=protein_dim, n_heads=gt_heads)
        self.protein_embedding_fc = nn.Linear(esm_dim, protein_dim)

        self.joint_attn_prot = nn.Linear(compound_dim, compound_dim)
        self.joint_attn_comp = nn.Linear(compound_dim, compound_dim)
        self.classifier = nn.Sequential(
            nn.Linear(compound_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim))

    def load_pretrained_encoders(self, compound_ckpt, protein_ckpt, map_location='cpu'):
        """Initialise the two GT encoders from Stage-1 (X-CLR) weights."""
        self.compound_gt.load_state_dict(torch.load(compound_ckpt, map_location=map_location))
        self.protein_gt.load_state_dict(torch.load(protein_ckpt, map_location=map_location))

    def forward(self, compound_graph, protein_graph, protein_embedding):
        compound_feats = to_dense_batch(compound_graph, self.compound_gt(compound_graph))   # [B, Nd, 128]
        protein_graph_feats = to_dense_batch(protein_graph, self.protein_gt(protein_graph))  # [B, Np, 128]

        # ESM residues attend to structurally related pocket residues
        esm = self.protein_embedding_fc(protein_embedding)                                    # [B, L, 128]
        protein_feats = self.crossAttention(esm, protein_graph_feats, protein_graph_feats)    # [B, L, 128]

        # Bilinear atom-residue interaction map, normalised to sum to 1 per pair
        inter = torch.sigmoid(torch.einsum('bij,bkj->bik',
                                           self.joint_attn_prot(F.relu(protein_feats)),
                                           self.joint_attn_comp(F.relu(compound_feats))))
        inter = inter / inter.sum(dim=(1, 2)).view(-1, 1, 1)

        joint = torch.tanh(torch.einsum('bij,bkj->bikj', protein_feats, compound_feats))
        cp_embedding = torch.einsum('bijk,bij->bk', joint, inter)                             # [B, 128]
        return self.classifier(cp_embedding)
