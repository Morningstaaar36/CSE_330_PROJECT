"""X-CLR: contrastive loss with soft targets from frozen oracle encoders (ChemBERTa / ESM-2)."""
import torch
import torch.nn.functional as F


def xclr_loss(z, oracle, tau=0.1, tau_s=0.05):
    """z: [B, d] learned embeddings; oracle: [B, d'] frozen reference embeddings."""
    z = F.normalize(z, dim=-1)
    g = F.normalize(oracle, dim=-1)
    off_diag = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    logits = (z @ z.T / tau).masked_fill(~off_diag, -1e9)
    targets = F.softmax((g @ g.T / tau_s).masked_fill(~off_diag, -1e9), dim=-1)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(-1).mean()
