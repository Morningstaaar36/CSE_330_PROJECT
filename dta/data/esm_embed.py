"""Per-residue ESM-2 (t6, 320-d) embeddings, sliced to the binding-pocket residues."""
import argparse
import os

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def embed_sequences(ids, seqs, device):
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)
    convert = alphabet.get_batch_converter()
    for pid, seq in zip(ids, seqs):
        _, _, tokens = convert([(pid, seq)])
        rep = model(tokens.to(device), repr_layers=[6])['representations'][6]
        yield pid, rep[0, 1:-1].cpu().numpy()          # drop BOS/EOS -> [L, 320]


if __name__ == '__main__':
    from dta.config import DATA_DIR, ESM_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', default=os.path.join(DATA_DIR, 'Davis_protein_mapping.csv'))
    ap.add_argument('--pocket_index', default=os.path.join(DATA_DIR, 'pocket_index.pt'))
    ap.add_argument('--out', default=ESM_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.mapping)
    pockets = torch.load(args.pocket_index)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for pid, emb in embed_sequences(df['PROTEIN_ID'], df['PROTEIN_SEQUENCE'], device):
        if pid in pockets:
            idx = [i for i in pockets[pid] if i < len(emb)]
            np.save(os.path.join(args.out, f'{pid}.npy'), emb[idx])
