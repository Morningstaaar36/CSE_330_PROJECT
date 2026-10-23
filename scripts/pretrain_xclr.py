"""Stage 1: X-CLR contrastive pretraining of the drug and protein graph-transformer encoders.

Soft targets come from frozen oracles: ChemBERTa ([CLS] of SMILES) for drugs and mean-pooled
ESM-2 for proteins. Only training-split entities are used, so cold-start test entities stay unseen.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dta import config as C
from dta.losses import xclr_loss
from dta.models.graph_transformer import GraphTransformer, scatter


def chemberta_oracle(smiles, device):
    from transformers import AutoModel, AutoTokenizer
    name = 'seyonec/ChemBERTa-zinc-base-v1'
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).to(device).eval()
    with torch.no_grad():
        enc = tok(list(smiles), padding=True, truncation=True, return_tensors='pt').to(device)
        return model(**enc).last_hidden_state[:, 0].cpu()                 # [N, 768]


def load_entities(kind, ids):
    graphs = []
    for i in ids:
        g = torch.load(os.path.join(C.DRUGS_DIR if kind == 'drug' else C.PROTEINS_DIR, f'{i}.pt'),
                       weights_only=False)
        if isinstance(g, dict):
            g = Data(**g)
        graphs.append(g)
    return graphs


def pretrain(encoder, graphs, oracle, args, device):
    encoder.to(device).train()
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=5)
    n = len(graphs)
    for epoch in range(1, args.epochs + 1):
        perm, total = np.random.permutation(n), 0.0
        opt.zero_grad()
        for step, s in enumerate(range(0, n, args.batch_size), 1):
            idx = perm[s:s + args.batch_size]
            if len(idx) < 2:
                continue
            bg = Batch.from_data_list([graphs[i] for i in idx]).to(device)
            z = scatter(encoder(bg), bg.batch, dim=0, dim_size=bg.num_graphs, reduce='mean')
            loss = xclr_loss(z, oracle[idx].to(device), args.tau, args.tau_s) / args.accum
            loss.backward()
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            total += loss.item() * args.accum
        sched.step(total)
        print(f'epoch {epoch:3d}  loss {total:.4f}')
    return encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='cold_drug', choices=C.SCENARIOS)
    ap.add_argument('--fold', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.05)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(C.CKPT_DIR, exist_ok=True)

    train = pd.read_csv(C.split_csv(args.scenario, args.fold, 'train'))
    drug_ids, prot_ids = train['COMPOUND_ID'].unique(), train['PROTEIN_ID'].unique()

    drugs = load_entities('drug', drug_ids)
    drug_oracle = chemberta_oracle([g.smiles for g in drugs], device)
    compound_gt = GraphTransformer(C.NUM_LAYERS, 44, 10, C.HIDDEN_DIM, C.HIDDEN_DIM, C.NUM_HEADS)
    pretrain(compound_gt, drugs, drug_oracle, args, device)
    torch.save(compound_gt.state_dict(),
               os.path.join(C.CKPT_DIR, f'stage1_compound_{args.scenario}_fold{args.fold}.pth'))

    prots = load_entities('protein', prot_ids)
    prot_oracle = torch.stack([torch.tensor(np.load(os.path.join(C.ESM_DIR, f'{p}.npy'))).float().mean(0)
                               for p in prot_ids])
    protein_gt = GraphTransformer(C.NUM_LAYERS, 41, 5, C.HIDDEN_DIM, C.HIDDEN_DIM, C.NUM_HEADS)
    pretrain(protein_gt, prots, prot_oracle, args, device)
    torch.save(protein_gt.state_dict(),
               os.path.join(C.CKPT_DIR, f'stage1_protein_{args.scenario}_fold{args.fold}.pth'))


if __name__ == '__main__':
    main()
