"""Stage 2: train MDGTDTInet for affinity regression, initialised from Stage-1 encoders."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dta import config as C
from dta.data.dataset import DTADataset, collate_dta
from dta.metrics import all_metrics
from dta.models.mdgtdti import MDGTDTInet


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for drug, prot, esm, y, _, _ in loader:
        ps.append(model(drug.to(device), prot.to(device), esm.to(device)).view(-1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='cold_drug', choices=C.SCENARIOS)
    ap.add_argument('--fold', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--no_pretrain', action='store_true', help='random init instead of X-CLR encoders')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    loaders = {part: DataLoader(DTADataset(C.split_csv(args.scenario, args.fold, part)),
                                batch_size=args.batch_size, shuffle=(part == 'train'), collate_fn=collate_dta)
               for part in ('train', 'val', 'test')}

    model = MDGTDTInet(C.HIDDEN_DIM, C.HIDDEN_DIM, C.NUM_LAYERS, C.NUM_HEADS, C.ESM_DIM)
    if not args.no_pretrain:
        tag = f'{args.scenario}_fold{args.fold}'
        model.load_pretrained_encoders(os.path.join(C.CKPT_DIR, f'stage1_compound_{tag}.pth'),
                                       os.path.join(C.CKPT_DIR, f'stage1_protein_{tag}.pth'))
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.8, patience=10)
    loss_fn = torch.nn.MSELoss()
    best, ckpt = float('inf'), os.path.join(C.CKPT_DIR, f'best_{args.scenario}_fold{args.fold}.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        for drug, prot, esm, y, _, _ in loaders['train']:
            out = model(drug.to(device), prot.to(device), esm.to(device))
            loss = loss_fn(out.view(-1), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        val = all_metrics(*predict(model, loaders['val'], device))
        sched.step(val['MSE'])
        if val['MSE'] < best:
            best = val['MSE']
            torch.save(model.state_dict(), ckpt)
        print(f'epoch {epoch:3d}  val MSE {val["MSE"]:.4f}  CI {val["CI"]:.4f}')

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = all_metrics(*predict(model, loaders['test'], device))
    print('test', {k: round(v, 4) for k, v in test.items()})


if __name__ == '__main__':
    main()
