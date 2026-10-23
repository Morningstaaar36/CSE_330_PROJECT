"""Evaluate a trained checkpoint on a test split: metrics, predictions.csv and plots."""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dta import config as C
from dta.data.dataset import DTADataset, collate_dta
from dta.metrics import all_metrics
from dta.models.mdgtdti import MDGTDTInet

ACCENT, ACCENT2, GRAY = '#5C6BC0', '#EF5350', '#90A4AE'


def plot_results(y, p, m, title, out_dir):
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, p, alpha=0.35, s=14, color=ACCENT, linewidths=0, label='Predictions')
    lo, hi = min(y.min(), p.min()) - 0.2, max(y.max(), p.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], '--', color=GRAY, lw=1.5, label='Ideal (y=x)')
    slope, icpt = np.polyfit(y, p, 1)
    ax.plot([lo, hi], [slope * lo + icpt, slope * hi + icpt], color=ACCENT2, lw=1.5, label='Linear fit')
    ax.text(0.04, 0.96, '\n'.join(f'{k} = {m[k]:.4f}' for k in ('MSE', 'RMSE', 'CI', 'Pearson')),
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=GRAY, alpha=0.85))
    ax.set(xlabel='Actual pKd', ylabel='Predicted pKd', xlim=(lo, hi), ylim=(lo, hi))
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'predicted_vs_actual.png'), bbox_inches='tight')
    plt.close(fig)

    r = p - y
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=40, color=ACCENT, edgecolor='white', alpha=0.85, density=True, label='Residuals')
    xs = np.linspace(r.min(), r.max(), 300)
    ax.plot(xs, np.exp(-0.5 * ((xs - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi)),
            color=ACCENT2, lw=2, label=f'N(μ={r.mean():.3f}, σ={r.std():.3f})')
    ax.axvline(0, color=GRAY, linestyle='--')
    ax.set(xlabel='Prediction error (predicted − actual)', ylabel='Density')
    ax.set_title('Error Distribution', fontweight='bold'); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'error_distribution.png'), bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='cold_drug', choices=C.SCENARIOS)
    ap.add_argument('--fold', type=int, default=4)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = args.checkpoint or os.path.join(C.CKPT_DIR, f'best_{args.scenario}_fold{args.fold}.pth')

    model = MDGTDTInet(C.HIDDEN_DIM, C.HIDDEN_DIM, C.NUM_LAYERS, C.NUM_HEADS, C.ESM_DIM).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    loader = DataLoader(DTADataset(C.split_csv(args.scenario, args.fold, 'test')),
                        batch_size=args.batch_size, collate_fn=collate_dta)
    ys, ps, cids, pids = [], [], [], []
    with torch.no_grad():
        for drug, prot, esm, y, c, p in loader:
            ps += model(drug.to(device), prot.to(device), esm.to(device)).view(-1).cpu().tolist()
            ys += y.tolist(); cids += c; pids += p
    y, p = np.array(ys), np.array(ps)
    m = all_metrics(y, p)
    for k, v in m.items():
        print(f'{k:<10}{v:>10.4f}')

    out = os.path.join(C.RESULTS_DIR, args.scenario)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({'COMPOUND_ID': cids, 'PROTEIN_ID': pids, 'actual': y, 'predicted': p, 'error': p - y}) \
        .to_csv(os.path.join(out, 'predictions.csv'), index=False)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump({'scenario': args.scenario, 'fold': args.fold, 'n_test': len(y), **m}, f, indent=2)
    plot_results(y, p, m, f'{args.scenario.replace("_", "-").title()} · Fold {args.fold}', out)


if __name__ == '__main__':
    main()
