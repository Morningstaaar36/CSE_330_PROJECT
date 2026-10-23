import numpy as np
from scipy import stats


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def concordance_index(y, p):
    dt = y[:, None] - y[None, :]
    dp = p[:, None] - p[None, :]
    mask = dt > 0
    return float(((mask & (dp > 0)).sum() + 0.5 * (mask & (dp == 0)).sum()) / max(mask.sum(), 1))


def rm2(y, p):
    """Modified r^2 (Roy et al.), as used by DeepDTA / GraphDTA."""
    r2 = np.corrcoef(y, p)[0, 1] ** 2
    k = np.sum(y * p) / np.sum(p * p)
    r02 = 1 - np.sum((y - k * p) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(r2 * (1 - np.sqrt(abs(r2 - r02))))


def all_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {'MSE': mse(y, p), 'RMSE': float(np.sqrt(mse(y, p))), 'CI': concordance_index(y, p),
            'rm2': rm2(y, p), 'Pearson': float(stats.pearsonr(y, p)[0]),
            'Spearman': float(stats.spearmanr(y, p)[0])}
