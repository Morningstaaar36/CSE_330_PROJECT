import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from dta.config import DRUGS_DIR, ESM_DIR, PROTEINS_DIR


class DTADataset(Dataset):
    """Drug graph, protein pocket graph, pocket ESM embedding and pKd label for each pair in a split CSV."""

    def __init__(self, csv_file, drugs_dir=DRUGS_DIR, proteins_dir=PROTEINS_DIR, esm_dir=ESM_DIR):
        self.df = pd.read_csv(csv_file)
        self.drugs_dir, self.proteins_dir, self.esm_dir = drugs_dir, proteins_dir, esm_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid, pid = row['COMPOUND_ID'], row['PROTEIN_ID']
        drug = torch.load(os.path.join(self.drugs_dir, f'{cid}.pt'), weights_only=False)
        p = torch.load(os.path.join(self.proteins_dir, f'{pid}.pt'), weights_only=False)
        protein = Data(x=p['x'], edge_index=p['edge_index'], edge_attr=p['edge_attr'],
                       laplacian_pe=p['laplacian_pe'])
        esm = np.load(os.path.join(self.esm_dir, f'{pid}.npy'), allow_pickle=True)
        return drug, protein, esm, float(row['REG_LABEL']), cid, pid


def collate_dta(batch):
    drugs, proteins, esms, labels, cids, pids = zip(*batch)
    max_len = max(e.shape[0] for e in esms)
    esm = np.stack([np.pad(e, ((0, max_len - e.shape[0]), (0, 0))) for e in esms])
    return (Batch.from_data_list(drugs), Batch.from_data_list(proteins),
            torch.FloatTensor(esm), torch.FloatTensor(labels), list(cids), list(pids))
