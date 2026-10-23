"""Download AlphaFold2 predicted structures for every protein in a mapping CSV."""
import argparse
import os
from time import sleep

import pandas as pd
import requests


def download(protein_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    failed = []
    for i, pid in enumerate(protein_ids, 1):
        path = os.path.join(out_dir, f'{pid}.pdb')
        if os.path.exists(path):
            continue
        try:
            r = requests.get(f'https://alphafold.ebi.ac.uk/api/prediction/{pid}', timeout=30)
            pdb_url = r.json()[0].get('pdbUrl') if r.status_code == 200 and r.json() else None
            pdb = requests.get(pdb_url, timeout=30) if pdb_url else None
            if pdb is not None and pdb.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(pdb.content)
                print(f'{i}/{len(protein_ids)} {pid} downloaded')
            else:
                failed.append(pid)
        except Exception as e:  # network / JSON errors
            print(f'{pid}: {str(e)[:60]}')
            failed.append(pid)
        sleep(0.5)
    print(f'Failed: {len(failed)}')
    return failed


if __name__ == '__main__':
    from dta.config import DATA_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', default=os.path.join(DATA_DIR, 'Davis_protein_mapping.csv'))
    ap.add_argument('--out', default=os.path.join(DATA_DIR, 'PDB_AF2'))
    args = ap.parse_args()
    download(pd.read_csv(args.mapping)['PROTEIN_ID'].unique().tolist(), args.out)
