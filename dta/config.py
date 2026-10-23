import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data', 'Davis')

DRUGS_DIR = os.path.join(DATA_DIR, 'drugs')                   # PyG drug graphs  (<COMPOUND_ID>.pt)
PROTEINS_DIR = os.path.join(DATA_DIR, 'proteins')             # pocket graphs     (<PROTEIN_ID>.pt)
ESM_DIR = os.path.join(DATA_DIR, 'ESM_embedding_pocket')      # ESM-2 [L x 320]   (<PROTEIN_ID>.npy)
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')                 # <scenario>/fold/<k>/Davis_{train,val,test}.csv
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(ROOT, 'results')

SCENARIOS = ['random', 'cold_drug', 'cold_protein', 'cold_both']

# Model hyper-parameters (must match released checkpoints)
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 10
DROPOUT = 0.2
ESM_DIM = 320


def split_csv(scenario, fold, part):
    return os.path.join(SPLITS_DIR, scenario, 'fold', str(fold), f'Davis_{part}.csv')
