"""SMILES -> PyG molecular graph (44-d atom features, 10-d bond features, 8-d Laplacian PE)."""
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy import sparse as sp
from torch_geometric.data import Data


class SMILESToPyG:
    """
    Converts SMILES strings to PyG Data objects.
    Replicates the DGL implementation exactly.
    """
    
    def __init__(self, pos_enc_dim=8, use_chirality=True, explicit_H=False):
        self.pos_enc_dim = pos_enc_dim
        self.use_chirality = use_chirality
        self.explicit_H = explicit_H
    
    def one_of_k_encoding(self, x, allowable_set):
        """Strict encoding - raises exception if x not in allowable_set"""
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return [x == s for s in allowable_set]
    
    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]
    
    def atom_features(self, atom):
        """Generate atom features including atom symbol(17),degree(7),formal charge(1),
        radical electrons(1),hybridization(6),aromatic(1),hydrogen atoms attached(5),Chirality(3)
        """
        symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']  # 17-dim
        degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
        hybridizationType = [Chem.rdchem.HybridizationType.SP,
                             Chem.rdchem.HybridizationType.SP2,
                             Chem.rdchem.HybridizationType.SP3,
                             Chem.rdchem.HybridizationType.SP3D,
                             Chem.rdchem.HybridizationType.SP3D2,
                             'other']  # 6-dim
        
        results = self.one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
                  self.one_of_k_encoding(atom.GetDegree(), degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  self.one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                      atom.GetIsAromatic()]  # 17+7+2+6+1=33
        
        if not self.explicit_H:
            results = results + self.one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                          [0, 1, 2, 3, 4])  # 33+5=38
        if self.use_chirality:
            try:
                results = results + self.one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 38+3 =41
        
        return results
    
    def bond_features(self, bond):
        """Generate bond features including bond type(4), conjugated(1), in ring(1), stereo(4)"""
        bt = bond.GetBondType()
        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        if self.use_chirality:
            bond_feats = bond_feats + self.one_of_k_encoding_unk(
                str(bond.GetStereo()),
                ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        return np.array(bond_feats).astype(int)
    
    def laplacian_positional_encoding(self, edge_index, num_nodes):
        """
        Graph positional encoding v/ Laplacian eigenvectors
        Exactly replicates the DGL author's implementation
        """
        # Laplacian
        edge_index_np = edge_index.numpy()
        A = sp.csr_matrix(
            (np.ones(edge_index_np.shape[1]),
             (edge_index_np[0], edge_index_np[1])),
            shape=(num_nodes, num_nodes)
        ).astype(float)
        
        # CRITICAL: Use column sum (axis=0) to match author's g.in_degrees()
        # For undirected molecular graphs, in-degree = out-degree
        degrees = np.array(A.sum(axis=0)).flatten()
        
        # Create D^(-1/2) diagonal matrix with clipping to avoid division by zero
        N = sp.diags(degrees.clip(1) ** -0.5, dtype=float)
        
        # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        # CRITICAL: Use * operator for sparse matrix multiplication (not @)
        L = sp.eye(num_nodes) - N * A * N
        
        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        
        # Pad if needed
        if EigVec.shape[1] < self.pos_enc_dim + 1:
            PadVec = np.zeros((EigVec.shape[0], self.pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
            EigVec = np.concatenate((EigVec, PadVec), 1)
        
        # Return positional encoding (skip first eigenvector which is constant)
        lap_pos_enc = torch.from_numpy(EigVec[:, 1:self.pos_enc_dim + 1]).float()
        
        return lap_pos_enc
    
    def convert(self, smiles):
        """
        Convert SMILES string to PyG Data object
        Replicates smiles_to_graph() from DGL implementation
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        
        # Add nodes
        num_atoms = mol.GetNumAtoms()
        
        # Get atom features
        atom_feats = np.array([self.atom_features(a) for a in mol.GetAtoms()])
        
        if self.use_chirality:
            chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                      useLegacyImplementation=False)
            chiral_arr = np.zeros([num_atoms, 3])
            for (i, rs) in chiralcenters:
                if rs == 'R':
                    chiral_arr[i, 0] = 1
                elif rs == 'S':
                    chiral_arr[i, 1] = 1
                else:
                    chiral_arr[i, 2] = 1
            atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)
        
        x = torch.tensor(atom_feats, dtype=torch.float32)
        
        # Add edges
        src_list = []
        dst_list = []
        bond_feats_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_feats = self.bond_features(bond)
            src_list.extend([u, v])
            dst_list.extend([v, u])
            bond_feats_all.append(bond_feats)
            bond_feats_all.append(bond_feats)
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(bond_feats_all), dtype=torch.float32)
        
        # Compute Laplacian positional encoding
        laplacian_pe = self.laplacian_positional_encoding(edge_index, num_atoms)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            laplacian_pe=laplacian_pe,
            smiles=smiles
        )
        
        return data

    def batch_convert_and_save(self, ids, smiles_list, dir_output):
        """
        Creates and saves individual PyG compound graphs
        
        Args:
            ids: Array of compound IDs
            smiles_list: Array of SMILES strings
            dir_output: Output directory path
        """
        
        N = len(smiles_list)
        
        # Create directory for saving graphs
        os.makedirs(dir_output, exist_ok=True)
        
        for no, compound_id in enumerate(ids):
            print(f'{no + 1}/{N}')
            
            smiles_data = smiles_list[no]
            
            # Convert SMILES to PyG Data object
            compound_graph = self.convert(smiles_data)
            
            # Save using torch.save (PyG format)
            save_path = os.path.join(dir_output, f'{compound_id}.pt')
            torch.save(compound_graph, save_path)
            


if __name__ == '__main__':
    import argparse
    from dta.config import DRUGS_DIR

    ap = argparse.ArgumentParser(description='Build drug graphs from a compound mapping CSV')
    ap.add_argument('--mapping', default='data/Davis/Davis_compound_mapping.csv')
    ap.add_argument('--out', default=DRUGS_DIR)
    args = ap.parse_args()
    df = pd.read_csv(args.mapping)
    SMILESToPyG().batch_convert_and_save(df['COMPOUND_ID'].values, df['COMPOUND_SMILES'].values, args.out)
