"""AlphaFold2 structure -> binding-pocket residue graph (41-d residue features, 5-d contact features).

Pipeline: PDB -> residue features + 10 A contact edges -> DeepChem ConvexHull pocket -> pocket
subgraph -> Laplacian PE. The pocket residue indices are also used to slice ESM-2 embeddings.
"""
import os
from itertools import permutations

import numpy as np
import torch
from MDAnalysis.analysis import distances
from rdkit import Chem
from scipy import sparse as sp
from scipy.spatial import distance_matrix
from torch_geometric.utils import subgraph

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]
    
def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]
def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
                     obtain_self_dist(res) +  # 5
                     obtain_dihediral_angles(res)  # 4
                     )

def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname
    
def calc_dist(res1, res2):

    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array

def obatin_edge(u, cutoff=10.0):
    """
    Constructs edges between residues within cutoff distance.
    
    Returns:
        edge_index: List of [source, target] pairs
        edge_distances: Array of [min_dist, max_dist] for each edge
    """
    edgeids = []
    dismin = []
    dismax = []
    
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    
    return edgeids, np.array([dismin, dismax]).T

def check_connect(u, i, j):
    """
    Checks if two residues are covalently bonded (sequential in chain).
    
    Returns:
        1 if residues are connected, 0 otherwise
    """
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0
        
    
def obtain_ca_pos(res):
    """
    Extracts C-alpha position or center of mass for metal ions.
    
    Returns:
        3D coordinates of the residue representative position
    """
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  # Some residues may lack CA atoms
            return res.atoms.positions.mean(axis=0)
def laplacian_pe(edge_index, num_nodes, pos_enc_dim=8):
    """Eigenvectors of the normalised Laplacian (first, trivial one dropped)."""
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1
    A = sp.csr_matrix(adj.numpy())
    degrees = np.clip(np.asarray(adj.sum(axis=1)).flatten(), 1, None)
    D_inv_sqrt = sp.diags(degrees ** -0.5)
    L = sp.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    EigVal, EigVec = np.linalg.eig(L.toarray())
    EigVec = np.real(EigVec[:, EigVal.argsort()])
    if EigVec.shape[1] < pos_enc_dim + 1:
        EigVec = np.concatenate((EigVec, np.zeros((num_nodes, pos_enc_dim + 1 - EigVec.shape[1]))), axis=1)
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()


def find_pocket_residues(prot_pdb, ca_pos):
    """Residues whose C-alpha lies inside the top DeepChem ConvexHull pocket box."""
    import deepchem
    pockets = deepchem.dock.ConvexHullPocketFinder().find_pockets(prot_pdb)
    if len(pockets) == 0:
        return []
    box = pockets[0]
    (x0, x1), (y0, y1), (z0, z1) = box.x_range, box.y_range, box.z_range
    return [i for i, (x, y, z) in enumerate(ca_pos) if x0 < x < x1 and y0 < y < y1 and z0 < z < z1]


def prot_to_graph(prot_pdb, cutoff=10.0, pos_enc_dim=8):
    """Returns (graph_dict, pocket_idx). graph_dict has x, edge_index, edge_attr, laplacian_pe."""
    import MDAnalysis as mda

    prot = Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    Chem.AssignStereochemistryFrom3D(prot)
    u = mda.Universe(prot)

    x = torch.tensor(np.array([calc_res_features(res) for res in u.residues]), dtype=torch.float)

    edgeids, distm = obatin_edge(u, cutoff)
    src, dst = zip(*edgeids)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    ca_pos = np.array([obtain_ca_pos(res) for res in u.residues])
    center_pos = u.atoms.center_of_mass(compound='residues')
    dis_ca, dis_center = distance_matrix(ca_pos, ca_pos), distance_matrix(center_pos, center_pos)
    edge_attr = torch.cat([
        torch.tensor([check_connect(u, i, j) for i, j in edgeids], dtype=torch.float).view(-1, 1),
        torch.tensor([dis_ca[i, j] for i, j in edgeids], dtype=torch.float).view(-1, 1) * 0.1,
        torch.tensor([dis_center[i, j] for i, j in edgeids], dtype=torch.float).view(-1, 1) * 0.1,
        torch.tensor(distm, dtype=torch.float),
    ], dim=1)

    pocket_idx = find_pocket_residues(prot_pdb, ca_pos) or list(range(x.size(0)))
    subset = torch.tensor(pocket_idx, dtype=torch.long)
    edge_index, edge_attr = subgraph(subset, edge_index, edge_attr, relabel_nodes=True, num_nodes=x.size(0))
    x = x[subset]
    graph = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
             'laplacian_pe': laplacian_pe(edge_index, x.size(0), pos_enc_dim)}
    return graph, pocket_idx


if __name__ == '__main__':
    import argparse
    import pandas as pd
    from dta.config import PROTEINS_DIR, DATA_DIR

    ap = argparse.ArgumentParser(description='Build pocket graphs from AlphaFold2 PDBs')
    ap.add_argument('--mapping', default=os.path.join(DATA_DIR, 'Davis_protein_mapping.csv'))
    ap.add_argument('--pdb_dir', default=os.path.join(DATA_DIR, 'PDB_AF2'))
    ap.add_argument('--out', default=PROTEINS_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pocket_index = {}
    for pid in pd.read_csv(args.mapping)['PROTEIN_ID']:
        pdb = os.path.join(args.pdb_dir, f'{pid}.pdb')
        if not os.path.exists(pdb):
            print(f'skip {pid}: no structure'); continue
        graph, pocket_index[pid] = prot_to_graph(pdb)
        torch.save(graph, os.path.join(args.out, f'{pid}.pt'))
        print(f'{pid}: {graph["x"].size(0)} pocket residues')
    torch.save(pocket_index, os.path.join(args.out, '..', 'pocket_index.pt'))
