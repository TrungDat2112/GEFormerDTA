import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def pad_attn_bias_unsqueeze(tensor, max_d_node):
    current_size = tensor.size(0)
    pad_size = max_d_node - current_size
    padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size), value=float('-inf'))
    padded_tensor = padded_tensor.unsqueeze(0)
    
    return padded_tensor

def pad_spatial_pos_unsqueeze(tensor, max_d_node):
 
    current_size = tensor.size(0)
    pad_size = max_d_node - current_size
    padded_tensor = F.pad(tensor, (0, pad_size, 0, pad_size), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)
    
    return padded_tensor

def pad_1d_unsqueeze(tensor, max_d_node):
    current_size = tensor.size(0)
    pad_size = max_d_node - current_size
    padded_tensor = F.pad(tensor, (0, pad_size), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)

    return padded_tensor

def pad_4d_unsqueeze(tensor, max_dim1, max_dim2, max_drug_dist):
    dim1, dim2, current_dist, _ = tensor.size()
    pad_dim1 = max_dim1 - dim1
    pad_dim2 = max_dim2 - dim2
    pad_dist = max_drug_dist - current_dist
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_dist, 0, pad_dim2, 0, pad_dim1), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)

    return padded_tensor

def pad_2d_unsqueeze(tensor, max_dim):
 
    current_dim = tensor.size(0)
    pad_size = max_dim - current_dim
    padded_tensor = F.pad(tensor, (0, 0, 0, pad_size), value=0)
    padded_tensor = padded_tensor.unsqueeze(0)

    return padded_tensor


def bond_to_feature_vector(bond):
    bond_type = bond.GetBondTypeAsDouble()
    bond_stereo = bond.GetStereo()
    is_conjugated = bond.GetIsConjugated()
    features = np.array([
        bond_type,
        bond_stereo,
        is_conjugated
    ], dtype=np.float32)

    return features

def atom_features(atom):
    atom_type = atom.GetAtomicNum()
    chirality = atom.GetChiralTag()
    num_bonds = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    num_hydrogens = atom.GetTotalNumHs()
    num_radical_electrons = atom.GetNumRadicalElectrons()
    hybridization = atom.GetHybridization()
    is_aromatic = atom.GetIsAromatic()
    is_in_ring = atom.IsInRing()
    features = np.array([
        atom_type,
        chirality,
        num_bonds,
        formal_charge,
        num_hydrogens,
        num_radical_electrons,
        hybridization,
        is_aromatic,
        is_in_ring
    ], dtype=np.float32)

    return features


def atom_to_feature_vector(atom):

    degree = atom.GetDegree()
    num_hydrogens = atom.GetTotalNumHs()
    atomic_mass = atom.GetMass()
    feature_vector = [degree, num_hydrogens, atomic_mass]
    
    return feature_vector


def floyd_warshall(adj):

    num_nodes = adj.shape[0]
    shortest_path = np.full((num_nodes, num_nodes), np.inf)
    path = np.full((num_nodes, num_nodes), -1)

    for i in range(num_nodes):
        shortest_path[i, i] = 0
        for j in range(num_nodes):
            if adj[i, j]: 
                shortest_path[i, j] = 1
                path[i, j] = j
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if shortest_path[i, j] > shortest_path[i, k] + shortest_path[k, j]:
                    shortest_path[i, j] = shortest_path[i, k] + shortest_path[k, j]
                    path[i, j] = path[i, k]
    shortest_path[shortest_path == np.inf] = 0

    return shortest_path, path

def gen_edge_input(max_dist, path, attn_edge_type):
    num_nodes = path.shape[0]
    edge_input = np.zeros((num_nodes, num_nodes, max_dist, attn_edge_type.shape[-1]), dtype=np.int32)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and path[i, j] != -1: 
                distance = int(path[i, j])
                if distance < max_dist:
                    edge_input[i, j, distance - 1, :] = attn_edge_type[i, j]

    return edge_input

def aa_sas_feature(prot_target):
    sas_features = []
    acc_file_path = f'davis/profile/{prot_target}_PROP/{prot_target}.acc'
    with open(acc_file_path, 'r') as f:
        lines = f.readlines()[3:]  
        for line in lines:
            values = line.strip().split()
            sas_probs = list(map(float, values[3:6])) 
            sas_features.append(sas_probs)

    return np.array(sas_features)

def aa_ss_feature(prot_target):
    ss_features = []

    ss3_file_path = f'davis/profile/{prot_target}_PROP/{prot_target}.ss8'
    
    with open(ss3_file_path, 'r') as f:
        lines = f.readlines()[2:]  
        for line in lines:
            values = line.strip().split()
            ss_probs = list(map(float, values[3:11])) 
            ss_features.append(ss_probs)

    return np.array(ss_features)

class MolEmbedding(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=256):
        super(MolEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)  

    def forward(self, x):
        x = x.float()
        return self.linear(x) 


def mol_to_single_emb(x, embedding_dim=256):

    embedding_layer = MolEmbedding(input_dim=3, embedding_dim=embedding_dim)

    x_embedded = embedding_layer(x) 
    
    return x_embedded

