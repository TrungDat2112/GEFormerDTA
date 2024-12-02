from rdkit import Chem
import networkx as nx
import numpy as np
import torch
from utils import *

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def sdf2graph(smile):
    drug = 'davis/sdf/' + smile + '.sdf'
    mol = Chem.MolFromMolFile(drug)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    num_bond_features = 3  
    if len(mol.GetBonds()) > 0:  
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int32).T
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else: 
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)

    x = torch.from_numpy(x)
    edge_attr = torch.from_numpy(edge_attr)
    edge_index = torch.from_numpy(edge_index)
    return x, edge_attr, edge_index




def drug_embedding(smile, max_node=128):
    x, edge_attr, edge_index = sdf2graph(smile)

    N = x if x.size(0) >= max_node else pad_2d_unsqueeze(x, max_node)

    N = N.size(1)
    x = mol_to_single_emb(x)

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, 256], dtype=torch.float)
    embedded_edge_attr = mol_to_single_emb(edge_attr)

    attn_edge_type[edge_index[0, :], edge_index[1, :]] = embedded_edge_attr + 1

    shortest_path_result, path = floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    max_dist = int(max_dist)
    edge_input = gen_edge_input(max_dist, path, attn_edge_type.detach().numpy())

    edge_input = torch.from_numpy(edge_input).long()
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    attn_bias = torch.zeros(
        [N, N], dtype=torch.float) 
    node = x
    attn_bias = attn_bias
    spatial_pos = spatial_pos
    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    edge_input = edge_input

    return node, attn_bias, spatial_pos, in_degree, out_degree, edge_input


def prot_to_graph(seq, prot_contactmap, prot_target):
    c_size = len(seq)
    eds_seq = []
    for i in range(c_size - 1):
        eds_seq.append([i, i + 1])
    eds_seq = np.array(eds_seq)
    eds_contact = []
    eds_contact = np.array(np.argwhere(prot_contactmap >= 0.5))

    eds_d = []
    for i in range(c_size):
        eds_d.append([i, c_size])

    eds_d = np.array(eds_d)
    eds = np.concatenate((eds_seq, eds_contact, eds_d))

    edges = [tuple(i) for i in eds]
    g = nx.Graph(edges).to_directed()
    features = []
    ss_feat = []
    sas_feat = []
    ss_feat = aa_ss_feature(prot_target)
    sas_feat = aa_sas_feature(prot_target)
    sequence_output = np.load('davis/emb/' + prot_target + '.npz', allow_pickle=True)
    sequence_output = sequence_output[prot_target].reshape(-1, 1)[0][0]['seq'][1:-1, :]
    sequence_output = sequence_output.reshape(sequence_output.shape[0], sequence_output.shape[1])
    for i in range(c_size):
        aa_feat = np.concatenate((np.asarray(sequence_output[i], dtype=float), ss_feat[i], sas_feat[i]))
        features.append(aa_feat)
    place_holder = np.zeros(features[0].shape, dtype=float)
    features.append(place_holder)

    edge_index = []
    edge_weight = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        edge_weight.append(1.0)
    return c_size, features, edge_index, edge_weight


