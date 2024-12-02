import copy
import os
import pickle
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric import data as DATA
from tqdm import tqdm
from utils import *
from collate import *
from dataset import *
from data_process import prot_to_graph, smile_to_graph, drug_embedding
from metrics import *
from model import GEFormerDTA


torch.manual_seed(2)
np.random.seed(3)

num_feat_xp = 0
num_feat_xd = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
run_model = 6
cuda = 0
LR = 0.0005
LOG_INTERVAL = 20

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):

        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

        drug = data[0].to(device)
        prot = data[1].to(device)
        optimizer.zero_grad()
        output = model(drug, prot)
        affinity = drug.y.view(-1, 1).float()
        affinity = affinity if affinity.shape[0] == output.shape[1] else pad_2d_unsqueeze(affinity, output.shape[1]).squeeze(0)
        loss = loss_fn(output, affinity.to(device))
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    print('Average loss: {:.4f}'.format(total_train_loss / (batch_idx + 1)))
    return total_train_loss / (batch_idx + 1)


def adjust_learning_rate(optimizer, LR, scale=0.7):
    """Sets the learning rate to the initial LR decayed by 10 every interval epochs"""
    lr = LR * scale

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

compound_iso_smiles = []
pdbs = []
pdbs_seqs = []
all_labels = []

df = pd.read_csv('davis/split/trainer.csv')
compound_iso_smiles += list(df['compound_iso_smiles'])
pdbs += list(df['target_name'])
pdbs_seqs += list(df['target_sequence'])
all_labels += list(df['affinity'])
pdbs_tseqs = set(zip(pdbs, pdbs_seqs, compound_iso_smiles, all_labels))

dta_graph = {}
print('Pre-processing protein')
print('Pre-processing...')
saved_prot_graph = {}
if os.path.isfile('saved_prot_graph.pickle'):
    print("Load pre-processed file for protein graph")
    with open('saved_prot_graph.pickle', 'rb') as handle:
        saved_prot_graph = pickle.load(handle)
else:
    for target, seq in set(zip(pdbs, pdbs_seqs)):
        if os.path.isfile('davis/map/' + target + '.npy'):
            contactmap = np.load('davis/map/' + target + '.npy')
        else:
            raise FileNotFoundError
        c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target)
        g = DATA.Data(
            x = torch.tensor(np.array(features), dtype=torch.float32),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            edge_attr=torch.FloatTensor(edge_weight),
            prot_len=c_size
        )
        saved_prot_graph[target] = g
    with open('saved_prot_graph.pickle', 'wb') as handle:
        pickle.dump(saved_prot_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
saved_drug_graph = {}

for smiles in compound_iso_smiles:
    c_size2, features2, edge_index2 = smile_to_graph(smiles)
    d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug_embedding(smiles)
    g2 = DATA.Data(
        x=torch.tensor(np.array(features2)),
        edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
        node=d_node,
        attn_bias=d_attn_bias,
        spatial_pos=d_spatial_pos,
        in_degree=d_in_degree,
        out_degree=d_out_degree,
        edge_input=d_edge_input
        )
    saved_drug_graph[smiles] = g2
with open('saved_drug_graph.pickle', 'wb') as handle:
    pickle.dump(saved_drug_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('load pre-processed file for protein graph and saved drug graph pickle file success!!!!!!!!!!!')

for i in tqdm(pdbs_tseqs):
    g = copy.copy(saved_prot_graph[i[0]])
    g2 = copy.copy(saved_drug_graph[i[2]])
    g.y = torch.FloatTensor([i[3]])
    g2.y = torch.FloatTensor([i[3]])
    dta_graph[(i[0], i[2])] = [g, g2]
    num_feat_xp = g.x.size()[1]
    num_feat_xd = g2.x.size()[1]
pd.DataFrame(dta_graph).to_csv('./dta_graph.csv', index=False, index_label=False)

df = pd.read_csv('davis/split/trainer.csv')
train_drugs, train_prots, train_prots_seq, train_Y = list(df['compound_iso_smiles']), list(df['target_name']), list(
    df['target_sequence']), list(df['affinity'])
train_drugs, train_prots, train_prots_seq, train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(
    train_prots_seq), np.asarray(train_Y)


train_data = GraphPairDataset(smile_list=train_drugs, dta_graph=dta_graph, prot_list=train_prots)

train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate,
                          num_workers=0, pin_memory=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GEFormerDTA(num_features_xd=num_feat_xd, num_features_xt=num_feat_xp,device=device).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_ci = 0
best_epoch = -1

train(model,device=device,train_loader=train_loader,optimizer=optimizer,epoch=NUM_EPOCHS)