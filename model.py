from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from utils import *

torch.manual_seed(1)
np.random.seed(1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, outfeature):
        super(ResidualBlock, self).__init__()
        self.outfeature = outfeature
        self.gcn = GCNConv(outfeature,outfeature)
        self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.gcn.lin.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, edge_index):
        identity = x
        out = self.gcn(x, edge_index)
        out = self.relu(out)
        out = self.ln(out)
        out += identity
        out = self.relu(out)
        return out

class GEFormerDTA(torch.nn.Module):
    def __init__(self, num_features_xd, num_features_xt,
                     latent_dim=64, dropout=0.2, n_output=1, device='cpu', **kwargs):
        super(GEFormerDTA, self).__init__()

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(0.5)
        self.device = device
        self.num_rblock = 4

        self.gpus = torch.cuda.device_count()
        self.num_layers = 3
        self.num_heads = 8
        self.hidden_dim = 256
        self.inter_dim = 16
        self.flatten_dim = 4600
        self.multi_hop_max_dist = 20

        self.encoder_dropout = 0
        self.attention_dropout = 0
        self.input_dropout = nn.Dropout(0)

        self.d_node_encoder = nn.Embedding(512 * 9 + 1, self.hidden_dim, padding_idx=0)
        self.d_edge_encoder = nn.Embedding(512 * 3 + 1, self.num_heads, padding_idx=0)
        self.d_edge_dis_encoder = nn.Embedding(128 * self.num_heads * self.num_heads, 1)
        self.d_spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.d_in_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.d_out_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)

        self.d_encoders = drugEncoder(hidden_dim=self.hidden_dim, inter_dim=self.inter_dim,
                                      n_layers=self.num_layers, n_heads=self.num_heads)

        self.d_final_ln = nn.LayerNorm(self.hidden_dim)
        self.d_graph_token = nn.Embedding(1, self.hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, 8)
        self.icnn = nn.Conv1d(self.hidden_dim, 8, (3,))

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),

            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(True),

            nn.BatchNorm1d(256),
            nn.Linear(256, 86)

        )
        self.conv1_xd = GCNConv(num_features_xd, num_features_xd)
        self.conv2_xd = GCNConv(num_features_xd, num_features_xd * 2)
        self.rblock_xd = ResidualBlock(num_features_xd*2)
        self.fc_g1_d = torch.nn.Linear(num_features_xd*2, 1024)
        self.fc_g2_d = torch.nn.Linear(1024, num_features_xt)
        self.fc_g3_d = torch.nn.Linear(num_features_xt, latent_dim * 2)

        self.first_linear = torch.nn.Linear(num_features_xt, num_features_xt)
        self.second_linear = torch.nn.Linear(num_features_xt, 1)

        self.conv1_xt = GCNConv(num_features_xt, latent_dim)
        self.conv2_xt = GCNConv(latent_dim, latent_dim * 2)
        self.rblock_xt = ResidualBlock(latent_dim * 2)
        self.fc_g1_t = torch.nn.Linear(latent_dim * 2, 1024)
        self.fc_g2_t = torch.nn.Linear(1024, latent_dim * 2)

        self.fc1 = nn.Linear(4 * latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, drug, prot, max_d_node=256, multi_hop_max_dist=20, spatial_pos_max=20):
        drug_node, drug_attn_bias, drug_spatial_pos, drug_in_degree, drug_out_degree, drug_edge_input = [], [], [], [], [], []
        x, edge_index, batch, d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug.x, drug.edge_index, drug.batch, drug.node, drug.attn_bias, drug.spatial_pos, drug.in_degree, drug.out_degree, drug.edge_input
        x2, edge_index2, batch2, prot_lens, edge_attr2 = prot.x, prot.edge_index, prot.batch, prot.prot_len, prot.edge_attr

        if d_node.size(1) <= max_d_node:
            drug_node.append(d_node)
            d_attn_bias[:, :][d_spatial_pos >= spatial_pos_max] = float('-inf')
            drug_attn_bias.append(d_attn_bias)
            drug_spatial_pos.append(d_spatial_pos)
            drug_in_degree.append(d_in_degree)
            drug_out_degree.append(d_out_degree)
            drug_edge_input.append(d_edge_input[:, :, :multi_hop_max_dist, :])
        max_drug_dist = max(i.size(-2) for i in drug_edge_input)

        drug_node = torch.cat([pad_2d_unsqueeze(i, max_d_node) for i in drug_node])

        drug_attn_bias = torch.cat([pad_attn_bias_unsqueeze(
            i, max_d_node) for i in drug_attn_bias])

        drug_spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_d_node)
                                      for i in drug_spatial_pos])

        drug_in_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node)
                                    for i in drug_in_degree])


        drug_out_degree = torch.cat([pad_1d_unsqueeze(i, max_d_node)
                                     for i in drug_out_degree])

        drug_edge_input = torch.cat([pad_4d_unsqueeze(
            i, max_d_node, max_d_node, max_drug_dist) for i in drug_edge_input])

        drug_n_node, drug_n_graph = drug_node.squeeze(0).size()[:2] 

        drug_graph_attn_bias = drug_attn_bias.clone() 
        drug_graph_attn_bias = drug_graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        drug_spatial_pos_bias = self.d_spatial_pos_encoder(drug_spatial_pos).permute(0, 3, 1, 2)  # 4 8 256 256
        drug_graph_attn_bias = drug_graph_attn_bias + drug_spatial_pos_bias  # 4 8 256 256

        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)  # 1 8 1
        drug_graph_attn_bias[:, :, 1:, 0] = drug_graph_attn_bias[:, :, 1:, 0] + t  # 4 8 256
        drug_graph_attn_bias[:, :, 0, :] = drug_graph_attn_bias[:, :, 0, :] + t  # 4 8 257

        # edge_input
        drug_spatial_pos = drug_spatial_pos.clone()
        drug_spatial_pos[drug_spatial_pos == 0] = 1  # set pad to 1

        drug_spatial_pos = torch.where(drug_spatial_pos > 1, drug_spatial_pos - 1, drug_spatial_pos)

        drug_spatial_pos = drug_spatial_pos.clamp(0, self.multi_hop_max_dist)
        drug_edge_input = drug_edge_input[:, :, :, :self.multi_hop_max_dist, :]

        drug_edge_input = self.d_edge_encoder(drug_edge_input).mean(-2)
        max_dist = drug_edge_input.size(-2)
        drug_edge_input_flat = drug_edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug_edge_input_flat = torch.bmm(drug_edge_input_flat,
                                         self.d_edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[
                                         :max_dist, :, :])

        drug_edge_input = (drug_edge_input.sum(-2) / (drug_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1,
                                                                                                       2)  

        drug_graph_attn_bias = drug_graph_attn_bias + drug_edge_input
        drug_graph_attn_bias = drug_graph_attn_bias + drug_attn_bias.unsqueeze(1) 

        drug_node_feature = self.d_node_encoder(drug_node).sum(dim=-2)  # 4 256 64
        drug_node_feature = pad_2d_unsqueeze(drug_node_feature.squeeze(0), self.hidden_dim*self.num_heads*4)

        drug_node_feature = drug_node_feature + self.d_in_degree_encoder(drug_in_degree.long()) + self.d_out_degree_encoder(drug_out_degree.long())
        drug_graph_token_feature = self.d_graph_token.weight.unsqueeze(0).repeat(drug_n_graph, 1, 1)
        drug_graph_node_feature = torch.cat([drug_graph_token_feature.permute(1,0,2), drug_node_feature], dim=1)

        # transfomrer encoder
        drug_output = self.input_dropout(drug_graph_node_feature)  # 4 257 64
        drug_output = self.d_encoders(drug_output, drug_graph_attn_bias)  # 4 65 64

        x = self.conv1_xd(x, edge_index)
        x = self.relu(x)
        x = self.conv2_xd(x, edge_index)
        x = self.relu(x)
        for i in range(self.num_rblock):
            x = self.rblock_xd(x, edge_index)
        x = gmp(x, batch)      
        x = self.relu(self.fc_g1_d(x))
        x = self.dropout(x)
        x = self.fc_g2_d(x)
        x = self.dropout(x)
        x_changedim = self.relu(self.fc_g3_d(x))

        dense_node, bool_node = to_dense_batch(x2, batch2) 
        cur_idx = -1
        cur_batch = 0
        mask = torch.ones(batch2.size(0), dtype=torch.bool)
        for size in prot_lens:
            batch_dense_node = dense_node[cur_batch]
            masked_batch_dense_node = batch_dense_node[bool_node[cur_batch]][:-1]
            node_att = F.tanh(self.first_linear(masked_batch_dense_node))
            node_att = self.dropout1(node_att)
            node_att = self.second_linear(node_att)
            node_att = self.dropout1(node_att)
            node_att = node_att.squeeze()
            node_att = F.softmax(node_att, 0)
            cur_idx += size+1
            idx_target = (edge_index2[0] == cur_idx).nonzero()
            edge_attr2[idx_target.squeeze()] = node_att.squeeze()
            idx_target = (edge_index2[1] == cur_idx).nonzero()
            edge_attr2[idx_target.squeeze()] = node_att.squeeze()
            x2[cur_idx] = x[cur_batch]
            mask[cur_idx] = False
            cur_batch += 1
        mask_drug = ~mask
        x2 = self.conv1_xt(x2, edge_index2, edge_attr2)
        x2 = self.relu(x2)
        x2 = self.conv2_xt(x2, edge_index2, edge_attr2)
        x2 = self.relu(x2)
        for i in range(self.num_rblock):
            x2 = self.rblock_xt(x2, edge_index2)
        x2_nodrug = x2[mask]
        batch2_nodrug = batch2[mask]
        drug_after = x2[mask_drug]
        x2 = gmp(x2_nodrug, batch2_nodrug)
        x2 = self.relu(self.fc_g1_t(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2_t(x2)
        x2 = self.dropout(x2)

        x = x_changedim.unsqueeze(2)
        drug_after = drug_after.unsqueeze(2)
        x2 = x2.unsqueeze(-1) if x2.shape[1] == x2.shape[0] else pad_2d_unsqueeze(x2, drug_output.shape[0]*drug_output.shape[1]).reshape(-1, drug_output.shape[0], drug_output.shape[0]).permute(2,1,0)
        x = x if x.shape[1] == x.shape[0] else pad_2d_unsqueeze(x.squeeze(-1), drug_output.shape[0]*drug_output.shape[1]).reshape(-1, drug_output.shape[0], drug_output.shape[0]).permute(2,1,0)
        drug_after = drug_after if drug_after.shape[1] == drug_after.shape[0] else pad_2d_unsqueeze(drug_after.squeeze(-1), drug_output.shape[0]*drug_output.shape[1]).reshape(-1, drug_output.shape[0], drug_output.shape[0]).permute(2,1,0)
        x = torch.cat((x, drug_after, drug_output), 2)
        x = torch.max_pool1d(x, 2, 1)
        x = x.squeeze(2)
        xc = torch.cat((x, x2), 2)[:,:,:-1]
        xc = xc if xc.shape[2] == self.hidden_dim else xc[:,:,:self.hidden_dim]
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class drugEncoder(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_layers, n_heads, dropout=0.0):
        super(drugEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(
            Encoder_layer(hidden_dim, inter_dim, n_heads, dropout) for l in range(n_layers))
        self.conv_layers = nn.ModuleList(Distilling_layer(hidden_dim) for _ in range(n_layers - 1))
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
        attns.append(attn)
        x = self.norm(x)
        return x

class Encoder_layer(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_heads, dropout):
        super(Encoder_layer, self).__init__()
        self.attention = AttentionLayer(hidden_dim=hidden_dim, n_heads=n_heads)
        self.conv1 = nn.Conv1d(hidden_dim, inter_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(inter_dim, hidden_dim, kernel_size=(1,))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = F.relu

    def forward(self, x, attn_mask=None):
        attn_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_x)
        y = x = self.norm1(x)
        y = self.dropout(self.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionLayer, self).__init__()

        key_dim = hidden_dim // n_heads
        value_dim = hidden_dim // n_heads

        self.inner_attention = ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False)
        self.query_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.key_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.value_projection = nn.Linear(hidden_dim, value_dim * n_heads)
        self.out_projection = nn.Linear(value_dim * n_heads, hidden_dim)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  

        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :] 
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) 

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:

            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V)  
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) 

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn

class Distilling_layer(nn.Module):
    def __init__(self, channel):
        super(Distilling_layer, self).__init__()

        self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=(3,), padding=1,
                              padding_mode='circular')
        self.norm = nn.BatchNorm1d(channel)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        out = self.maxPool(self.activation(self.norm(x))).transpose(1, 2)

        return out

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask