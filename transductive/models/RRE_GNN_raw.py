"""
anather implementation of RRE-GNN 
"""
import torch
import torch.nn as nn
from torch_scatter import scatter
from .QRGRU import GateUnit

class RRE_GNN_raw(torch.nn.Module):
    def __init__(self, params, loader):
        super(RRE_GNN_raw, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(
                IdentityLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.n_extra_layer = params.n_extra_layer
        self.extra_gnn_layers = []
        for i in range(self.n_extra_layer):
            self.extra_gnn_layers.append(
                IdentityLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.extra_gnn_layers = nn.ModuleList(self.extra_gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(
            self.hidden_dim, 1, bias=False)  
        self.gate = GateUnit(self.hidden_dim, self.hidden_dim)

    def forward(self, subs, rels, mode='train'):
        n = len(subs)
        if torch.cuda.is_available():
            q_sub = torch.LongTensor(subs).cuda()
            q_rel = torch.LongTensor(rels).cuda()
            h0 = torch.zeros((n, self.hidden_dim)).cuda()
            nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
            hidden = torch.zeros(n, self.hidden_dim).cuda()
        else:
            q_sub = torch.LongTensor(subs)
            q_rel = torch.LongTensor(rels)
            h0 = torch.zeros((n, self.hidden_dim))
            nodes = torch.cat([torch.arange(n).unsqueeze(1), q_sub.unsqueeze(1)], 1)
            hidden = torch.zeros(n, self.hidden_dim)

        scores_all = []
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            hidden, h_n_qr = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            if torch.cuda.is_available():
                h0 = torch.zeros(nodes.size(0), hidden.size(1)).cuda().index_copy_(0, old_nodes_new_idx, h0)
            else:
                h0 = torch.zeros(nodes.size(0), hidden.size(1)).index_copy_(0, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden = self.gate(hidden, h_n_qr, h0)
            h0 = hidden
        for i in range(self.n_extra_layer):
            
            hidden = hidden[old_nodes_new_idx]
            hidden, h_n_qr = self.extra_gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            hidden = self.dropout(hidden)
            hidden = self.gate(hidden, h_n_qr, h0)
            h0 = hidden

        scores = self.W_final(hidden).squeeze(-1)
        if torch.cuda.is_available():
            scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        else:
            scores_all = torch.zeros((n, self.loader.n_ent))
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all


class IdentityLayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(IdentityLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.relu = nn.ReLU()
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.gate = GateUnit(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        unique_edges, inverse_indices = torch.unique(edges[:, [0, 2, 4]],dim=0, sorted=True, return_inverse=True)
        sub = unique_edges[:, 2]
        rel = unique_edges[:, 1]
        obj = edges[:, 5]
        h_s = hidden[sub]
        h_r = self.rela_embed(rel)
        r_idx = unique_edges[:, 0]

        h_qr = self.rela_embed(q_rel)[r_idx]

        node_group = torch.zeros(n_node, dtype=torch.int64).to(h_qr.device)
        node_group[edges[:, 5]] = edges[:, 0]  
        h_n_qr = self.rela_embed(q_rel)[node_group]  

        unique_relation_identity = self.gate(h_r, h_qr, h_s)

        unique_message = unique_relation_identity
        unique_attend_weight = self.w_alpha(self.relu(self.Ws_attn(unique_message) + self.Wqr_attn(h_qr)))

        unique_exp_attend = torch.exp(unique_attend_weight)
        exp_attend = unique_exp_attend[inverse_indices]
        unique_message = unique_exp_attend * unique_message
        message = unique_message[inverse_indices]
        sum_exp_attend = scatter(exp_attend, dim=0, index=obj, dim_size=n_node, reduce="sum")
        no_attend_message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = no_attend_message_agg / sum_exp_attend

        hidden_new = self.act(self.W_h(message_agg))
        return hidden_new, h_n_qr

