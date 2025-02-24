from torch.nn import Sequential, Linear, ReLU
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from dgl.nn import EdgeWeightNorm
import random
import os
import torch
import torch.nn as nn
import pandas as pd
import dgl.function as fn
from tqdm import tqdm
from utils import *
import torch
torch.set_printoptions(threshold=np.inf)
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
EOS = 1e-10
norm = EdgeWeightNorm(norm='both')
import cupy as cp

from dgl.nn.pytorch.conv import GraphConv
class GCL(nn.Module):
    def __init__(self, nlayers, nlayers_proj, in_dim, emb_dim, proj_dim, dropout, sparse, batch_size):
        super(GCL, self).__init__()
        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)
        if nlayers_proj == 1:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim))
        elif nlayers_proj == 2:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
        self.batch_size = batch_size


    def get_embedding(self, x, a1, a2, source='all'):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        return torch.cat((emb1, emb2), dim=1)


    def get_projection(self, x, a1, a2):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        return torch.cat((proj1, proj2), dim=1)


    def forward(self, x1, a1, x2, a2):
        emb1 = self.encoder1(x1, a1)
        emb2 = self.encoder2(x2, a2)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        loss = self.batch_nce_loss(proj1, proj2)
        return loss


    def set_mask_knn(self, X, k, dataset, metric='cosine'):
        if k != 0:
            path = '../data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)
            if os.path.exists(file_name):
                knn = sparse.load_npz(file_name)
                # print('Load exist knn graph.')
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric)
                sparse.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))
            knn = torch.tensor(knn.toarray()) + torch.eye(X.shape[0])
        else:
            knn = torch.eye(X.shape[0])
        self.pos_mask = knn
        self.neg_mask = 1 - self.pos_mask


    def batch_nce_loss(self, z1, z2, temperature=0.2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask, temperature)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask, temperature)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:], temperature)
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss


    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        pos_mask = pos_mask.cuda()
        neg_mask = neg_mask.cuda()
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()


class lambda_2(nn.Module):
    def __init__(self):
        super(lambda_2, self).__init__()
        self.lbd = nn.Parameter(torch.Tensor(1))
        self.reset_parameter()

    
    def reset_parameter(self):
        self.lbd.data.fill_(1e-3)

    
    def forward(self, N_i, M_i):
        Lbd = F.relu(self.lbd)
        obj = F.relu((N_i + Lbd) / M_i)
        return obj.sum()


class Edge_Discriminator(nn.Module):
    def __init__(self, nnodes, adj, adj_two_order, input_dim, alpha, sparse, dataset_name, hidden_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()
        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)
        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.adj = adj
        self.adj_two_order = adj_two_order
        self.sparse = sparse
        self.alpha = alpha
        self.dataset_name = dataset_name

    
    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h

    
    def get_adj_2_edges(self, adj, adj_two_order, alpha=0.5):
        adj_1 = adj
        adj_combined = (1 - alpha) * adj_1 + alpha * adj_two_order
        second_order_edges_indices = torch.nonzero(adj_combined).t()
        edges_2 = torch.zeros((2, len(second_order_edges_indices[0])), dtype=torch.long).cuda()
        edges_2[:, :second_order_edges_indices.shape[1]] = second_order_edges_indices
        return edges_2



    def get_edge_weight(self, embeddings, edges):
        s1 = torch.sum(embeddings[edges[0]] * embeddings[edges[1]], dim=1)
        s2 = torch.sum(embeddings[edges[1]] * embeddings[edges[0]], dim=1)
        return (s1 + s2) / 2


    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()


    def merge_edge_weights(self, edges_1_index, edges_1_weights_raw, edges_2_index, edges_2_weights_raw, alpha=0.5):
        edges_1_index_np = edges_1_index.cpu().numpy()
        edges_1_weights_np = edges_1_weights_raw.cpu().detach().numpy()
        edges_2_index_np = edges_2_index.cpu().numpy()
        edges_2_weights_np = edges_2_weights_raw.cpu().detach().numpy()
        merged_weights_dict = {}

        for edge_index, edge_weights, weight_factor in [
            (edges_1_index_np, edges_1_weights_np, alpha),
            (edges_2_index_np, edges_2_weights_np, 1 - alpha)]:
            rows, cols = edge_index[0], edge_index[1]

            for row, col, weight in zip(rows, cols, edge_weights):
                key = (row, col)
                merged_weights_dict[key] = merged_weights_dict.get(key, 0) + weight #* weight_factor
                
        merge_edge_index = torch.tensor(list(merged_weights_dict.keys()), dtype=torch.long).t().cuda()
        merged_edge_weights = torch.tensor(list(merged_weights_dict.values()), dtype=torch.float).cuda()
        return merged_edge_weights

    
    def weight_forward(self, features, adj, adj_two_order):
        embeddings = self.get_node_embedding(features)
        edges_2 = self.get_adj_2_edges(adj, adj_two_order)
        edges_1_index, edges_1_attr = from_scipy_sparse_matrix(csr_matrix(adj))
        edges_2_index, edges_2_attr = from_scipy_sparse_matrix(csr_matrix(adj_two_order))
        edges_1_weights_raw = self.get_edge_weight(embeddings, edges_1_index).cuda()
        edges_2_weights_raw = self.get_edge_weight(embeddings, edges_2_index).cuda()
        edges_weights_raw = self.merge_edge_weights(edges_1_index, edges_1_weights_raw, edges_2_index, edges_2_weights_raw)
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp, edges_2

    
    def weight_to_adj(self, edges, weights_lp, weights_hp):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, weights_lp, self.nnodes)
            adj_lp += torch.eye(self.nnodes).cuda()
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)
            adj_hp = get_adj_from_edges(edges, weights_hp, self.nnodes)
            adj_hp += torch.eye(self.nnodes).cuda()
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)
            mask = torch.zeros(adj_lp.shape).cuda()
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes).cuda() - adj_hp * mask * self.alpha
        else:
            adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_lp = dgl.add_self_loop(adj_lp)
            weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes).cuda())) + EOS
            weights_lp = norm(adj_lp, weights_lp)
            adj_lp.edata['w'] = weights_lp
            adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_hp = dgl.add_self_loop(adj_hp)
            weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes).cuda())) + EOS
            weights_hp = norm(adj_hp, weights_hp)
            weights_hp *= - self.alpha
            weights_hp[edges.shape[1]:] = 1
            adj_hp.edata['w'] = weights_hp
        return adj_lp, adj_hp


    def forward(self, features, edges, adj, adj_two_order):
        weights_lp, weights_hp, edges_2 = self.weight_forward(features, adj, adj_two_order)
        adj_lp, adj_hp = self.weight_to_adj(edges_2, weights_lp, weights_hp)
        return adj_lp, adj_hp, weights_lp, weights_hp, edges_2



class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

    
    def forward(self, x, g):
        x = torch.relu(self.linear(x))

        if self.sparse:
            with g.local_scope():
                g.ndata['h'] = x
                for _ in range(self.k):
                    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                return g.ndata['h']
        else:
            for _ in range(self.k):
                x = torch.matmul(g, x)
            return x

