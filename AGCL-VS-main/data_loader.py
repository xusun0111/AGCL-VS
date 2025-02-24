import warnings
import torch
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import numba
from tqdm import tqdm
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import remove_self_loops
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_scipy_sparse_matrix

warnings.simplefilter("ignore")


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, num_splits: int = 10):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)
        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True
        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True
        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True
        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)
    return train_mask_all, val_mask_all, test_mask_all


def split_masks(train_mask, val_mask, test_mask):
    train_masks = [train_mask[i] for i in range(train_mask.size(0))]
    val_masks = [val_mask[i] for i in range(val_mask.size(0))]
    test_masks = [test_mask[i] for i in range(test_mask.size(0))]
    return train_masks, val_masks, test_masks


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def get_structural_encoding(edges, nnodes, str_enc_dim=20):
    row = edges[0, :].numpy()
    col = edges[1, :].numpy()
    data = np.ones_like(row)
    A = sp.csr_matrix((data, (row, col)), shape=(nnodes, nnodes))
    D = sp.diags(np.array(A.sum(1)).flatten())
    L = D - A
    SE = torch.from_numpy(eigenvectors).float()
    return SE


def create_W(features, scale=16):
    features = torch.tensor(features, dtype=torch.float32)
    distances = torch.cdist(features, features, p=2)
    K = distances / scale
    return K.numpy()


def op_tmp(adj, K):
    L = adj.shape[0]
    adj_2 = adj @ adj  
    N = torch.zeros((L, L))
    M = torch.zeros((L, L))

    for i in range(L):
        for j in range(L):
            C_f = adj_2[i, :] - adj[j, :] * adj[i, j] - adj[i, :]
            sum_s_jf_2 = torch.sum(adj[j, :] ** 2) - adj[j, j] ** 2  
            sum_s_jf_Cf = torch.sum(adj[j, :] * C_f) - adj[j, j] * C_f[j]
            s_ij_N = 2 * adj_2[i, j] - K[i, j] - 2 * sum_s_jf_Cf
            s_ij_M = 4 + 2 * sum_s_jf_2
            N[i, j] = s_ij_N
            M[i, j] = s_ij_M
    return N, M

def normalize_matrix(adj, eps=1e-12):
    D = torch.sum(adj, dim=1) + eps
    D = torch.pow(D, -0.5)
    D[torch.isinf(D)] = 0
    D[torch.isnan(D)] = 0
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(D, adj), D)
    return adj

def compute_NM(X, A):
    I = np.eye(A.shape[0])
    A = A.to_dense()
    A = A + I
    A = normalize_matrix(A)
    K = create_W(X) 
    N, M = op_tmp(A, K) 
    return N, M

class lambda_2(nn.Module):
    def __init__(self):
        super(lambda_2, self).__init__()
        self.lbd = nn.Parameter(torch.Tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        self.lbd.data.fill_(1e-3)

    def forward(self, N_i, M_i):
        Lbd = F.relu(self.lbd)
        obj = F.relu((N_i + Lbd)/M_i)
        return obj.sum()

def optimize_lbd2(N_i, M_i, epochs=10000, learning_rate=1e-4, convengence=1e-16):
    N_ii = N_i.cuda()
    M_ii = M_i.cuda()
    model = lambda_2().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_last = torch.zeros(1).cuda()
    for epoch in range(epochs):
        optimizer.zero_grad()
        obj = model(N_ii, M_ii)
        loss = 10 * torch.square(obj - 1)
        if torch.abs_(loss_last - loss) < convengence * loss_last:
            break
        else:
            loss_last = loss.clone()
        loss.backward()
        optimizer.step()
        return F.relu(model.lbd).item()

def op_S( lb2, N, M):
    S = (N.T + lb2) / M.T
    S[S < 0] = 0
    return S.T

def compute_A_bar(N, M):
    lb2 = []
    print('N:', N.t)
    for i in tqdm(range(N.shape[0])):
        lb = optimize_lbd2(N[i], M[i])
        lb2.append(lb)
    lb2 = np.array(lb2)
    S = op_S(lb2, N, M)
    return S

def A_final(S, nbs=50.0):
    S[S < (1 / nbs)] = 0
    S[S >= (1 / nbs)] = 1
    S[S > 0] = 1
    S = S.T + S
    S[S >= 1] = 1
    S[S < 1] = 0
    return S

def homophily_v2(A, labels, ignore_negative=False) :
    A = A.to_dense().cpu().numpy() if A.is_sparse else A.cpu().numpy()
    labels = labels.cpu().numpy()
    src_node, targ_node = np.nonzero(A)
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) & (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = np.mean(matching)
    return edge_hom

def load_data(dataset_name):

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name)
    elif dataset_name in ['chameleon']:
        dataset = WikipediaNetwork(path, dataset_name)
    elif dataset_name in ['squirrel']:
        dataset = WikipediaNetwork(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['actor']:
        dataset = Actor(path)
    elif dataset_name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset_name)
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['wikics']:
        dataset = WikiCS(path)

    data = dataset[0]
    edges = remove_self_loops(data.edge_index)[0]
    adj = to_scipy_sparse_matrix(edges, num_nodes=data.x.size(0)).tocoo()
    adj = sparse_mx_to_torch_sparse_tensor(adj)


    features = data.x
    [nnodes, nfeats] = features.shape
    nclasses = torch.max(data.y).item() + 1

    if dataset_name in ['computers', 'photo', 'cs', 'physics', 'wikics']:
        train_mask, val_mask, test_mask = get_split(nnodes)
    else:
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
        val_mask = val_mask.unsqueeze(1)
        test_mask = test_mask.unsqueeze(1)
    print('train_mask:', train_mask, train_mask.shape)
    labels = data.y

    path = '../data_mono/adj_two_order/{}'.format(dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/{}.pt'.format(dataset_name)
    if os.path.exists(file_name):
        adj_two_order = torch.load(file_name)
        print('Load exist similarity matrix S.')
    else:
        print("Computing similarity matrix S.")
        N, M = compute_NM(features, adj)
        lb2 = []
        for i in tqdm(range(N.shape[0])):
            lb = optimize_lbd2(N[i], M[i])
            lb2.append(lb)
        lb2 = np.array(lb2)
        S = op_S(lb2, N, M)
        adj_two_order = A_final(S)
        print("homophily_ora:", homophily_v2(adj, labels)) 
        print("homophily_two_order", homophily_v2(adj_two_order, labels))
        torch.save(adj_two_order, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))

    path = '../data_mono/se/{}'.format(dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/{}_{}.pt'.format(dataset_name, 16)
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edges, nnodes)
        torch.save(se, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))

    return features, edges, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats, adj, adj_two_order



