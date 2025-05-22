import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random
import time
from dataset_load import load_dataset
from model import *
from utils import *

EOS = 1e-10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def train_cl(cl_model, edge_separator, optimizer_cl, features, str_encodings, edges, adj, adj_two_order):
    cl_model.train()
    edge_separator.eval()
    adj_1, adj_2, weights_lp, _, second_order_neighbors = edge_separator(torch.cat((features, str_encodings), 1), edges, adj, adj_two_order)
    features_1, adj_1, features_2, adj_2 = augmentation(features, adj_1, features, adj_2, args, cl_model.training)
    cl_loss = cl_model(features_1, adj_1, features_2, adj_2)
    optimizer_cl.zero_grad()
    cl_loss.backward()
    optimizer_cl.step()
    return cl_loss.item()


def train_edge_separator(cl_model, edge_separator, optimizer_disc, features, str_encodings, edges, adj, adj_two_order, args):
    cl_model.eval()
    edge_separator.train()
    adj_1, adj_2, weights_lp, weights_hp, second_order_neighbors = edge_separator(torch.cat((features, str_encodings), 1), edges, adj, adj_two_order)
    rand_np = generate_random_node_pairs(features.shape[0], second_order_neighbors.shape[1])
    psu_label = torch.ones(second_order_neighbors.shape[1]).cuda()
    embedding = cl_model.get_embedding(features, adj_1, adj_2)
    edge_emb_sim = F.cosine_similarity(embedding[second_order_neighbors[0]], embedding[second_order_neighbors[1]])
    rnp_emb_sim_lp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=args.margin_hom, reduction='none')
    loss_lp *= torch.relu(weights_lp - 0.5)
    rnp_emb_sim_hp = F.cosine_similarity(embedding[rand_np[0]], embedding[rand_np[1]])
    loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=args.margin_het, reduction='none')
    loss_hp *= torch.relu(weights_hp - 0.5)
    rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2
    optimizer_disc.zero_grad()
    rank_loss.backward()
    optimizer_disc.step()
    return rank_loss.item()


def main(args):
    start_time = time.time()
    setup_seed(0)
    features, edges, str_encodings, train_mask, val_mask, test_mask, labels, nnodes, nfeats, adj, adj_two_order = load_dataset(args.dataset)
    results = []

    for trial in range(args.ntrials):
        setup_seed(trial)
        cl_model = DC_contrastive(nlayers=args.nlayers_enc, nlayers_proj=args.nlayers_proj, in_dim=nfeats, emb_dim=args.emb_dim,
                    proj_dim=args.proj_dim, dropout=args.dropout, sparse=args.sparse, batch_size=args.cl_batch_size).cuda()
        cl_model.set_mask_knn(features.cpu(), k=args.k, dataset=args.dataset)
        edge_separator = Graph_edge_separator(nnodes, adj, adj_two_order, nfeats + str_encodings.shape[1], args.alpha, args.sparse, dataset_name=args.dataset).cuda()
        optimizer_cl = torch.optim.Adam(cl_model.parameters(), lr=args.lr_dc_contrastive, weight_decay=args.w_decay)
        optimizer_edge_separator = torch.optim.Adam(edge_separator.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)
        features = features.cuda()
        str_encodings = str_encodings.cuda()
        edges = edges.cuda()
        best_acc_val = 0
        best_acc_test = 0

        for epoch in range(1, args.epochs + 1):

            for _ in range(args.cl_rounds):
                cl_loss = train_cl(cl_model, edge_separator, optimizer_cl, features, str_encodings, edges, adj, adj_two_order)
                
            rank_loss = train_edge_separator(cl_model, edge_separator, optimizer_edge_separator, features, str_encodings, edges, adj, adj_two_order, args)
            print("[TRAIN] Epoch:{:04d} | CL Loss {:.4f} | RANK loss:{:.4f} ".format(epoch, cl_loss, rank_loss))

            if epoch % args.eval_freq == 0:
                cl_model.eval()
                edge_separator.eval()
                adj_1, adj_2, _, _, _ = edge_separator(torch.cat((features, str_encodings), 1), edges, adj, adj_two_order)
                embedding = cl_model.get_embedding(features, adj_1, adj_2)
                cur_split = 0 if (train_mask.shape[1]==1) else (trial % train_mask.shape[1])
                acc_test, acc_val = eval_test_mode(embedding, labels, train_mask[:, cur_split],
                                                 val_mask[:, cur_split], test_mask[:, cur_split])
                print(
                    '[TEST] Epoch:{:04d} | CL loss:{:.4f} | RANK loss:{:.4f} | VAL ACC:{:.2f} | TEST ACC:{:.2f}'.format(
                        epoch, cl_loss, rank_loss, acc_val, acc_test))

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    best_acc_test = acc_test

        results.append(best_acc_test)
        total_time = time.time() - start_time

    print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f} | Total Time: {:.2f} seconds'.format(args.dataset, args.ntrials, np.mean(results),
                                                                           np.std(results), total_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cornell',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell',
                                 'texas', 'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics', 
                                 'roman_empire', 'minesweeper', 'tolokers'])
    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=20)
    parser.add_argument('-epochs', type=int, default=400)
    parser.add_argument('-lr_dc_contrastive', type=float, default=0.001)
    parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-cl_rounds', type=int, default=2)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-margin_hom', type=float, default=0.5)
    parser.add_argument('-margin_het', type=float, default=0.5)
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1, choices=[1, 2])
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)

    args = parser.parse_args()

    print(args)
    main(args)
