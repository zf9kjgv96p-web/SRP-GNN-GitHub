from __future__ import division
from __future__ import print_function

import os
import time
import json
import argparse
import numpy as np
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Beta

from utils import *
from model import *
# ★★★  Degree 消融实验开关 (ABLATION CONFIG) ★★★
# =========================================================================
USE_DEGREE_WEIGHTING = True      # False → 完全去掉 degree+scorer，退化为等权 mean
USE_DEGREE_LOG      = True        # False → 用原始 degree 代替 log(degree)
USE_SCORER_NETWORK  = True        # False → 只用 degree，不用 scorer 网络
DEGREE_NORMALIZE    = True        # False → 类内不做归一化
# =========================================================================
# ★★★  C2 模块功能总控开关 (MASTER CONFIG)  ★★★
# 说明：
# - 下面这版 C2 是“严谨的 label-agnostic transductive”用法：
#   利用 query 的特征分布信息修正原型，但不使用 query 类别信息，
#   也不依赖 query 按类别分块的顺序（不再 view(n_way, ...)）。
# =========================================================================
USE_C2_MODULE = True
USE_ADAPTIVE_ALPHA = True         # True: alpha_net 自适应；False: 固定融合系数
USE_ALPHA_RANGE_LIMIT = True        # 仅当 USE_ADAPTIVE_ALPHA=True 时生效
ALPHA_MIN = 0.8
ALPHA_MAX = 0.9

C2_TAU = 1.0                        # soft-assignment 温度系数（越小越“硬”）
C2_EPS = 1e-12
# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='sgc', help='Graph encoder')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1600, help='Number of episodes.')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=9e-4, help='Weight decay.')
parser.add_argument('--hidden', type=int, default=16, help='Hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--num_tasks', type=int, default=5, help='Number of meta-training tasks.')
parser.add_argument('--intra', type=int, default=1, help='Intra-task augmentation multiples.')
parser.add_argument('--inter', type=int, default=5, help='Inter-task augmentation multiples.')
parser.add_argument('--way', type=int, default=5, help='N-way.')
parser.add_argument('--shot', type=int, default=5, help='K-shot.')
parser.add_argument('--qry', type=int, help='Query shot.', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Source Dataset')
parser.add_argument('--dataset_cr', default='corafull', help='Target (Cross-Domain) Dataset')
parser.add_argument('--q', type=int, default=7000, help='PCA dimension')

args, unknown = parser.parse_known_args()
args.cuda = torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

print("--- Experiment Configuration ---")
print("Source -> Target : {} -> {}".format(args.dataset, args.dataset_cr))
print(f"Device           : {args.device}")
print(f"C2 Module        : {USE_C2_MODULE} (label-agnostic query refinement)")
if USE_C2_MODULE:
    print(f"  - Adaptive Alpha : {USE_ADAPTIVE_ALPHA}")
    if USE_ADAPTIVE_ALPHA:
        print(f"  - Range Limit    : {USE_ALPHA_RANGE_LIMIT}")
        if USE_ALPHA_RANGE_LIMIT:
            print(f"    - Alpha Range  : [{ALPHA_MIN}, {ALPHA_MAX}]")
    print(f"  - C2 Tau         : {C2_TAU}")
print("---------------------------------")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# -------------------- Data --------------------
adj, features, labels, degrees, class_list_train, class_list_valid, _, id_by_class = load_data(
    args.dataset) if args.dataset in ('Amazon_clothing', 'Amazon_electronics', 'dblp') else load_cora_data()

adj_cr, features_cr, labels_cr, degrees_cr, _, _, class_list_test_cr, id_by_class_cr = load_data(
    args.dataset_cr) if args.dataset_cr in ('Amazon_clothing', 'Amazon_electronics', 'dblp') else load_cora_data()

# PCA
q_dim_src = min(args.q, features.shape[0] - 1, features.shape[1] - 1) if features.shape[0] > 1 and features.shape[1] > 1 else args.q
if features.shape[1] > q_dim_src:
    _, _, v_pca_src = torch.pca_lowrank(features, q=q_dim_src)
    features = torch.matmul(features, v_pca_src)

q_dim_cr = min(args.q, features_cr.shape[0] - 1, features_cr.shape[1] - 1) if features_cr.shape[0] > 1 and features_cr.shape[1] > 1 else args.q
if features_cr.shape[1] > q_dim_cr:
    _, _, v_pca_cr = torch.pca_lowrank(features_cr, q=q_dim_cr)
    features_cr = torch.matmul(features_cr, v_pca_cr)

if features.shape[1] != features_cr.shape[1]:
    raise ValueError("Feature dimensions after PCA do not match.")

# -------------------- Model --------------------
encoder = SGC_Encoder(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)
scorer = SGC_Valuator(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)

# 只有在需要自适应 alpha 时才实例化 alpha_net（否则不浪费）
alpha_net = None
optimizer_alpha = None

if USE_C2_MODULE and USE_ADAPTIVE_ALPHA:
    class AlphaGenerator(nn.Module):
        def __init__(self, hidden_dim):
            super(AlphaGenerator, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, proto_s, proto_q):
            return self.net(torch.cat([proto_s, proto_q], dim=-1))

    alpha_net = AlphaGenerator(hidden_dim=args.hidden)

optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_scorer = optim.Adam(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if USE_C2_MODULE and USE_ADAPTIVE_ALPHA:
    optimizer_alpha = optim.Adam(alpha_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

encoder.to(args.device)
scorer.to(args.device)
if alpha_net is not None:
    alpha_net.to(args.device)

features = features.to(args.device)
adj = adj.to(args.device)
labels = labels.to(args.device)
degrees = degrees.to(args.device)

features_cr = features_cr.to(args.device)
adj_cr = adj_cr.to(args.device)
labels_cr = labels_cr.to(args.device)
degrees_cr = degrees_cr.to(args.device)

# -------------------- C2: label-agnostic query refinement --------------------
def compute_proto_q_unlabeled(qry, proto_s, tau=C2_TAU, eps=C2_EPS):
    """
    Label-agnostic (no query labels, no class-block assumption).

    qry:     [Nq, D]      query embeddings (unlabeled)
    proto_s: [Nway, D]    support prototypes
    return:  proto_q [Nway, D]
    """
    # [Nq, Nway]
    dists = euclidean_dist(qry, proto_s)
    # soft assignment of each query sample to each class
    weights = torch.softmax(-dists / tau, dim=1)  # [Nq, Nway]
    # weighted aggregation to obtain query prototype per class
    proto_q = (weights.t() @ qry) / (weights.sum(0).unsqueeze(1) + eps)  # [Nway, D]
    return proto_q

# -------------------- Train / Eval --------------------
def cross_task(task1, task2, lam_mix, n_way, k_shot, q_qry):
    new_task = {}
    update = k_shot * (args.intra + 1)
    update_eval = q_qry * (args.intra + 1)

    task_2_shuffle_id = np.random.permutation(n_way)
    task_2_shuffle_id_s = np.concatenate([np.arange(update) + task_2_shuffle_id[i] * update for i in range(n_way)])
    task_2_shuffle_id_q = np.concatenate([np.arange(update_eval) + task_2_shuffle_id[i] * update_eval for i in range(n_way)])

    x2s = task2['spt'][task_2_shuffle_id_s]
    x2q = task2['qry'][task_2_shuffle_id_q]

    new_task['spt'], _ = mixup_data(task1['spt'], x2s, lam_mix)
    new_task['qry'], _ = mixup_data(task1['qry'], x2q, lam_mix)
    new_task['lab'] = task1['lab']
    return new_task

def train(class_selected, id_support, id_query, n_way, k_shot, q_qry, n_tasks):
    encoder.train()
    scorer.train()
    if USE_C2_MODULE and USE_ADAPTIVE_ALPHA:
        alpha_net.train()

    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    if USE_C2_MODULE and USE_ADAPTIVE_ALPHA:
        optimizer_alpha.zero_grad()

    embeddings = encoder(features, adj)
    scores = scorer(features, adj)
    z_dim = embeddings.size(1)

    dist = Beta(torch.tensor([0.5], device=args.device), torch.tensor([0.5], device=args.device))

    loss_train = 0
    output_all, labels_all = [], []
    task_dict = defaultdict(dict)

    # build base tasks
    for i in range(n_tasks):
        ori_support_embeddings = embeddings[id_support[i]].view(n_way, k_shot, z_dim)
        ori_query_embeddings = embeddings[id_query[i]].view(n_way, q_qry, z_dim)

        # avoid inplace ops
        if USE_DEGREE_WEIGHTING:
            raw_deg = degrees[id_support[i]].view(n_way, k_shot).float()
            if USE_DEGREE_LOG:
                degree_term = torch.log(raw_deg)
            else:
                degree_term = raw_deg

            if USE_SCORER_NETWORK:
                score_term = scores[id_support[i]].view(n_way, k_shot)
                combined = degree_term * score_term
            else:
                combined = degree_term

            support_scores = torch.sigmoid(combined).unsqueeze(-1)
            if DEGREE_NORMALIZE:
                support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
        else:
            support_scores = torch.ones(n_way, k_shot, 1, device=args.device)
        ori_support_embeddings = ori_support_embeddings * support_scores

        shuffle_id_s = np.random.permutation(k_shot)
        shuffle_id_q = np.random.permutation(q_qry)
        x2s = ori_support_embeddings[:, shuffle_id_s, :]
        x2q = ori_query_embeddings[:, shuffle_id_q, :]

        if args.intra > 0:
            x_mix_s = torch.cat([mixup_data(ori_support_embeddings, x2s, dist.sample())[0] for _ in range(args.intra)], dim=1)
            x_mix_q = torch.cat([mixup_data(ori_query_embeddings, x2q, dist.sample())[0] for _ in range(args.intra)], dim=1)
        else:
            x_mix_s = torch.empty(n_way, 0, z_dim, device=args.device)
            x_mix_q = torch.empty(n_way, 0, z_dim, device=args.device)

        support_embeddings = torch.cat([ori_support_embeddings, x_mix_s], dim=1).view(-1, z_dim)
        query_embeddings = torch.cat([ori_query_embeddings, x_mix_q], dim=1).view(-1, z_dim)

        labels_new = torch.LongTensor(
            [class_selected[i].index(j.item()) for j in labels[id_query[i]]]
        ).repeat_interleave(args.intra + 1).to(args.device)

        task_dict[i] = {'spt': support_embeddings, 'qry': query_embeddings, 'lab': labels_new}

    # build cross tasks
    for i in range(n_tasks):
        base = n_tasks + args.inter * i
        for j in range(args.inter):
            task_dict[base + j] = cross_task(
                task_dict[i],
                task_dict[(i + 1) % n_tasks],
                dist.sample(),
                n_way, k_shot, q_qry
            )

    # compute loss
    for k in range(len(task_dict)):
        spt, qry = task_dict[k]['spt'], task_dict[k]['qry']

        # prototype from support (use mean to avoid scale change caused by augmentation)
        proto_s = spt.view(n_way, -1, z_dim).mean(1)

        if USE_C2_MODULE:
            # label-agnostic query refinement (NO view(n_way, ...) !)
            proto_q = compute_proto_q_unlabeled(qry, proto_s)

            if USE_ADAPTIVE_ALPHA:
                raw_alpha = alpha_net(proto_s.detach(), proto_q.detach())
                if USE_ALPHA_RANGE_LIMIT:
                    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * raw_alpha
                else:
                    alpha = raw_alpha
                refined_proto = alpha * proto_s + (1.0 - alpha) * proto_q
            else:

                refined_proto = 0.75 * proto_s + 0.25 * proto_q
        else:
            refined_proto = proto_s

        output = F.log_softmax(-euclidean_dist(qry, refined_proto), dim=1)
        loss_train = loss_train + F.nll_loss(output, task_dict[k]['lab'])
        output_all.append(output)
        labels_all.append(task_dict[k]['lab'])

    if not output_all:
        return torch.tensor(0.0), torch.tensor(0.0)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()
    if USE_C2_MODULE and USE_ADAPTIVE_ALPHA:
        optimizer_alpha.step()

    out_cat = torch.cat(output_all).detach().cpu()
    lab_cat = torch.cat(labels_all).detach().cpu()
    return accuracy(out_cat, lab_cat), f1(out_cat, lab_cat)

def run_evaluation(class_selected, id_support, id_query, n_way, k_shot, domain='source'):
    encoder.eval()
    scorer.eval()
    if USE_C2_MODULE and USE_ADAPTIVE_ALPHA:
        alpha_net.eval()

    with torch.no_grad():
        eval_features, eval_adj, eval_labels, eval_degrees = (
            (features, adj, labels, degrees) if domain == 'source'
            else (features_cr, adj_cr, labels_cr, degrees_cr)
        )

        embeddings = encoder(eval_features, eval_adj)
        z_dim = args.hidden

        support_embeddings = embeddings[id_support].view(n_way, k_shot, z_dim)
        query_embeddings = embeddings[id_query]

        scores = scorer(eval_features, eval_adj)
        if USE_DEGREE_WEIGHTING:
            raw_deg = eval_degrees[id_support].view(n_way, k_shot).float()
            if USE_DEGREE_LOG:
                degree_term = torch.log(raw_deg)
            else:
                degree_term = raw_deg

            if USE_SCORER_NETWORK:
                score_term = scores[id_support].view(n_way, k_shot)
                combined = degree_term * score_term
            else:
                combined = degree_term

            support_scores = torch.sigmoid(combined).unsqueeze(-1)
            if DEGREE_NORMALIZE:
                support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
        else:
            support_scores = torch.ones(n_way, k_shot, 1, device=args.device)
        weighted_support_embeddings = support_embeddings * support_scores

        # weighted sum here equals weighted mean (weights sum to 1 per class)
        proto_s = weighted_support_embeddings.sum(1)

        if USE_C2_MODULE:
            proto_q = compute_proto_q_unlabeled(query_embeddings, proto_s)

            if USE_ADAPTIVE_ALPHA:
                raw_alpha = alpha_net(proto_s, proto_q)
                if USE_ALPHA_RANGE_LIMIT:
                    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * raw_alpha
                else:
                    alpha = raw_alpha
                refined_proto = alpha * proto_s + (1.0 - alpha) * proto_q
            else:
                refined_proto = 0.75 * proto_s + 0.25 * proto_q
        else:
            refined_proto = proto_s

        output = F.log_softmax(-euclidean_dist(query_embeddings, refined_proto), dim=1)
        labels_new = torch.LongTensor([class_selected.index(i.item()) for i in eval_labels[id_query]]).to(args.device)

        return accuracy(output.detach().cpu(), labels_new.detach().cpu()), f1(output.detach().cpu(), labels_new.detach().cpu())

# -------------------- Main --------------------
if __name__ == '__main__':
    n_way, k_shot, n_query, num_tasks = args.way, args.shot, args.qry, args.num_tasks
    meta_test_num, meta_valid_num = 50, 50

    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query, 1) for _ in range(meta_valid_num)]
    test_pool = [task_generator(id_by_class_cr, class_list_test_cr, n_way, k_shot, n_query, 1) for _ in range(meta_test_num)]

    print(f"Generating a fixed set of {num_tasks} meta-training tasks...")
    train_support, train_query, train_class_selected = task_generator(
        id_by_class, class_list_train, n_way, k_shot, n_query, num_tasks
    )

    t_total = time.time()
    best_val_acc = 0.0
    test_acc_at_best_val = 0.0
    test_f1_at_best_val = 0.0
    best_val_episode = 0

    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_test_episode = 0

    meta_train_acc_list = []

    for episode in range(1, args.episodes + 1):
        acc_train, f1_train = train(train_class_selected, train_support, train_query, n_way, k_shot, n_query, num_tasks)
        meta_train_acc_list.append(acc_train.item())

        if episode > 0 and episode % 50 == 0:
            print(f"\n------- Episode {episode} -------")
            avg_train_acc = np.mean(meta_train_acc_list[-50:])
            print(f"Meta-Train Acc (avg over last 50): {avg_train_acc:.4f}")

            meta_valid_results = [run_evaluation(cs, sup, qry, n_way, k_shot, domain='source') for sup, qry, cs in valid_pool]
            current_val_acc = np.mean([res[0] for res in meta_valid_results])

            meta_test_results = [run_evaluation(cs, sup, qry, n_way, k_shot, domain='target') for sup, qry, cs in test_pool]
            current_test_acc = np.mean([res[0] for res in meta_test_results])
            current_test_f1 = np.mean([res[1] for res in meta_test_results])

            print(f"Meta-Valid Acc  : {current_val_acc:.4f}  |  Best Valid: {best_val_acc:.4f}")

            # best by val
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_val_episode = episode
                test_acc_at_best_val = current_test_acc
                test_f1_at_best_val = current_test_f1
                print(f"Meta-Test Acc   : {current_test_acc:.4f}  |  Test Acc @ Best Valid: {test_acc_at_best_val:.4f} (New Best Valid!)")
            else:
                print(f"Meta-Test Acc   : {current_test_acc:.4f}  |  Test Acc @ Best Valid: {test_acc_at_best_val:.4f}")

            # best by TEST (as in your original code)
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
                best_test_f1 = current_test_f1
                best_test_episode = episode
            print(f"Best Test so far: {best_test_acc:.4f} @ Episode {best_test_episode}")

    # Save json
    parameter = defaultdict(list)
    parameter[str((best_test_acc, best_test_f1))].append({
        'lr': args.lr,
        'wd': args.weight_decay,
        'hidden': args.hidden,
        'dropout': args.dropout,
        'num_tasks': args.num_tasks,
        'intra': args.intra,
        'inter': args.inter,
        'use_c2': USE_C2_MODULE,
        'use_adaptive': USE_ADAPTIVE_ALPHA,
        'range_limit': USE_ALPHA_RANGE_LIMIT,
        'c2_tau': C2_TAU,
        'best_val_acc': float(best_val_acc),
        'best_val_episode': int(best_val_episode),
        'test_acc_at_best_val': float(test_acc_at_best_val),
        'test_f1_at_best_val': float(test_f1_at_best_val),
        'best_test_acc': float(best_test_acc),
        'best_test_f1': float(best_test_f1),
        'best_test_episode': int(best_test_episode),
    })

    file_name = f'{args.dataset}_to_{args.dataset_cr}_{args.way}way_{args.shot}shot.json'
    with open(file_name, 'a', newline='\n') as f:
        json.dump(parameter, f)

    print("\n================= Training Finished ==================")
    print(f"Total time elapsed: {time.time() - t_total:.4f}s")

    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f} at Episode {best_val_episode}")
    print(f"Test Accuracy at Best Validation: {test_acc_at_best_val:.4f}")
    print(f"Test F1 Score at Best Validation: {test_f1_at_best_val:.4f}")

    print(f"Best Test Accuracy achieved: {best_test_acc:.4f} at Episode {best_test_episode}")
    print(f"Best Test F1 Score achieved: {best_test_f1:.4f}")
