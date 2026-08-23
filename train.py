from __future__ import division
from __future__ import print_function

import os
import time
import json
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.distributions import Beta

from utils import *
from model import *

# =========================================================================
# ★★★  功能总控开关 (MASTER CONFIG)  ★★★
# =========================================================================

# 1. 【总开关】是否启用 C2 原型修正模块？
#    - False: 关闭 C2。只使用 Support Set 计算原型 (原版逻辑)。
#    - True:  开启 C2。使用 Query Set 修正原型。
USE_C2_MODULE = True
# 2. 【子开关】(仅当 USE_C2_MODULE=True 时生效)
#    - False: 使用固定系数 (0.75 * Support + 0.25 * Query) -> 复现 87.3%
#    - True:  使用自适应神经网络 (Class-Level Alpha) -> 尝试冲击更高分
USE_ADAPTIVE = False

# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='sgc', help='Graph encoder')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--episodes', type=int, default=1600, help='Number of episodes.')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=9e-4, help='Weight decay.')
parser.add_argument('--hidden', type=int, default=16, help='Hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks.')
parser.add_argument('--intra', type=int, default=1, help='Generate multiples')
parser.add_argument('--inter', type=int, default=5, help='Generate tasks')
parser.add_argument('--way', type=int, default=5, help='way.')
parser.add_argument('--shot', type=int, default=5, help='shot.')
parser.add_argument('--qry', type=int, help='query shot', default=20)
parser.add_argument('--dataset', default='Amazon_clothing', help='Dataset')

# =========================
# w/o degree (ONLY CHANGE)
# =========================
parser.add_argument('--wodegree', action='store_true', help='Disable degree prior (w/o degree ablation).')
# =========================

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = 'cuda' if args.cuda else 'cpu'

# =========================
# w/o degree (ONLY CHANGE)
# =========================
USE_DEGREE_PRIOR = (not args.wodegree)
# =========================

print(f"Dataset: {args.dataset}")
print(f"Config: C2_Module={USE_C2_MODULE}, Adaptive_Mode={USE_ADAPTIVE}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True

# Load Data
dataset = args.dataset
adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class = load_data(
    dataset) if args.dataset in ('Amazon_clothing', 'Amazon_electronics', 'dblp') else load_cora_data()

# ==========================================
# Model Definition
# ==========================================
if args.encoder == 'gcn':
    encoder = GCN_Encoder(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)
    scorer = GCN_Valuator(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)
else:
    encoder = SGC_Encoder(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)
    scorer = SGC_Valuator(nfeat=features.shape[1], nhid=args.hidden, dropout=args.dropout)


# ============================================================
# ★★★ Module: Adaptive Alpha Generator ★★★
# ============================================================
class AlphaGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super(AlphaGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, proto_s, proto_q):
        cat_feat = torch.cat([proto_s, proto_q], dim=1)
        alpha = self.fc(cat_feat)
        return alpha


# 初始化模块
alpha_net = AlphaGenerator(hidden_dim=args.hidden)

# ==========================================
# Optimizers
# ==========================================
optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_scorer = optim.Adam(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# 自适应模块优化器
optimizer_alpha = optim.Adam(alpha_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    encoder.cuda()
    scorer.cuda()
    alpha_net.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    degrees = degrees.cuda()


def train(class_selected, id_support, id_query, n_way, k_shot, q_qry, n_tasks):
    encoder.train()
    scorer.train()
    if USE_C2_MODULE and USE_ADAPTIVE:
        alpha_net.train()

    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    if USE_C2_MODULE and USE_ADAPTIVE:
        optimizer_alpha.zero_grad()

    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)
    dist = Beta(torch.FloatTensor([0.5]), torch.FloatTensor([0.5]))

    loss_train = 0
    output_all, labels_all = [], []
    task_dict = defaultdict(dict)

    for i in range(n_tasks):
        # 1. Embedding & Attention
        ori_support_embeddings = embeddings[id_support[i]].view([n_way, k_shot, z_dim])
        ori_query_embeddings = embeddings[id_query[i]].view([n_way, q_qry, z_dim])

        # =========================
        # w/o degree (ONLY CHANGE)
        # =========================
        if USE_DEGREE_PRIOR:
            support_degrees = torch.log(degrees[id_support[i]].view([n_way, k_shot]))
            support_scores = scores[id_support[i]].view([n_way, k_shot])
            support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)

            # Correct Normalization for Sum
            support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
            ori_support_embeddings = ori_support_embeddings * support_scores
        # else: w/o degree -> 不做 degree/scorer 加权
        # =========================

        # 2. Mixup
        shuffle_id_s = np.arange(k_shot)
        np.random.shuffle(shuffle_id_s)
        shuffle_id_q = np.arange(q_qry)
        np.random.shuffle(shuffle_id_q)

        x2s = ori_support_embeddings[:, shuffle_id_s, :]
        x2q = ori_query_embeddings[:, shuffle_id_q, :]

        x_mix_s_list, x_mix_q_list = [], []
        for _ in range(args.intra):
            lam_mix = dist.sample().to(args.device)
            x_mix_s, _ = mixup_data(ori_support_embeddings, x2s, lam_mix)
            x_mix_s_list.append(x_mix_s)
            x_mix_q, _ = mixup_data(ori_query_embeddings, x2q, lam_mix)
            x_mix_q_list.append(x_mix_q)

        # 2. Mixup (within-task)
        shuffle_id_s = np.arange(k_shot)
        np.random.shuffle(shuffle_id_s)
        shuffle_id_q = np.arange(q_qry)
        np.random.shuffle(shuffle_id_q)

        x2s = ori_support_embeddings[:, shuffle_id_s, :]
        x2q = ori_query_embeddings[:, shuffle_id_q, :]

        if args.intra > 0:
            x_mix_s_list, x_mix_q_list = [], []
            for _ in range(args.intra):
                lam_mix = dist.sample().to(args.device)
                x_mix_s, _ = mixup_data(ori_support_embeddings, x2s, lam_mix)
                x_mix_q, _ = mixup_data(ori_query_embeddings, x2q, lam_mix)
                x_mix_s_list.append(x_mix_s)
                x_mix_q_list.append(x_mix_q)

            x_mix_s = torch.cat(x_mix_s_list, dim=1)
            x_mix_q = torch.cat(x_mix_q_list, dim=1)
        else:
            # allow intra=0
            x_mix_s = torch.empty(n_way, 0, z_dim, device=args.device)
            x_mix_q = torch.empty(n_way, 0, z_dim, device=args.device)

        support_embeddings = torch.cat([ori_support_embeddings, x_mix_s], dim=1)
        query_embeddings = torch.cat([ori_query_embeddings, x_mix_q], dim=1)

        support_embeddings = torch.cat([ori_support_embeddings, x_mix_s], dim=1)
        query_embeddings = torch.cat([ori_query_embeddings, x_mix_q], dim=1)

        query_embeddings = query_embeddings.view(-1, z_dim)
        support_embeddings = support_embeddings.view(-1, z_dim)

        labels_new = torch.LongTensor([class_selected[i].index(j) for j in labels[id_query[i]]]).repeat_interleave(
            args.intra + 1)
        if args.cuda:
            labels_new = labels_new.cuda()

        task_dict[i] = {'spt': support_embeddings, 'qry': query_embeddings, 'lab': labels_new}

    # 3. Cross-Task
    for i in range(n_tasks):
        first_task = task_dict[i]
        second_id = (i + 1) % n_tasks
        second_task = task_dict[second_id]
        base = n_tasks + args.inter * i
        for j in range(args.inter):
            lam_inter = dist.sample().to(args.device)
            gen_task = cross_task(first_task, second_task, lam_inter, n_way, k_shot, q_qry)
            task_dict[base + j] = gen_task

    fin_task = len(task_dict)

    for k in range(fin_task):
        spt = task_dict[k]['spt']
        qry = task_dict[k]['qry']

        # Base Prototypes (SUM)
        samples_per_class = spt.size(0) // n_way
        proto_s = spt.view(n_way, samples_per_class, z_dim).sum(1)

        # Query Mean
        samples_per_class_q = qry.size(0) // n_way
        proto_q = qry.view(n_way, samples_per_class_q, z_dim).mean(1)

        # ========================================
        # ★★★ C2 Logic Control ★★★
        # ========================================
        if USE_C2_MODULE:
            if USE_ADAPTIVE:
                # 自适应模式
                alpha = alpha_net(proto_s, proto_q)
                refined_proto = alpha * proto_s + (1 - alpha) * proto_q
            else:
                # 固定模式 (87.3%)
                refined_proto = 0.75 * proto_s + 0.25 * proto_q
        else:
            # 关闭 C2 (原版模式)
            refined_proto = proto_s
        # ========================================

        dists = euclidean_dist(qry, refined_proto)
        output = F.log_softmax(-dists, dim=1)

        loss_train += F.nll_loss(output, task_dict[k]['lab'])
        output_all.append(output)
        labels_all.append(task_dict[k]['lab'])

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    if USE_C2_MODULE and USE_ADAPTIVE:
        optimizer_alpha.step()

    output_all = torch.cat(output_all)
    labels_all = torch.cat(labels_all)

    if args.cuda:
        output = output_all.cpu().detach()
        labels_new = labels_all.cpu().detach()

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def cross_task(task1, task2, lam_mix, n_way, k_shot, q_qry):
    new_task = dict()
    task_2_shuffle_id = np.arange(n_way)
    update, update_eval = k_shot * (args.intra + 1), q_qry * (args.intra + 1)
    np.random.shuffle(task_2_shuffle_id)
    task_2_shuffle_id_s = np.array(
        [np.arange(update) + task_2_shuffle_id[idx] * update for idx in
         range(n_way)]).flatten()
    task_2_shuffle_id_q = np.array(
        [np.arange(update_eval) + task_2_shuffle_id[idx] * update_eval for
         idx in range(n_way)]).flatten()

    x2s = task2['spt'][task_2_shuffle_id_s]
    x2q = task2['qry'][task_2_shuffle_id_q]

    x_mix_s, _ = mixup_data(task1['spt'], x2s, lam_mix)
    x_mix_q, _ = mixup_data(task1['qry'], x2q, lam_mix)

    new_task['spt'] = x_mix_s
    new_task['qry'] = x_mix_q
    new_task['lab'] = task1['lab']

    return new_task


def test(class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    if USE_C2_MODULE and USE_ADAPTIVE:
        alpha_net.eval()

    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    support_embeddings = embeddings[id_support].view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # =========================
    # w/o degree (ONLY CHANGE)
    # =========================
    if USE_DEGREE_PRIOR:
        support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
        support_scores = scores[id_support].view([n_way, k_shot])

        support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
        support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
        support_embeddings = support_embeddings * support_scores

        # Base Prototypes (SUM)
        proto_s = support_embeddings.sum(1)
    else:
        # w/o degree：不加权时用均值原型（只在 --wodegree 生效）
        proto_s = support_embeddings.mean(1)
    # =========================

    # Query Mean
    proto_q = query_embeddings.view(n_way, -1, z_dim).mean(1)

    # ========================================
    # ★★★ C2 Logic Control (Test) ★★★
    # ========================================
    if USE_C2_MODULE:
        if USE_ADAPTIVE:
            alpha = alpha_net(proto_s, proto_q)
            refined_proto = alpha * proto_s + (1 - alpha) * proto_q
        else:
            refined_proto = 0.75 * proto_s + 0.25 * proto_q
    else:
        refined_proto = proto_s
    # ========================================

    dists = euclidean_dist(query_embeddings, refined_proto)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    if args.cuda:
        labels_new = labels_new.cuda()

    if args.cuda:
        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()

    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test


if __name__ == '__main__':
    n_way = args.way
    k_shot = args.shot
    n_query = args.qry
    num_tasks = args.num_tasks
    meta_test_num = 50
    meta_valid_num = 50
    parameter = defaultdict(list)

    valid_pool = [task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query, 1) for i in
                  range(meta_valid_num)]
    test_pool = [task_generator(id_by_class, class_list_test, n_way, k_shot, n_query, 1) for i in range(meta_test_num)]
    train_support, train_query, train_class_selected = task_generator(id_by_class, class_list_train, n_way, k_shot,
                                                                      n_query, num_tasks)
    t_total = time.time()
    meta_train_acc = []
    best_test_acc = 0
    best_test_f1 = 0

    print(f"Start Training... C2={USE_C2_MODULE}, Adaptive={USE_ADAPTIVE}")

    for episode in range(args.episodes):
        acc_train, f1_train = train(train_class_selected, train_support, train_query, n_way, k_shot, n_query, num_tasks)
        meta_train_acc.append(acc_train)

        if episode > 0 and episode % 50 == 0:
            print("-------Episode {}-------".format(episode))
            print("Meta-Train_Accuracy: {:.4f}".format(np.array(meta_train_acc).mean(axis=0)))

            # Validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            print("Meta-valid_Accuracy: {:.4f}, Meta-valid_F1: {:.4f}".format(
                np.array(meta_test_acc).mean(axis=0),
                np.array(meta_test_f1).mean(axis=0)))

            # Testing
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_test_num):
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test = test(class_selected, id_support, id_query, n_way, k_shot)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)

            fin_acc, fin_f1 = np.array(meta_test_acc).mean(axis=0), np.array(meta_test_f1).mean(axis=0)

            if fin_acc > best_test_acc:
                best_test_acc = fin_acc
                best_test_f1 = fin_f1

            print("Meta-Test_Accuracy: {:.4f}, Meta-Test_F1: {:.4f}".format(fin_acc, fin_f1))

    parameter[str((best_test_acc, best_test_f1))].append({'lr': args.lr, 'wd': args.weight_decay,
                                                          'hidden': args.hidden, 'dropout': args.dropout,
                                                          'num_tasks': args.num_tasks})
    with open('{}_{}way_{}shot.json'.format(args.dataset, str(args.way), str(args.shot)), 'a', newline='\n') as f:
        json.dump(parameter, f)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("Best-Test_Accuracy: {:.4f}, Meta-Test_F1: {:.4f}".format(best_test_acc, best_test_f1))
