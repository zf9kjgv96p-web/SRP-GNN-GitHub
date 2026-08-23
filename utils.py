import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
from dgl.data import CoraFullDataset
import scipy.sparse.linalg as spla
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "few_shot_data"

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_electronics': 36, 'dblp': 27}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(dataset_source):
    n1s = []
    n2s = []
    network_path = DATA_ROOT / f"{dataset_source}_network"
    if not network_path.exists():
        network_path = DATA_ROOT / dataset_source / f"{dataset_source}_network"
    for line in open(network_path, encoding="utf-8"):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    num_nodes = max(max(n1s), max(n2s)) + 1
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                        shape=(num_nodes, num_nodes))

    train_path = DATA_ROOT / f"{dataset_source}_train.mat"
    if not train_path.exists():
        train_path = DATA_ROOT / dataset_source / f"{dataset_source}_train.mat"
    data_train = sio.loadmat(train_path)
    train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))

    test_path = DATA_ROOT / f"{dataset_source}_test.mat"
    if not test_path.exists():
        test_path = DATA_ROOT / dataset_source / f"{dataset_source}_test.mat"
    data_test = sio.loadmat(test_path)
    class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

    labels = np.zeros((num_nodes, 1))
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]

    features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(features)

    labels = torch.LongTensor(np.where(labels)[1])

    adj = sparse_mx_to_torch_sparse_tensor(adj)

    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

    class_list_train = list(set(train_class).difference(set(class_list_valid)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class


def load_cora_data():
    print("this is CoraFull")
    data = CoraFullDataset(raw_dir=str(DATA_ROOT / "corafull"))
    minus_node = [1, 4, 43, 68, 69]  # node number less than 70
    g = data[0]
    # features = torch.FloatTensor(normalize(g.ndata['feat'].numpy())).to(device) # test case
    features = g.ndata['feat'].to(device)
    label = g.ndata['label']
    np_label = label.numpy()
    label = label.to(device)

    degree = g.in_degrees()
    degree = torch.FloatTensor(degree.numpy())
    adj = g.adjacency_matrix(scipy_fmt='coo')
    adj_noloop = normalize_adj(adj)  # useless
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    class_list = []
    for cla in np_label:
        if cla not in class_list:
            class_list.append(cla)

    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(np_label):
        id_by_class[cla].append(id)

    class_train = random.sample(class_list, 55)
    class_test = list(set(class_list).difference(set(class_train)))
    class_valid = random.sample(class_train, 15)
    class_train = list(set(class_train).difference(set(class_valid)))

    # minus less number node
    class_train = list(set(class_train).difference(set(minus_node)))
    class_valid = list(set(class_valid).difference(set(minus_node)))
    class_test = list(set(class_test).difference(set(minus_node)))

    return adj, features, label, degree, class_train, class_valid, class_test, id_by_class


def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='macro')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def task_generator(id_by_class, class_list, n_way, k_shot, m_query, n_tasks):
    if n_tasks == 1:
        # sample class indices
        class_selected = random.sample(class_list, n_way)
        id_support = []
        id_query = []
        for cla in class_selected:
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])
        return np.array(id_support), np.array(id_query), class_selected
    else:
        class_selected_fin = []
        id_support_fin = []
        id_query_fin = []
        for i in range(n_tasks):
            class_selected = random.sample(class_list, n_way)
            id_support = []
            id_query = []
            for cla in class_selected:
                temp = random.sample(id_by_class[cla], k_shot + m_query)
                id_support.extend(temp[:k_shot])
                id_query.extend(temp[k_shot:])
            class_selected_fin.append(class_selected)
            id_support_fin.append(id_support)
            id_query_fin.append(id_query)
        return np.array(id_support_fin), np.array(id_query_fin), class_selected_fin


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def mixup_data(xs, xq, lam):
    mixed_x = lam * xq + (1 - lam) * xs

    return mixed_x, lam


def normalize_adj_torch_fixed(adj):
    """
    adj: torch.sparse_coo_tensor
    return: normalized sparse adj (D^{-1/2} A D^{-1/2})
    不使用 sparse.sum，不使用稠密矩阵，安全不 OOM
    """

    # indices: [2, E]
    # values:  [E]
    indices = adj._indices()
    values = adj._values()
    N = adj.size(0)

    row = indices[0]
    col = indices[1]

    # ----- 1. 手动做行和（避免 sparse.sum 的 CUDA bug）-----
    row_sum = torch.zeros(N, device='cpu')  # 必须 CPU，避免 CUDA bug
    for r, v in zip(row.cpu(), values.cpu()):
        row_sum[r] += v

    # D^{-1/2}
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    # ----- 2. 计算 normalized values -----
    new_values = values.cpu() * d_inv_sqrt[row.cpu()] * d_inv_sqrt[col.cpu()]

    # ----- 3. 返回到 GPU sparse -----
    new_indices = indices.cpu()
    new_adj = torch.sparse_coo_tensor(new_indices, new_values, adj.shape)

    return new_adj.coalesce().to(adj.device)

def build_local_soft_graph_from_embeddings(embeddings, topk=30):
    """
    embeddings: torch.Tensor [M, D] (already on device)
    返回 dense normalized adjacency matrix A_norm [M, M] (float, device = embeddings.device)
    ◦ uses cosine similarity (by normalizing embeddings)

    ◦ puts zeros on diagonal

    ◦ keeps topk neighbors per node (row)

    ◦ returns symmetric matrix and symmetric normalization D^{-1/2} A D^{-1/2} (dense)

    Note: M is small (e.g. <= 500), so dense ops are fine.
    """
    device = embeddings.device
    M = embeddings.size(0)
    if M == 0:
        return torch.zeros((0, 0), device=device)

    # normalize rows -> cosine
    E = F.normalize(embeddings, p=2, dim=1)  # [M, D]
    sim = torch.matmul(E, E.t())              # [M, M]

    # prevent self-selection
    sim.fill_diagonal_(-1e9)

    # topk (if topk >= M, we just take all)
    k = min(topk, M - 1) if M > 1 else 0
    if k > 0:
        vals, idx = torch.topk(sim, k=k, dim=1)  # vals [M,k], idx [M,k]
        # build sparse indices then symmetrize
        row_idx = torch.arange(M, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        col_idx = idx.reshape(-1)
        vals_flat = vals.reshape(-1)

        # create dense adjacency from these topk entries
        A = torch.zeros((M, M), device=device)
        A[row_idx, col_idx] = vals_flat
        # symmetric: average with transpose (keeps weights)
        A = 0.5 * (A + A.t())
    else:
        # fallback: zero adjacency (single node)
        A = torch.zeros((M, M), device=device)

    # add self-loop
    A = A + torch.eye(M, device=device)

    # row/col normalization D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)  # [M]
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    D_left = deg_inv_sqrt.view(M, 1)
    D_right = deg_inv_sqrt.view(1, M)
    A_norm = D_left * A * D_right

    return A_norm  # dense [M, M]
