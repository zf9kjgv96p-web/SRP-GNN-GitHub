import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution


class GCN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Encoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.dropout = dropout

        # 新增原型存储器（Amazon-Clothing有40个基础类）
        self.register_buffer(
            'prototypes',
            torch.zeros(40, nhid)  # [num_classes, hidden_dim]
        )

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class SGC_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SGC_Encoder, self).__init__()
        self.lin = nn.Linear(nfeat, nhid)

        # 新增原型存储器
        self.register_buffer(
            'prototypes',
            torch.zeros(40, nhid)  # [num_classes, hidden_dim]
        )

    def preprocess(self, x, adj):
        degree = 1
        for i in range(degree):
            x = torch.spmm(adj, x)
        return x

    def forward(self, x, adj):
        x = self.preprocess(x, adj)
        return self.lin(x)

class GCN_Valuator(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Valuator, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 2 * nhid)
        self.gc2 = GraphConvolution(2 * nhid, nhid)
        self.fc3 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc3(x)

        return x


class SGC_Valuator(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SGC_Valuator, self).__init__()
        self.lin = nn.Linear(nfeat, 1)

    def preprocess(self, x, adj):
        degree = 1
        for i in range(degree):
            x = torch.spmm(adj, x)
        return x

    def forward(self, x, adj):
        x = self.preprocess(x, adj)

        return self.lin(x)

