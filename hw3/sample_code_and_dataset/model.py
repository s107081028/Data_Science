import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn import GraphConv, SAGEConv, GATConv
from torch_geometric.nn import SplineConv

class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')
        self.dropout = nn.Dropout(0.2)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=8, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = GATConv(hidden_feats*8, out_feats, num_heads =1, allow_zero_in_degree=True)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = h.reshape(h.shape[0], -1)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = h.reshape(h.shape[0], -1)
        return h

class GAT_emb(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GAT_emb, self).__init__()
        self.hidden = hidden_feats
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=8)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = GATConv(hidden_feats*8, hidden_feats, num_heads =1)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = GATConv(hidden_feats, out_feats, num_heads =1)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = h.reshape(h.shape[0], -1)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = h.reshape(h.shape[0], -1)
        embedding = h
        h = self.dropout2(h)
        h = self.conv3(g, h)
        h = h.reshape(h.shape[0], -1)
        return h, embedding

class SplineConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SplineConvNet, self).__init__()
        self.conv1 = SplineConv(in_channels, hidden_channels, dim=2, kernel_size=5, aggr='add')
        # self.conv2 = SplineConv(hidden_channels*2, hidden_channels, dim=2, kernel_size=5, aggr='add')
        self.conv3 = SplineConv(hidden_channels, out_channels, dim=2, kernel_size=5, aggr='add')
        self.fc = nn.Linear(out_channels, 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(self.conv1(x, edge_index, edge_attr).relu())
        # x = self.dropout(self.conv2(x, edge_index, edge_attr).relu())
        x = self.dropout(self.conv3(x, edge_index, edge_attr).relu())
        # x = x.mean(dim=0)  # Global pooling operation
        x = self.fc(x)
        return x