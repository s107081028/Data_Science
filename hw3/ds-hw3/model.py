import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout, Parameter
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, MessagePassing

from dgl.nn.pytorch import GraphConv
import dgl.function as fn

from torch_geometric.nn.inits import uniform
import pdb

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
        self.dropout = nn.Dropout(0.8)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, x_cen):
        h = torch.matmul(x, self.weight)
        aggr_out = self.propagate(edge_index, size=None, h=h, edge_weight=None)
        return aggr_out + self.lin(x_cen)

    def message(self, h_j):
        return h_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class MixupNet(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(Net, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)

    def forward(self, x0, edge_index, edge_index_b, lam, id_new_value_old):

        x1 = self.conv1(x0, edge_index, x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index, x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)
        
        x0_b = x0[id_new_value_old]
        x1_b = x1[id_new_value_old]
        x2_b = x2[id_new_value_old]

        x0_mix = x0 * lam + x0_b * (1 - lam)

        new_x1 = self.conv1(x0, edge_index, x0_mix)
        new_x1_b = self.conv1(x0_b, edge_index_b, x0_mix)
        new_x1 = F.relu(new_x1)
        new_x1_b = F.relu(new_x1_b)

        x1_mix = new_x1 * lam + new_x1_b * (1 - lam)
        x1_mix = F.dropout(x1_mix, p=0.4, training=self.training)

        new_x2 = self.conv2(x1, edge_index, x1_mix)
        new_x2_b = self.conv2(x1_b, edge_index_b, x1_mix)
        new_x2 = F.relu(new_x2)
        new_x2_b = F.relu(new_x2_b)

        x2_mix = new_x2 * lam + new_x2_b * (1 - lam)
        x2_mix = F.dropout(x2_mix, p=0.4, training=self.training)

        new_x3 = self.conv3(x2, edge_index, x2_mix)
        new_x3_b = self.conv3(x2_b, edge_index_b, x2_mix)
        new_x3 = F.relu(new_x3)
        new_x3_b = F.relu(new_x3_b)

        x3_mix = new_x3 * lam + new_x3_b * (1 - lam)
        x3_mix = F.dropout(x3_mix, p=0.4, training=self.training)

        x = x3_mix
        x = self.lin(x)
        return x.log_softmax(dim=-1)
    
class GIN(nn.Module):
    """
    TODO: Use GCN model as reference, implement your own model here to achieve higher accuracy on testing data
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        # self.conv1 = GINConv(
        #     Sequential(Linear(in_size, hid_size),
        #                BatchNorm1d(hid_size), ReLU(),
        #                Linear(hid_size, hid_size), ReLU()))
        self.conv1 = GINConv(
            Sequential(Linear(in_size, hid_size), ReLU(),
                       Linear(hid_size, hid_size), ReLU()))
        # self.conv2 = GINConv(
        #     Sequential(Linear(hid_size, hid_size), BatchNorm1d(hid_size), ReLU(),
        #                Linear(hid_size, hid_size), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hid_size, hid_size), ReLU(),
                       Linear(hid_size, hid_size), ReLU()))
        # self.conv3 = GINConv(
        #     Sequential(Linear(hid_size, hid_size), BatchNorm1d(hid_size), ReLU(),
        #                Linear(hid_size, hid_size), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hid_size, hid_size), ReLU(),
                       Linear(hid_size, hid_size), ReLU()))
        self.lin1 = Linear(hid_size*3, hid_size*3)
        self.lin2 = Linear(hid_size*3, out_size)
    
    def forward(self, g, features, edge_index, batch):
        h1 = self.conv1(features, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        # return h, F.log_softmax(h, dim=1)
        return h

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.7, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.7, training=self.training)
    h = self.gat2(h, edge_index)
    return h

class GraphSAGE(torch.nn.Module):
  """GraphSAGE"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.sage1 = SAGEConv(dim_in, dim_h)
    self.sage2 = SAGEConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

def FC(din, dout):
    return nn.Sequential(
        nn.BatchNorm1d(din),
        nn.LayerNorm(din),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(din, dout))

class MLP(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3, A=None):
        super(self.__class__, self).__init__()
        self.A = A
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(din, hid))
        for _ in range(n_layers - 2):
            self.layers.append(FC(hid, hid))
        self.layers.append(FC(hid, dout))

    def forward(self, x):
        for layer in self.layers:
            if self.A is not None:
                x = self.A @ x
            x = layer(x)
        return x

class LinkDist(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3):
        super(self.__class__, self).__init__()
        self.mlp = MLP(din, hid, hid, n_layers=n_layers - 1)
        self.out = FC(hid, dout)
        self.inf = FC(hid, dout)

    def forward(self, x):
        x = self.mlp(x)
        return self.out(x), self.inf(x)

class Net_orig(torch.nn.Module):
    def __init__(self, dataset):
        super(Net2, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
class Net(torch.nn.Module):
    def __init__(self, din, dhid, dout):
        super(Net, self).__init__()
        self.crd = CRD(din, dhid, 0.5)
        self.cls = CLS(dhid, dout)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, x, edge_index):
        x = self.crd(x, edge_index)
        x = self.cls(x, edge_index)
        return x

class MLP1(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.LeakyReLU(),
            nn.Linear(hid_feats, out_feats),
        )

    def forward(self, x):
        return self.pred(x)

class Res(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(self.__class__, self).__init__()
        self.dec = nn.Linear(in_feats, hid_feats)
        self.pred = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(2 * hid_feats, out_feats),
            nn.Tanh(),
        )
        # self.w = nn.Parameter(gpu(torch.ones(1, out_feats)))
        # self.b = nn.Parameter(gpu(torch.zeros(1, out_feats)))

    def forward(self, x, y):
        h = torch.cat((x, y), dim=-1)
        return self.pred(h)
        # return self.w * self.pred(h) + self.b

class Res3(nn.Module):
    def __init__(self, din, hid, dout):
        super(self.__class__, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(din, hid),
            nn.LeakyReLU(),
        )
        self.res = nn.Linear(hid * 2, dout)

    def forward(self, x, x0, y0):
        return self.res(torch.cat((
            self.enc(x).unsqueeze(1).repeat(1, x0.shape[1], 1),
            self.enc(x0)
        ), dim=-1))

class TF(nn.Module):
    def __init__(self, din, hid, dout):
        super(self.__class__, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(din, hid),
            nn.LeakyReLU(),
        )
        self.res = nn.Linear(hid * 2 + dout, dout)

    def forward(self, x, x0, y0, logw=None):
        y = self.res(torch.cat((
            self.enc(x).unsqueeze(1).repeat(1, x0.shape[1], 1),
            self.enc(x0),
            y0,
        ), dim=-1))
        w = y0 * 0
        if logw is not None:
            w = w + logw.unsqueeze(-1)
        y = (y * torch.softmax(w, dim=1)).sum(dim=1)
        return y

class KD(nn.Module):
    def __init__(self, din, hid, dout):
        super(self.__class__, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(din, hid),
            nn.LeakyReLU(),
        )
        self.res = nn.Linear(hid * 2, dout)
        self.W = nn.Parameter(torch.rand(dout, dout))

    def forward(self, x, x0, y0):
        return y0 @ self.W + self.res(
            torch.cat((self.enc(x), self.enc(x0)), dim=-1))

class MixHopConv(nn.Module):
    r"""

    Description
    -----------
    MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
     <https://arxiv.org/pdf/1905.00067.pdf>`__.

    .. math::
        H^{(i+1)} =\underset{j \in P}{\Bigg\Vert} \sigma\left(\widehat{A}^j H^{(i)} W_j^{(i)}\right),

    where :math:`\widehat{A}` denotes the symmetrically normalized adjacencymatrix with self-connections,
    :math:`D_{ii} = \sum_{j=0} \widehat{A}_{ij}` its diagonal degree matrix,
    :math:`W_j^{(i)}` denotes the trainable weight matrix of different MixHop layers.

    Parameters
    ----------
    in_dim : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_dim : int
        Output feature size for each power.
    p: list
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    dropout: float, optional
        Dropout rate on node features. Defaults: ``0``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    batchnorm: bool, optional
        If True, use batch normalization. Defaults: ``False``.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        p=[0, 1, 2],
        dropout=0,
        activation=None,
        batchnorm=False,
    ):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict(
            {str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p}
        )

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):
                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata.pop("h")
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final

class MixHop(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        num_layers=2,
        p=[0, 1, 2],
        input_dropout=0.0,
        layer_dropout=0.0,
        activation=None,
        batchnorm=False,
    ):
        super(MixHop, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(
            MixHopConv(
                self.in_dim,
                self.hid_dim,
                p=self.p,
                dropout=self.input_dropout,
                activation=self.activation,
                batchnorm=self.batchnorm,
            )
        )

        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(
                MixHopConv(
                    self.hid_dim * len(self.p),
                    self.hid_dim,
                    p=self.p,
                    dropout=self.layer_dropout,
                    activation=self.activation,
                    batchnorm=self.batchnorm,
                )
            )

        self.fc_layers = nn.Linear(
            self.hid_dim * len(self.p), self.out_dim, bias=False
        )

    def forward(self, graph, feats):
        feats = self.dropout(feats)
        for layer in self.layers:
            feats = layer(graph, feats)

        feats = self.fc_layers(feats)

        return feats

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

class Grace(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Grace, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'), best_acc=float(0.0)
    ):
        self.best_valid_loss = best_valid_loss
        self.best_acc = best_acc
        
    def __call__(
        self, current_valid_loss, current_acc,
        epoch, model, optimizer, criterion
    ):
        if current_acc > self.best_acc:
            self.best_valid_loss = current_valid_loss
            self.best_acc = current_acc
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nBest validation accuracy: {self.best_acc}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'acc': self.best_acc
                }, 'outputs/best_model.pth')

