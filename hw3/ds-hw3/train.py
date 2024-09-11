from argparse import ArgumentParser

from data_loader import load_data

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from dgl.nn.pytorch.conv import SGConv

from model import GCN, GAT, GIN, LinkDist, Net, MixHop, Encoder, Grace, drop_feature
from model import Net_orig as SSP
from model import SaveBestModel
    
import os
import copy
import warnings
warnings.filterwarnings("ignore")

def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype = torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype = torch.long)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data

def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old

def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        # make a array index from 0 to features.shape[0]
        # batch = torch.arange(features.shape[0])
        # logits = model(g, features, torch.stack((g.edges()[0], g.edges()[1]), 0), batch)
        # logits = model(g, features)
        logits = model(features, torch.stack((g.edges()[0], g.edges()[1])))
        # logits, _ = model(features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, es_iters=None, lr=0.01):
    save_best_model = SaveBestModel()
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = nn.functional.nll_loss
    # loss_fcn = nn.NLLLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # logits = model(g, features)
        logits = model(features, torch.stack((g.edges()[0], g.edges()[1])))
        loss = loss_fcn(logits[train_mask], train_labels)
        acc = evaluate(g, features, val_labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        val_loss = loss_fcn(logits[val_mask], val_labels)

        save_best_model(
            val_loss, acc, epoch, model, optimizer, loss_fcn
        )
        loss = loss + val_loss
        loss.backward()
        optimizer.step()
        if es_iters:
            if val_loss < loss_min:
                loss_min = val_loss
                es_i = 0
            else:
                es_i += 1

            if es_i >= es_iters:
                print(f"Early stopping at epoch={epoch+1}")
                break

if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--hid', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    # Initialize the model (Baseline Model: GCN)
    in_size = features.shape[1]
    out_size = num_classes

    config = {
        'lr': [0.01, 0.1, 0.5, 0.05],
        'hid': [16, 32, 64, 128, 256],
    }
    # for hid in config['hid']:
        # for lr in config['lr']
    # model = GAT(in_size, args.hid, out_size).to(device)
    # model = SGConv(in_feats=in_size,
    #                out_feats=out_size,
    #                k=2,
    #                cached=True,
    #                bias=False).to(device)
    # model = GCN(in_size, hid, out_size).to(device)
    # model = GIN(in_size, 32, out_size).to(device)
    # model = GAT(in_size, 32, out_size).to(device)
    # model = GraphSAGE(in_size, 16, out_size).to(device)
    # model = LinkDist(in_size, 256, out_size).to(device)
    model = Net(in_size, args.hid, out_size).to(device)
    # model  = MixupNet(hid, in_size, out_size).to(device)
    # model = MixHop(
    #     in_dim=in_size,
    #     hid_dim=hid,
    #     out_dim=out_size,
    #     num_layers=3,
    #     p=[0,1,2],
    #     input_dropout=0.7,
    #     layer_dropout=0.9,
    #     activation=torch.tanh,
    #     batchnorm=True,
    # )
    # encoder = Encoder(in_size, hid, F.relu,
                    #   base_model=GCNConv, k=2).to(device)
    # model = Grace(encoder, hid, hid, 0.7).to(device)


    # model training
    print("Training...")
    train(graph, features, train_labels, val_labels, train_mask, val_mask, model, args.epochs, args.es_iters, args.lr)
    print("Validating...")
    # load best model
    checkpoint = torch.load('outputs/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    acc = evaluate(graph, features, val_labels, val_mask, model)
    print("Accuracy: {:.4f}".format(acc))

    print("Testing...")
    
    with torch.no_grad():
        # batch = torch.arange(features.shape[0])
        # logits = model(graph, features, torch.stack((graph.edges()[0], graph.edges()[1])), batch)
        # logits = model(graph, features)
        logits  = model(features, torch.stack((graph.edges()[0], graph.edges()[1])))
        # logits, _ = model(features)
        logits = logits[test_mask]
        _, indices = torch.max(logits, dim=1)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring