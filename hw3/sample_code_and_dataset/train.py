from argparse import ArgumentParser

from data_loader import load_data
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy
import random

# from model import YourGNNModel # Build your model in model.py
from model import GCN, SplineConvNet, GraphSAGE, GAT
    
from dgl import DropEdge
import os
import warnings
warnings.filterwarnings("ignore")

def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, PATH, es_iters=None):
    
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        if random.random() < 0.5:
          transform = DropEdge(0.3)
          new_g = transform(g)
        else:
          new_g = g
        logits = model(new_g, features)
        loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(new_g, features, val_labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        
        logits = model(g, features)
        val_loss = loss_fcn(logits[val_mask], val_labels).item()
        if es_iters:
            if val_loss < loss_min:
                loss_min = val_loss
                es_i = 0
                torch.save(model, PATH)
            else:
                es_i += 1

            if es_i >= es_iters:
                print(f"Early stopping at epoch={epoch+1}")
                break

def evaluate_spline(features, edge_index, edge_attr, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index, edge_attr)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train_spline(features, edge_index, edge_attr, train_labels, val_labels, train_mask, val_mask, model, epochs, PATH, es_iters=None):
    
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        logits = model(features, edge_index, edge_attr)
        loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate_spline(features, edge_index, edge_attr, val_labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        
        val_loss = loss_fcn(logits[val_mask], val_labels).item()
        if es_iters:
            if val_loss < loss_min:
                loss_min = val_loss
                es_i = 0
                torch.save(model, PATH)
            else:
                es_i += 1

            if es_i >= es_iters:
                print(f"Early stopping at epoch={epoch+1}")
                break

if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping', default=100)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    epochs = args.epochs
    es_iters = args.es_iters

    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()
    
    edge_index = torch.stack(graph.edges())
    edge_index = edge_index.to(device)
    edge_attr = torch.zeros(torch.stack(graph.edges()).shape[1], 2).to(device)


    in_size = features.shape[1]
    out_size = num_classes
    
    SAGEmodel = GraphSAGE(in_size, 64, out_size)
    GATmodel = GAT(in_size, 64, out_size)
    Splinemodel = SplineConvNet(in_size, 128, 8).to(device)
    
    # model training
    print("Training Ensemble Models...")
    train(graph, features, train_labels, val_labels, train_mask, val_mask, SAGEmodel, epochs, 'SAGE_1.pt', es_iters)
    train(graph, features, train_labels, val_labels, train_mask, val_mask, GATmodel, epochs, 'GAT_1.pt', es_iters)
    train_spline(features.to(device), edge_index, edge_attr, train_labels.to(device), val_labels.to(device), train_mask, val_mask, Splinemodel, epochs, 'Spline_1.pt', es_iters)
    
    # Semi-spervised
    print("Generating Pseudo Label with Ensemble method...")
    SAGEmodel = torch.load("SAGE_1.pt")
    GATmodel = torch.load("GAT_1.pt")
    Splinemodel = torch.load("Spline_1.pt")
    pseudo = ~(train_mask | val_mask | test_mask)

    SAGEmodel.eval()
    with torch.no_grad():
        logits = SAGEmodel(graph, features)
        logits_SAGE = logits[pseudo]
        _, indices_SAGE = torch.max(logits_SAGE, dim=1)

    GATmodel.eval()
    with torch.no_grad():
        logits = GATmodel(graph, features)
        logits_GAT = logits[pseudo]
        _, indices_GAT = torch.max(logits_GAT, dim=1)

    Splinemodel.eval()
    with torch.no_grad():
        logits = Splinemodel(features.to(device), edge_index, edge_attr)
        logits_Spline = logits[pseudo]
        _, indices_Spline = torch.max(logits_Spline, dim=1)


    pseudo_label = []
    count = [0]*3
    for i in range(indices_SAGE.shape[0]):
        count = [0]*3
        count[indices_SAGE[i]] += 1
        count[indices_GAT[i]] += 1
        count[indices_Spline[i]] += 1
        if count[0] == 1 and count[1] == 1 and count[2] == 1:
            pseudo_label.append(random.choice([0, 1, 2]))
        else:
            pseudo_label.append(numpy.argmax(count))

    pseudo_labels = torch.cat((train_labels, torch.tensor(pseudo_label)))
    pseudo_mask = ~(val_mask | test_mask)

    SAGEmodel = GraphSAGE(in_size, 64, out_size)

    print("Training SAGE with pseudo labels...")
    train(graph, features, pseudo_labels, val_labels, pseudo_mask, val_mask, SAGEmodel, epochs, 'SAGE_1.pt', es_iters)
       
    print("Testing...")
    SAGEmodel.eval()
    with torch.no_grad():
        logits = SAGEmodel(graph, features)
        logits = logits[test_mask]
        _, indices = torch.max(logits, dim=1)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring