from argparse import ArgumentParser
import pickle
import torch
from numpy import ndarray
from scipy import sparse as sp
from core.gcn import GCN
from attacker import RND, MyAttacker
from core import Judge
from pathlib import Path


# DO NOT modify
def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='target_nodes_list.txt', help='Target node file.')
    parser.add_argument('--data_path', type=str, default='./data/data.pkl', help='Input graph.')
    parser.add_argument('--model_path', type=str, default='saved-models/gcn.pt', help='GNN model to attack.')
    parser.add_argument('--use_gpu', action='store_true')
    return parser.parse_args()


# DO NOT modify this function signature
def attack(adj: sp.csr_matrix, features: sp.csr_matrix, labels: ndarray,
           idx_train: ndarray, idx_val: ndarray, idx_test: ndarray,
           target_node: int, n_perturbations: int, **kwargs) -> sp.spmatrix:
    """
    Args:
        adj (sp.csr_matrix): Original (unperturbed) adjacency matrix
        features (sp.csr_matrix): Original (unperturbed) node feature matrix
        labels (ndarray): node labels
        idx_train (ndarray):node training indices
        idx_val (ndarray): node validation indices
        idx_test (ndarray): node test indices
        target_node (int): target node index to be attacked
        n_perturbations (int): Number of perturbations on the input graph.

    Returns:
        sp.spmatrix: attacked (perturbed) adjacency matrix
    """
    # TODO: Setup your attack model
    print(f'other args: {kwargs}')
    # model = RND()
    # model = MyAttacker()
    model = MyAttacker(model = kwargs["arg1"])
    model = model.to(device)
    model.attack(features, adj, labels, idx_train, target_node, n_perturbations=n_perturbations, **kwargs)
    return model.modified_adj


if __name__ == '__main__':
    args = get_args()

    cuda = torch.cuda.is_available()
    print(f'cuda: {cuda}')
    device = torch.device('cuda' if cuda and args.use_gpu else 'cpu')
    
    # You can load the data like this
    data = pickle.load(Path(args.data_path).open('rb'))
    adj, features, labels, idx_train, idx_val, idx_test = data

    # setup the surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16, dropout=0.5,
                with_relu=True, with_bias=True, device=device)
    gcn_path = Path("./saved-models/surrogate.pt")
    if gcn_path.is_file() and isinstance(surrogate, GCN):
        surrogate.load_state_dict(torch.load(gcn_path, map_location=device))
        surrogate = surrogate.to(device)
    else:
        surrogate = surrogate.to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30, verbose=True)
        torch.save(surrogate.state_dict(), gcn_path)
    surrogate.eval()

    # You can pass other arguments like this
    judge = Judge(args.data_path, args.model_path, args.input_file, device=device)
    judge.multi_test(attack, arg1=surrogate)
