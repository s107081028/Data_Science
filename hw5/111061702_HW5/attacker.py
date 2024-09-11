import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module
import torch
import torch.nn.functional as F
from core.utils import sparse_mx_to_torch_sparse_tensor, to_tensor

class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.device = device

        self.modified_adj = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        raise NotImplementedError()


class RND(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(RND, self).__init__(model, nnodes, device=device)

    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, target_node: int, n_perturbations: int, **kwargs):
        """
        Randomly sample nodes u whose label is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """

        print(f'number of pertubations: {n_perturbations}')
        modified_adj = ori_adj.tolil()

        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[: n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[: n_perturbations-len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            pass

        self.modified_adj = modified_adj


# TODO: Implemnet your own attacker here
class MyAttacker(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(MyAttacker, self).__init__(model, nnodes, device=device)

    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, target_node: int, n_perturbations: int, **kwargs):
        # print(f'Number of perturbations: {n_perturbations}')
        modified_adj = ori_adj.tolil()
        
        # connect edges with different labeled nodes which are most likely to confuse the surrogate model
        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        adj, features = to_tensor(ori_adj, ori_features, device=self.device)
        # ori_surrogate_labels = self.surrogate.predict(features, adj)
        loss_list = []
        for i in diff_label_nodes:
            new_adj = ori_adj.tolil()
            new_adj[target_node, i] = 1
            new_adj[i, target_node] = 1
            adj = sparse_mx_to_torch_sparse_tensor(new_adj).to(self.device)
            surrogate_labels = self.surrogate.predict(features, adj)
            # loss_list.append(torch.sum(torch.square(surrogate_labels[target_node] - ori_surrogate_labels[target_node])))
            loss_list.append(surrogate_labels[target_node][labels[target_node]])
        # sort_ls = [diff_label_nodes[i] for i in sorted(range(len(loss_list)), key=lambda k: loss_list[k], reverse=True)]
        add_sort_ls = [diff_label_nodes[i] for i in sorted(range(len(loss_list)), key=lambda k: loss_list[k])]

        # delete edges between same labeled nodes which are most likely to make the surrogate model confident
        same_label_nodes = [x for x in idx_train if labels[x] == labels[target_node] and row[x] == 1]
        loss_list = []
        for i in same_label_nodes:
            new_adj = ori_adj.tolil()
            new_adj[target_node, i] = 0
            new_adj[i, target_node] = 0
            adj = sparse_mx_to_torch_sparse_tensor(new_adj).to(self.device)
            surrogate_labels = self.surrogate.predict(features, adj)
            loss_list.append(surrogate_labels[target_node][labels[target_node]])
        delete_sort_ls = [same_label_nodes[i] for i in sorted(range(len(loss_list)), key=lambda k: loss_list[k])]

        # if the adding edges are more than the budget
        if len(add_sort_ls) >= n_perturbations:
            changed_nodes = add_sort_ls[:n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        # if the adding edges plus the deleting edges are more than perturbation
        elif len(add_sort_ls) + len(delete_sort_ls) >= n_perturbations:
            changed_nodes = add_sort_ls
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            delete_nodes = delete_sort_ls[:n_perturbations - len(add_sort_ls)]
            modified_adj[target_node, delete_nodes] = 0
            modified_adj[delete_nodes, target_node] = 0
        # if the adding edges plus the deleting edges are less than perturbation
        else:
            changed_nodes = add_sort_ls
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[:n_perturbations - len(add_sort_ls) - len(delete_sort_ls)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            delete_nodes = delete_sort_ls
            modified_adj[target_node, delete_nodes] = 0
            modified_adj[delete_nodes, target_node] = 0

        self.modified_adj = modified_adj