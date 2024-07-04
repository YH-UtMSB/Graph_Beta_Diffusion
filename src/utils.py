import os
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx


def F_deg(deg, degree_dist):
    """ cdf transformation on node degrees """
    cdf = torch.cumsum(degree_dist, dim=0).to(deg.device)
    deg = deg.to(torch.int64).clamp(0, len(cdf)-1)
    deg_percentile = cdf[deg.to(torch.int64)]
    return deg_percentile


def F_deg_inv(deg_percentile, degree_dist):
    """ mapping deg_percentile within [0,1] back to integer degrees """
    cdf = torch.cumsum(degree_dist, dim=0).to(deg_percentile.device)
    deg = torch.searchsorted(cdf, deg_percentile, side='right')
    deg = torch.clamp(deg, 0, len(cdf) - 1)
    return deg


def extract_valid_degrees(deg, node_masks):
    """
    Extract valid degrees for each sample in the batch
    Args:
    - deg (tensor): Node degrees, with shape (bs, n, 1)
    - node_masks (tensor): Binary node masks, with shape (bs, n)
    Returns:
    - valid_degrees (list of numpy arrays): Valid degrees for each sample in the batch
    """
    bs, n, _ = deg.shape
    valid_degrees = []

    for i in range(bs):
        # Extract valid degrees using node_masks
        valid_degrees_i = deg[i, node_masks[i].bool()]
        # Convert to NumPy array
        valid_degrees_i_np = valid_degrees_i.cpu().numpy()
        # Append to the list
        valid_degrees.append(valid_degrees_i_np)

    return valid_degrees


def tensor_to_deg_arrays(batch, node_masks):
    """ mapping x back to integer degree arrays """
    bs = batch.shape[0]
    degs_list = []
    for i in range(bs):
        degs_i = batch[i, node_masks[i].bool()]
        degs_i = degs_i.cpu().numpy()
        degs_list.append(degs_i)
    return degs_list



def to_dense(x, edge_index, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, _ = remove_self_loops(edge_index)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    # E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    # E = encode_no_edge(E)
    E = to_dense_adj(edge_index=edge_index, batch=batch, max_num_nodes=max_num_nodes)
    E = E.unsqueeze(-1)     # bs, n, n, 1

    return PlaceHolder(X=X, E=E, y=None), node_mask


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
            
        return self


###### new functions ######

def create_masks(node_mask):
    """ expand the (bs, n) sized node mask to x_mask (bs, n, 1), e_mask (bs, n, n) and create triu_mask """
    x_mask = node_mask.unsqueeze(-1)          # bs, n, 1

    bs, n = node_mask.shape
    device = node_mask.device
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    triu_mask = triu_mask.unsqueeze(0).expand(bs, n, n).unsqueeze(3)  # upper triangular mask for the batch, (bs, n, n, 1)
    
    e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
    e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
    diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).expand(bs, n, n).unsqueeze(3)      # bs, n, n, 1
    e_mask = e_mask1 * e_mask2 * diag_mask

    return x_mask, e_mask, triu_mask, diag_mask


def VE_to_nx_graphs(lol):
    """
        converting an list-of-lists (lol) into a list of nx graphs
        lol: a list of graphs, with each graph represented by a two-item list
    """
    nx_graph_list = []
    for graph in lol:
        node_types, edge_types = graph
        A = edge_types.bool().cpu().numpy()
        nx_graph = nx.from_numpy_array(A)
        nx_graph_list.append(nx_graph)
    return nx_graph_list


def tensor_to_nx_graphs(batch, node_masks=None):
    """
        converts a tensor batch (bs, n, n) to a list of bs nx graphs
    """
    bs = batch.shape[0]
    nx_graph_list = []
    for i in range(bs):
        adj = batch[i].detach().cpu().numpy()               # (n, n)
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        if node_masks is not None:                          # mask the dummy nodes if a valid node_masks is passed to the function
            node_mask = node_masks[i].detach().cpu().numpy()    # (n, )
            valid_nodes = [v for v in G.nodes if node_mask[v] == 1.]
            G = G.subgraph(valid_nodes).copy()
        else:
            G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        nx_graph_list.append(G)
    return nx_graph_list


### GDSS similar function ###
# def adjs_to_graphs(adjs, is_cuda=False):
#     graph_list = []
#     for adj in adjs:
#         if is_cuda:
#             adj = adj.detach().cpu().numpy()
#         G = nx.from_numpy_array(adj)
#         G.remove_edges_from(nx.selfloop_edges(G))
#         G.remove_nodes_from(list(nx.isolates(G)))
#         if G.number_of_nodes() < 1:
#             G.add_node(1)
#         graph_list.append(G)
#     return graph_list


def discretize_nx_graphs(nx_graphs, p=0.9):
    out = []
    for g in nx_graphs:
        adj = nx.to_numpy_array(g)
        adj_bin = nx.from_numpy_array(np.where(adj>p, 1, 0))
        out.append(adj_bin)
    return out


def get_unique_graphs(nx_graph_list):
    """
        find non-isomorphic graphs among a list of nx graphs
    """
    unique_graphs = [nx_graph_list[0]]
    for i in range(1, len(nx_graph_list)):
        nx_graph = nx_graph_list[i]
        matches = 0
        for g in unique_graphs:
            if nx.is_isomorphic(g, nx_graph):
                matches += 1
        if matches == 0:
            unique_graphs.append(nx_graph)
    return unique_graphs


def get_ug_set_difference(setA, setB):
    """
        both setA and setB are lists of unique nx graphs, find setA - setB
    """
    setA_minus_setB = []
    for gA in setA:
        matches = 0
        for gB in setB:
            if nx.is_isomorphic(gA, gB):
                matches += 1
        if matches == 0:
            setA_minus_setB.append(gA)
    return setA_minus_setB


def visualize_nx_graphs(nx_graph_list, threshold=0.9, plottables=['struct', 'adj', 'hist'], num_cols=10, titles=None):
    """
        plot attributes of nx graphs
        - structure: graph structure
        - adj: adjacency matrix, pertaining to the unquantized graph 
        - hist: histograms, pertaining to the unquantized adj
    """
    H = len(plottables)
    num_rows = math.ceil(len(nx_graph_list) / num_cols) * H
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))

    for idx in range(len(nx_graph_list)):
        graph = nx_graph_list[idx]
        adj = nx.to_numpy_array(graph)
        graph = nx.from_numpy_array(np.where(adj>threshold, 1, 0))

        i, j = divmod(idx, num_cols)
        counter = 0

        # graph architecture
        if 'struct' in plottables:
            ax = axes[H*i+counter, j]
            w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
            vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
            m = max(np.abs(vmin), vmax)
            vmin, vmax = -m, m

            nx.draw(graph, ax=ax, font_size=5, node_size=100, with_labels=True, node_color=U[:, 1],
                    cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')
            
            if counter == 0:
                if titles is None:
                    ax.set_title(f'graph {idx}')
                else:
                    ax.set_title(titles[idx])

            counter += 1

        # graph adjacency matrix
        if 'adj' in plottables:
            ax = axes[H*i+counter, j]
            if 'struct' in plottables:
                sorted_nodes = np.argsort(U[:, 1])
                adj_reordered = adj[:, sorted_nodes][sorted_nodes, :]
                ax.imshow(adj_reordered, cmap='binary', interpolation='nearest')
            else:
                ax.imshow(adj, cmap='binary', interpolation='nearest')
            
            if counter == 0:
                if titles is None:
                    ax.set_title(f'graph {idx}')
                else:
                    ax.set_title(titles[idx])

            counter += 1

        # histogram
        if 'hist' in plottables:
            ax = axes[H*i+counter, j]
            ax.hist(adj.flatten(), bins=100, density=True, alpha=0.5)
            if counter == 0:
                if titles is None:
                    ax.set_title(f'graph {idx}')
                else:
                    ax.set_title(titles[idx])

    plt.tight_layout()
    plt.show()

