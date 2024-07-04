import os
import pathlib
import pickle
import networkx as nx
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from .abstract_dataset import AbstractDataModule, AbstractDatasetInfos

class GenericGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.url_mappings = {
            'gdss-comm20': "https://github.com/harryjo97/GDSS/raw/master/data/community_small.pkl",
            'gdss-ego': "https://github.com/harryjo97/GDSS/raw/master/data/ego_small.pkl",
            'gdss-grid': "https://github.com/harryjo97/GDSS/raw/master/data/grid.pkl",
            'drum-sbm': "https://github.com/harryjo97/GruM/raw/master/DruM_2D/data/sbm_200.pt"
        }
        if dataset_name not in self.url_mappings:
            raise ValueError(f"Undefined dataset '{dataset_name}'.")
        self.dataset_name = dataset_name
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']
    
    @property
    def processed_file_names(self):
        return [self.split + ".pt"]
    
    def download(self):
        """ download data from urls listed in self.url_mappings """
        raw_path = download_url(self.url_mappings[self.dataset_name], self.raw_dir)
        
        if raw_path.endswith("pkl"):
            with open(raw_path, 'rb') as f:
                graph_list = pickle.load(f)
            tensor_list = []
            for g in graph_list:
                assert isinstance(g, nx.Graph)
                node_list = []
                for v, _ in g.nodes.data('feature'):
                    node_list.append(v)
                adj = nx.to_numpy_array(g, nodelist=node_list)
                tensor_list.append(torch.from_numpy(adj).to(torch.float32))

        elif raw_path.endswith("pt"):
            # adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(raw_path)
            tensor_list = torch.load(raw_path)[0]

        else:
            raise TypeError(f"data file {raw_path} has incorrect extension.")
        
        num_graphs = len(tensor_list)
        test_len = int(num_graphs * 0.2)
        val_len = int((num_graphs - test_len) * 0.2)
        train_data, val_data, test_data = tensor_list[(test_len+val_len):], tensor_list[test_len:(test_len+val_len)], tensor_list[:test_len]

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        """ process raw graphs to ensure compatibility with DiGress """
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_data = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_data:                # each instance in the raw data list is expected to be a 2-D tensor, indicating the adjacency matrix
            n = adj.shape[-1]

            # [v1 unique] degree as node feature
            # X = torch.ones(n, 1, dtype=torch.float)
            X = adj.sum(dim=-1, keepdim=True).to(torch.float32)   # (n,1)

            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=n_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                data_list.append(data)
        
        torch.save(self.collate(data_list), self.processed_paths[0])


class GenericGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {
            'train': GenericGraphDataset(dataset_name=self.cfg.dataset.name, root=root_path, split='train'),
            'val': GenericGraphDataset(dataset_name=self.cfg.dataset.name, root=root_path, split='val'),
            'test': GenericGraphDataset(dataset_name=self.cfg.dataset.name, root=root_path, split='test')
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
    

class GenericGraphDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'generic_graph_tensor_adjs'
        if dataset_config['name'] == 'gdss-grid':
            self.n_nodes = self.datamodule.node_counts(max_nodes_possible=400)
        else:
            self.n_nodes = self.datamodule.node_counts(max_nodes_possible=300)
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        self.degree_dist = self.datamodule.degree_counts()
        super().complete_infos(self.n_nodes, self.node_types)