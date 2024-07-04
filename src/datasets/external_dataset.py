import os
import pathlib

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from .abstract_dataset import AbstractDataModule, AbstractDatasetInfos