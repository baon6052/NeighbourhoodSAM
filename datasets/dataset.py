from enum import Enum

import lightning as L
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.loader.dataloader import DataLoader


class Dataset(L.LightningDataModule):
    def __init__(self):
        self.dataset = None

    def print_info(self):
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of features: {self.dataset.num_features}")
        print(f"Number of classes: {self.dataset.num_classes}")

        data = self.dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(
            f"Training node "
            f"label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
        )
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)


class KarateClubDataset(Dataset):
    def __init__(self):
        self.dataset = KarateClub()
        self.print_info()


class PlanetoidDatasetType(Enum):
    CORA = "Cora"
    CITESEER = "CiteSeer"
    PUBMED = "PubMed"


class PlanetoidDataset(Dataset):
    def __init__(self, name: PlanetoidDatasetType):
        self.dataset = Planetoid(root="./datasets", name=name.value)
        self.print_info()
