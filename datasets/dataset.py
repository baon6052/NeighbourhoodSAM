from enum import Enum, StrEnum, auto
from typing import Optional

import lightning as L
from torch_geometric.datasets import (
    GNNBenchmarkDataset,
    HeterophilousGraphDataset,
    KarateClub,
    Planetoid,
)
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.transforms.base_transform import BaseTransform


class DatasetType(StrEnum):
    NODE_HOMO = auto()
    NODE_HETERO = auto()
    GRAPH_HOMO = auto()


class Dataset(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.num_features = None
        self.num_classes = None

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

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)


class NodeClassificationDataset(L.LightningDataModule):
    def __init__(self, batch_size: int, num_neighbour: Optional[list[int]] = None,
                 use_neighbour_loader: bool = True):
        super().__init__()
        self.dataset = None
        self.num_features = None
        self.num_classes = None
        self.batch_size = batch_size
        self.use_neighbour_loader = use_neighbour_loader
        if num_neighbour is None:
            self.num_neighbour = [10, 10]
        else:
            self.num_neighbour = num_neighbour

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
        if self.use_neighbour_loader:
            return NeighborLoader(self.dataset.data, num_neighbors=self.num_neighbour,
                                  batch_size=self.batch_size)
        return DataLoader(self.dataset)

    def val_dataloader(self) -> DataLoader:
        if self.use_neighbour_loader:
            return NeighborLoader(self.dataset.data, num_neighbors=self.num_neighbour,
                                  batch_size=self.batch_size)
        return DataLoader(self.dataset)

    def test_dataloader(self) -> DataLoader:
        if self.use_neighbour_loader:
            return NeighborLoader(self.dataset.data, num_neighbors=self.num_neighbour,
                                  batch_size=self.batch_size)
        return DataLoader(self.dataset)



class KarateClubDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = KarateClub()
        self.print_info()


class PlanetoidDatasetType(Enum):
    CORA = "Cora"
    CITESEER = "CiteSeer"
    PUBMED = "PubMed"


class PlanetoidDataset(NodeClassificationDataset):
    def __init__(self, name: PlanetoidDatasetType, batch_size: int = 64,
                 num_neighbour: Optional[list[int]] = None,
                 use_neighbour_loader: bool = True):
        super().__init__(batch_size=batch_size, num_neighbour=num_neighbour,
                         use_neighbour_loader=use_neighbour_loader)
        self.dataset = Planetoid(root="data", name=name.value, split='full')
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.print_info()


class HeteroDatasetType(Enum):
    ROMAN = "Roman-empire"
    AMAZON = "amazon_ratings"
    MINESWEEPER = "minesweeper"
    TOLOKERS = "tolokers"
    QUESTIONS = "questions"


class SelectFoldTransform(BaseTransform):
    def __init__(self, fold_idx: int = 0):
        super().__init__()
        self.fold_idx = fold_idx

    def forward(self, data):
        data.train_mask = data.train_mask[:, self.fold_idx]
        data.val_mask = data.val_mask[:, self.fold_idx]
        data.test_mask = data.test_mask[:, self.fold_idx]

        return data


class HeteroDataset(Dataset):
    def __init__(self, name: HeteroDatasetType, fold_idx: int = 0):
        super().__init__()
        self.fold_idx = fold_idx
        self.dataset: HeterophilousGraphDataset = HeterophilousGraphDataset(
            root="data",
            name=name.value,
            transform=SelectFoldTransform(fold_idx),
        )
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.num_classes
        self.print_info()


class GraphHomoDatasetType(Enum):
    PATTERN = "PATTERN"
    CLUSTER = "CLUSTER"
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CSL = "CSL"


class GraphHomoDataset(Dataset):
    def __init__(self, name: GraphHomoDatasetType, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.train_data: GNNBenchmarkDataset = GNNBenchmarkDataset(
            root="data", name=name.value, split="train"
        )
        self.val_data: GNNBenchmarkDataset = GNNBenchmarkDataset(
            root="data", name=name.value, split="val"
        )
        self.test_data: GNNBenchmarkDataset = GNNBenchmarkDataset(
            root="data", name=name.value, split="test"
        )

        self.num_features = self.train_data.num_node_features
        self.num_classes = self.train_data.num_classes

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size)




def get_dataset(
        dataset_type: str, dataset_name: str, fold_idx: int, batch_size: int,
        neighbour_loader: bool, num_neighbours: list[int]
):
    if dataset_type == DatasetType.NODE_HOMO:
        datamodule = PlanetoidDataset(
            PlanetoidDatasetType[dataset_name], batch_size=batch_size,
            use_neighbour_loader=neighbour_loader, num_neighbours=num_neighbours
        )
    elif dataset_type == DatasetType.NODE_HETERO:
        datamodule = HeteroDataset(
            HeteroDatasetType[dataset_name], fold_idx=fold_idx
        )
    elif dataset_type == DatasetType.GRAPH_HOMO:
        datamodule = GraphHomoDataset(
            GraphHomoDatasetType[dataset_name], batch_size
        )
    return datamodule
