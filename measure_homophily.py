from datasets.dataset import get_dataset
from torch_geometric.utils import homophily


def calculate_homophily():
    datamodule = get_dataset('node_hetero', 'AMAZON', 0, 64, False, 4)
    data = datamodule.train_dataloader().dataset._data

    print(homophily(data.edge_index, data.y))


if __name__ == '__main__':
    calculate_homophily()
