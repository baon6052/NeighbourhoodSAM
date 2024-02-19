from lightning import Trainer

from datasets.dataset import PlanetoidDataset, PlanetoidDatasetType, KarateClubDataset
from models.gcn import GCN


def main():
    datamodule = KarateClubDataset()
    model = GCN(num_features=datamodule.dataset.num_features,
                num_classes=datamodule.dataset.num_classes)

    trainer = Trainer(limit_train_batches=100, max_epochs=500, fast_dev_run=True)

    trainer.fit(model, train_dataloaders=datamodule.train_dataloader())


if __name__ == "__main__":
    main()
