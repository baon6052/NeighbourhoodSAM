from lightning import Trainer

from datasets.dataset import (
    KarateClubDataset,
    PlanetoidDataset,
    PlanetoidDatasetType,
)
from models.gcn import GCN


def main():
    datamodule = PlanetoidDataset(PlanetoidDatasetType.CORA)
    model = GCN(
        num_features=datamodule.dataset.num_features,
        num_classes=datamodule.dataset.num_classes,
    )

    trainer = Trainer(
        limit_train_batches=100,
        max_epochs=500,
        fast_dev_run=False,
        accelerator="cpu",
    )

    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )


if __name__ == "__main__":
    main()
