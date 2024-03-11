import click
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from datasets.dataset import (
    DatasetType,
    GraphHomoDataset,
    GraphHomoDatasetType,
    HeteroDataset,
    HeteroDatasetType,
    PlanetoidDataset,
    PlanetoidDatasetType,
)
from models.gcn import GCN
from utilities.wandb_utilities import checkpoint_callback


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


def process_items(ctx, param, value):
    if value:
        items = value.split(",")
        items = [int(item.strip()) for item in items]
        return items


@click.command()
@click.option(
    "--dataset_type",
    type=click.Choice(
        ["node_homo", "node_hetero", "graph_homo"], case_sensitive=True
    ),
    default="node_homo",
    help="The dataset to use. Options are homo and hetero",
)
@click.option("--dataset_name", default="CORA")
@click.option("--fold_idx", type=int, default=0)
@click.option("--num_layers", type=int, default=3)
@click.option("--hidden_dim", type=int, default=64)
@click.option("--graph_classification", type=bool, default=False)
@click.option("--batch_size", type=int, default=64)
@click.option("--with_sam", type=bool, default=True)
@click.option("--seed", type=int, default=1234)
@click.option("--use_wandb", type=bool, default=True)
@click.option("--neighbour_loader/--no_neighbour_loader", default=False)
@click.option("--num_neighbours", callback=process_items)
@click.option(
    "--base_optimizer",
    type=click.Choice(
        ["adam", "sgd"], case_sensitive=True
    ),
    default="adam",
    help="The base optimizer to use. Options are adam and sgd. Can be used in conjunction with SAM",
)

def main(
    dataset_type: str,
    dataset_name: str,
    fold_idx: int,
    num_layers: int,
    hidden_dim: int,
    graph_classification: bool,
    batch_size: int,
    with_sam: bool,
    seed: int,
    use_wandb: bool,
        base_optimizer: str,
        neighbour_loader: bool,
        num_neighbours: list[int]
):
    datamodule = get_dataset(dataset_type, dataset_name, fold_idx, batch_size,
                             neighbour_loader, num_neighbours)
    model = GCN(
        num_features=datamodule.num_features,
        num_classes=datamodule.num_classes,
        num_hidden=hidden_dim,
        num_hidden_layers=num_layers,
        graph_classification=graph_classification,
        with_sam=with_sam,
        seed=seed,
        base_optimizer=base_optimizer,
    )

    wandb_logger = None

    if use_wandb:
        wandb_logger = WandbLogger(project="setup_tests", entity="r252_bel", log_model=True)

    trainer = Trainer(
        limit_train_batches=100,
        accelerator='auto',
        max_epochs=300,
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
